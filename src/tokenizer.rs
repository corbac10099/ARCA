/// BPE tokenizer — byte-level, self-contained, no external dependencies.
///
/// Replaces the trivial byte=token mapping used in the original demo.
///
/// # Design
///
/// The tokenizer operates at the *byte* level (like GPT-2 / tiktoken):
///   - The base vocabulary is all 256 single-byte sequences.
///   - Merge rules are learned by iteratively finding the most frequent
///     adjacent pair and fusing it into a new token.
///   - `encode()` applies the learned merges greedily (left to right) and
///     maps each resulting piece to its vocabulary id.
///   - `decode()` maps ids back to bytes.
///
/// # Vocabulary layout
///
/// ids 0..=255           — single bytes (always present)
/// ids 256..vocab_size   — merge tokens in order of creation
///
/// # Usage in ARCA
///
/// The `bpe_embeddings` table in the encoder has `BPE_VOCAB_SIZE = 4096` rows.
/// Any id ≥ BPE_VOCAB_SIZE is masked to `id % BPE_VOCAB_SIZE` (handled in
/// encoder.rs) so the tokenizer vocabulary can be larger than the embedding
/// table without crashing.
///
/// # Persistence
///
/// `save_to_json` / `load_from_json` serialise the merge list so the trained
/// tokenizer can be reloaded alongside a `.sovereign` weight file.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A learned merge: fuse `(left, right)` into `result`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    pub left: u32,
    pub right: u32,
    pub result: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BpeTokenizer {
    /// Vocabulary: token_string → id.  Encoded as UTF-8 escaped bytes for JSON.
    pub vocab: HashMap<Vec<u8>, u32>,
    /// Ordered list of merge rules (applied left-to-right during encoding).
    pub merges: Vec<MergeRule>,
    /// Inverse map id → bytes.
    pub id_to_bytes: Vec<Vec<u8>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Training
// ─────────────────────────────────────────────────────────────────────────────

impl BpeTokenizer {
    /// Initialise with the 256-byte base vocabulary.
    pub fn new_base() -> Self {
        let mut vocab: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut id_to_bytes: Vec<Vec<u8>> = Vec::with_capacity(256);

        for b in 0u8..=255 {
            let key = vec![b];
            vocab.insert(key.clone(), b as u32);
            id_to_bytes.push(key);
        }

        BpeTokenizer { vocab, merges: Vec::new(), id_to_bytes }
    }

    /// Train BPE on `corpus` for `num_merges` steps.
    ///
    /// Each step:
    ///   1. Count all adjacent pairs in the current tokenisation.
    ///   2. Find the most frequent pair `(a, b)`.
    ///   3. Create a new token `ab` and record the merge rule.
    ///   4. Replace every occurrence of `(a, b)` in the corpus with `ab`.
    ///
    /// Returns a trained tokenizer.  `num_merges` is capped at
    /// `max_vocab_size - 256` so the final vocabulary never exceeds
    /// `max_vocab_size`.
    pub fn train(corpus: &[u8], num_merges: usize, max_vocab_size: usize) -> Self {
        let mut tok = Self::new_base();
        let effective_merges = num_merges.min(max_vocab_size.saturating_sub(256));

        // Start with byte-level tokenisation of the corpus.
        let mut ids: Vec<u32> = corpus.iter().map(|&b| b as u32).collect();

        for _ in 0..effective_merges {
            if ids.len() < 2 {
                break;
            }

            // --- Count pairs ---
            let pair_counts = ids
                .par_windows(2)
                .fold(
                    || HashMap::new(),
                    |mut acc: HashMap<(u32, u32), u32>, w| {
                        *acc.entry((w[0], w[1])).or_insert(0) += 1;
                        acc
                    },
                )
                .reduce(
                    || HashMap::new(),
                    |mut acc1, acc2| {
                        for (k, v) in acc2 {
                            *acc1.entry(k).or_insert(0) += v;
                        }
                        acc1
                    },
                );

            // --- Find best pair (tie-break: lowest id sum for determinism) ---
            let best = pair_counts
                .iter()
                .max_by_key(|&(&(a, b), &c)| (c, std::cmp::Reverse(a + b)));

            let (&(left, right), _) = match best {
                Some(p) => p,
                None => break,
            };

            // --- Create new token ---
            let new_id = tok.id_to_bytes.len() as u32;
            let mut new_bytes = tok.id_to_bytes[left as usize].clone();
            new_bytes.extend_from_slice(&tok.id_to_bytes[right as usize]);
            tok.vocab.insert(new_bytes.clone(), new_id);
            tok.id_to_bytes.push(new_bytes);
            tok.merges.push(MergeRule { left, right, result: new_id });

            // --- Apply merge in-place ---
            ids = apply_merge(&ids, left, right, new_id);
        }

        tok
    }

    // ─────────────────────────────────────────────────────────────────────
    // Encoding
    // ─────────────────────────────────────────────────────────────────────

    /// Encode a byte slice into a sequence of token ids.
    ///
    /// Applies merge rules in order (same order as training), which is
    /// equivalent to greedy left-to-right BPE.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        // Start with byte-level ids
        let mut ids: Vec<u32> = text.iter().map(|&b| b as u32).collect();

        // Apply each merge rule once in order
        for rule in &self.merges {
            if ids.len() < 2 {
                break;
            }
            ids = apply_merge(&ids, rule.left, rule.right, rule.result);
        }

        ids
    }

    /// Encode and align back to byte positions.
    ///
    /// Returns a Vec of length `text.len()` where each entry is the token id
    /// that *covers* that byte position.  This is what `encoder.rs` needs:
    /// `bpe_ids[t]` is the token active at byte position `t`.
    pub fn encode_aligned(&self, text: &[u8]) -> Vec<u32> {
        let token_ids = self.encode(text);

        // Expand tokens back to per-byte coverage
        let mut aligned = Vec::with_capacity(text.len());
        for id in &token_ids {
            let token_bytes = &self.id_to_bytes[*id as usize];
            for _ in 0..token_bytes.len() {
                aligned.push(*id);
            }
        }

        // Guard: if the tokeniser produced a different byte count (shouldn't
        // happen with a correct BPE, but be safe) fall back to byte-level.
        if aligned.len() != text.len() {
            return text.iter().map(|&b| b as u32).collect();
        }

        aligned
    }

    // ─────────────────────────────────────────────────────────────────────
    // Decoding
    // ─────────────────────────────────────────────────────────────────────

    /// Decode a sequence of token ids back to bytes.
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let idx = (id as usize).min(self.id_to_bytes.len() - 1);
            out.extend_from_slice(&self.id_to_bytes[idx]);
        }
        out
    }

    /// Vocabulary size (256 base + number of merges performed).
    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Persistence
    // ─────────────────────────────────────────────────────────────────────

    /// Save tokenizer to a JSON file.
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        // Convert HashMap<Vec<u8>, u32> to a JSON-friendly Vec<(String, u32)>
        // using hex-escaped byte strings for binary safety.
        let vocab_ser: Vec<(String, u32)> = self
            .vocab
            .iter()
            .map(|(k, &v)| {
                let hex: String = k.iter().map(|b| format!("{:02x}", b)).collect();
                (hex, v)
            })
            .collect();

        let id_to_bytes_ser: Vec<String> = self
            .id_to_bytes
            .iter()
            .map(|b| b.iter().map(|x| format!("{:02x}", x)).collect())
            .collect();

        let payload = serde_json::json!({
            "vocab": vocab_ser,
            "merges": self.merges,
            "id_to_bytes": id_to_bytes_ser,
        });

        let json = serde_json::to_string_pretty(&payload).unwrap();
        std::fs::write(path, json)
    }

    /// Load tokenizer from a JSON file previously created by `save_to_json`.
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let merges: Vec<MergeRule> = serde_json::from_value(v["merges"].clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let id_to_bytes_ser: Vec<String> = serde_json::from_value(v["id_to_bytes"].clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let id_to_bytes: Vec<Vec<u8>> = id_to_bytes_ser
            .iter()
            .map(|s| {
                (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap_or(0))
                    .collect()
            })
            .collect();

        let vocab_ser: Vec<(String, u32)> = serde_json::from_value(v["vocab"].clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let vocab: HashMap<Vec<u8>, u32> = vocab_ser
            .into_iter()
            .map(|(hex, id)| {
                let bytes: Vec<u8> = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0))
                    .collect();
                (bytes, id)
            })
            .collect();

        Ok(BpeTokenizer { vocab, merges, id_to_bytes })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helper
// ─────────────────────────────────────────────────────────────────────────────

/// Replace every occurrence of consecutive `(left, right)` in `ids` with
/// `result`.  Single pass, O(n).
fn apply_merge(ids: &[u32], left: u32, right: u32, result: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
            out.push(result);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_vocab_size() {
        let tok = BpeTokenizer::new_base();
        assert_eq!(tok.vocab_size(), 256);
    }

    #[test]
    fn roundtrip_no_merges() {
        let tok = BpeTokenizer::new_base();
        let text = b"Hello, ARCA!";
        let ids = tok.encode(text);
        let back = tok.decode(&ids);
        assert_eq!(back, text);
    }

    #[test]
    fn train_reduces_sequence_length() {
        let corpus = b"aaabdaaabac";
        let tok = BpeTokenizer::train(corpus, 10, 300);
        let encoded = tok.encode(corpus);
        // After merging common pairs the sequence must be shorter than byte-level
        assert!(encoded.len() < corpus.len(), "BPE should compress repeating bytes");
    }

    #[test]
    fn roundtrip_after_training() {
        let corpus = b"the quick brown fox jumps over the lazy dog";
        let tok = BpeTokenizer::train(corpus, 20, 300);
        let ids = tok.encode(corpus);
        let back = tok.decode(&ids);
        assert_eq!(back, corpus);
    }

    #[test]
    fn aligned_encode_length_matches() {
        let corpus = b"hello world";
        let tok = BpeTokenizer::train(corpus, 5, 300);
        let aligned = tok.encode_aligned(corpus);
        assert_eq!(aligned.len(), corpus.len());
    }

    #[test]
    fn save_load_roundtrip() {
        let corpus = b"abracadabra";
        let tok = BpeTokenizer::train(corpus, 5, 300);
        let path = std::env::temp_dir().join("test_tokenizer.json");
        tok.save_to_json(&path).unwrap();
        let tok2 = BpeTokenizer::load_from_json(&path).unwrap();
        assert_eq!(tok.merges, tok2.merges);
        assert_eq!(tok.encode(corpus), tok2.encode(corpus));
    }
}
