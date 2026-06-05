/// Tokenless multi-scale encoder.
///
/// Produces x_t ∈ ℝ^512 from raw bytes at position t by fusing three sub-projections:
///   x_t = W_fusion · [ e_bytes(t) ‖ e_bpe(t) ‖ e_phrase(t) ]
///
/// Projections:
///   e_bytes  ∈ ℝ^128  — n-gram byte embeddings (n ∈ {1,2,3})
///   e_bpe    ∈ ℝ^256  — anchor BPE vocabulary embedding
///   e_phrase ∈ ℝ^128  — causal convolution over sliding window of 4–8 tokens
///
/// W_fusion ∈ ℝ^{512×512} is a learned online projection matrix.

use ndarray::{Array1, Array2, ArrayView1, s};

/// Minimal BPE anchor vocabulary size (hard-coded to keep the encoder self-contained).
pub const BPE_VOCAB_SIZE: usize = 4096;
pub const D_BYTES: usize = 128;
pub const D_BPE: usize = 256;
pub const D_PHRASE: usize = 128;
pub const D_CONCAT: usize = D_BYTES + D_BPE + D_PHRASE; // 512
pub const D_MODEL: usize = 512;

/// Window sizes used for the causal phrase convolution.
pub const PHRASE_WIN_MIN: usize = 4;
const PHRASE_WIN_MAX: usize = 8;

/// Number of 1-gram, 2-gram, 3-gram byte hash buckets mapped into D_BYTES.
const NGRAM_BUCKETS: usize = D_BYTES; // 128

pub struct MultiScaleEncoder {
    /// BPE embedding table: shape [BPE_VOCAB_SIZE, D_BPE]
    pub bpe_embeddings: Array2<f32>,
    /// Fusion projection: shape [D_MODEL, D_CONCAT]
    pub w_fusion: Array2<f32>,
    /// Phrase convolution kernel: shape [D_PHRASE, D_BPE * (PHRASE_WIN_MAX - PHRASE_WIN_MIN + 1)]
    /// We use a single 1-D causal conv with multiple dilation windows; approximated as a linear
    /// projection from the concatenated window embeddings.
    pub w_phrase: Array2<f32>,
    /// Phrase kernel width (number of BPE tokens in the sliding window)
    phrase_window: usize,
}

impl MultiScaleEncoder {
    pub fn new(
        bpe_embeddings: Array2<f32>,
        w_fusion: Array2<f32>,
        w_phrase: Array2<f32>,
    ) -> Self {
        assert_eq!(bpe_embeddings.shape(), &[BPE_VOCAB_SIZE, D_BPE]);
        assert_eq!(w_fusion.shape(), &[D_MODEL, D_CONCAT]);
        // w_phrase: [D_PHRASE, phrase_window * D_BPE]
        let phrase_window = w_phrase.shape()[1] / D_BPE;
        assert!(phrase_window >= PHRASE_WIN_MIN && phrase_window <= PHRASE_WIN_MAX);
        MultiScaleEncoder { bpe_embeddings, w_fusion, w_phrase, phrase_window }
    }

    // ------------------------------------------------------------------
    // Public entry-point: encode a single position in context.
    //
    // Parameters:
    //   raw_bytes   — full byte stream up to and including position t
    //   t           — current position index (0-based)
    //   bpe_ids     — pre-tokenized BPE token ids, length = T (same positions)
    //
    // Returns x_t ∈ ℝ^{D_MODEL=512}
    // ------------------------------------------------------------------
    pub fn encode_position(
        &self,
        raw_bytes: &[u8],
        t: usize,
        bpe_ids: &[u32],
    ) -> Array1<f32> {
        debug_assert!(t < raw_bytes.len(), "position t out of bounds");
        debug_assert_eq!(raw_bytes.len(), bpe_ids.len(), "bpe_ids must be same length as raw_bytes");

        let e_bytes = self.compute_byte_ngrams(raw_bytes, t);
        let e_bpe = self.lookup_bpe(bpe_ids, t);
        let e_phrase = self.causal_phrase_conv(bpe_ids, t);

        // Concatenate sub-projections into ℝ^512
        let mut concat = Array1::<f32>::zeros(D_CONCAT);
        concat.slice_mut(s![0..D_BYTES]).assign(&e_bytes);
        concat.slice_mut(s![D_BYTES..D_BYTES + D_BPE]).assign(&e_bpe);
        concat.slice_mut(s![D_BYTES + D_BPE..D_CONCAT]).assign(&e_phrase);

        // x_t = W_fusion · concat
        let x_t = self.w_fusion.dot(&concat);
        x_t
    }

    // ------------------------------------------------------------------
    // e_bytes(t) ∈ ℝ^128
    //
    // Hash-based projection of all n-grams (n=1..3) ending at position t
    // into a fixed-size accumulator, then L2-normalised.
    // ------------------------------------------------------------------
    fn compute_byte_ngrams(&self, bytes: &[u8], t: usize) -> Array1<f32> {
        let mut acc = Array1::<f32>::zeros(D_BYTES);

        for n in 1usize..=3 {
            if t + 1 < n {
                continue; // not enough history for this n-gram size
            }
            let start = t + 1 - n;
            let gram = &bytes[start..=t];
            let bucket = ngram_hash(gram) % NGRAM_BUCKETS;
            // Weight n-gram by 1/n to discount longer, sparser grams
            acc[bucket] += 1.0 / n as f32;
        }

        l2_normalize_inplace(&mut acc);
        acc
    }

    // ------------------------------------------------------------------
    // e_bpe(t) ∈ ℝ^256 — direct embedding table lookup
    // ------------------------------------------------------------------
    fn lookup_bpe(&self, bpe_ids: &[u32], t: usize) -> Array1<f32> {
        let id = (bpe_ids[t] as usize) % BPE_VOCAB_SIZE;
        self.bpe_embeddings.row(id).to_owned()
    }

    // ------------------------------------------------------------------
    // e_phrase(t) ∈ ℝ^128
    //
    // Causal convolution: gather the last `phrase_window` BPE embeddings
    // (zero-padded left), flatten, then project with W_phrase.
    // ------------------------------------------------------------------
    fn causal_phrase_conv(&self, bpe_ids: &[u32], t: usize) -> Array1<f32> {
        let win = self.phrase_window;
        let mut window_vec = Array1::<f32>::zeros(win * D_BPE);

        for k in 0..win {
            // positions t-win+1+k … t  (causal, no future look-ahead)
            let pos_signed = t as isize - (win as isize - 1) + k as isize;
            if pos_signed < 0 {
                // zero pad
                continue;
            }
            let pos = pos_signed as usize;
            let id = (bpe_ids[pos] as usize) % BPE_VOCAB_SIZE;
            let emb = self.bpe_embeddings.row(id);
            let base = k * D_BPE;
            window_vec.slice_mut(s![base..base + D_BPE]).assign(&emb);
        }

        self.w_phrase.dot(&window_vec)
    }
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

/// FNV-1a-style byte hash used for n-gram bucketing (no heap allocation).
#[inline]
fn ngram_hash(gram: &[u8]) -> usize {
    const BASIS: usize = 14_695_981_039_346_656_037;
    const PRIME: usize = 1_099_511_628_211;
    let mut h = BASIS;
    for &b in gram {
        h ^= b as usize;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// In-place L2 normalisation (leaves vector unchanged if norm < epsilon).
#[inline]
fn l2_normalize_inplace(v: &mut Array1<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.mapv_inplace(|x| x / norm);
    }
}

// ------------------------------------------------------------------
// Unit test shim
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_encoder() -> MultiScaleEncoder {
        let bpe = Array2::<f32>::from_elem((BPE_VOCAB_SIZE, D_BPE), 0.01_f32);
        let w_fusion = Array2::<f32>::eye(D_MODEL); // identity for shape test
        // Phrase window = 4 → w_phrase: [128, 4*256] = [128, 1024]
        let w_phrase = Array2::<f32>::zeros((D_PHRASE, PHRASE_WIN_MIN * D_BPE));
        MultiScaleEncoder::new(bpe, w_fusion, w_phrase)
    }

    #[test]
    fn output_shape() {
        let enc = make_encoder();
        let bytes = b"Hello, ARCA system!";
        let bpe_ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
        let x = enc.encode_position(bytes, 5, &bpe_ids);
        assert_eq!(x.len(), D_MODEL);
    }
}
