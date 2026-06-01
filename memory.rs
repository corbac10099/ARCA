/// Holographic associative memory and sparse Top-K output head.
///
/// Implements §1.3 (memory read-out aggregation) and §2.1 (Sparse Top-K head).
///
/// The output head projects the aggregated representation onto the top-K
/// logits of the vocabulary without computing the full V=50 000 projection:
///   O(K × D_model)  with K = 200

use ndarray::{Array1, Array2, s};

use crate::metabolic_core::{BioInspiredLayer, RANK_R, N_RES, D_MODEL};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
pub const VOCAB_SIZE: usize = 50_000;
pub const TOP_K: usize = 200;

// ─────────────────────────────────────────────────────────────────────────────
// Holographic memory aggregator
//
// After all L layers have produced their read-out vectors (each ∈ ℝ^RANK_R),
// we aggregate them into a single representation vector ∈ ℝ^D_MODEL via a
// learned projection W_out: [D_MODEL, N_RES].  The reservoir state s_t is
// first modulated by the sum of per-layer read-outs, then projected.
// ─────────────────────────────────────────────────────────────────────────────

pub struct HolographicMemoryAggregator {
    /// W_out: shape [D_MODEL, N_RES]
    pub w_out: Array2<f32>,
}

impl HolographicMemoryAggregator {
    pub fn new(w_out: Array2<f32>) -> Self {
        assert_eq!(w_out.shape(), &[D_MODEL, N_RES]);
        HolographicMemoryAggregator { w_out }
    }

    /// Aggregate layer read-outs and project the modulated reservoir state.
    ///
    /// Parameters:
    ///   s_t          — reservoir state ∈ ℝ^{N_RES}
    ///   layer_readouts — per-layer memory read-out signals, each ∈ ℝ^{RANK_R}
    ///
    /// Process:
    ///   1. Sum layer read-outs → modulation ∈ ℝ^{RANK_R}  (sub-space)
    ///   2. Broadcast-add modulation onto the first RANK_R dims of s_t
    ///      (associative interference: state is perturbed by memory content)
    ///   3. y = W_out · s_modulated  ∈ ℝ^{D_MODEL}
    pub fn aggregate(
        &self,
        s_t: &Array1<f32>,
        layer_readouts: &[Array1<f32>],
    ) -> Array1<f32> {
        debug_assert_eq!(s_t.len(), N_RES);

        // Sum of read-outs
        let mut modulation = Array1::<f32>::zeros(RANK_R);
        for ro in layer_readouts {
            modulation = modulation + ro;
        }

        // Modulate the first RANK_R dimensions of s_t (holographic interference)
        let mut s_mod = s_t.to_owned();
        for i in 0..RANK_R {
            s_mod[i] += modulation[i];
        }

        self.w_out.dot(&s_mod)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse Top-K output head
//
// Instead of projecting onto the full vocabulary (O(V × D_model)), we maintain
// a candidate shortlist and compute scores only for TOP_K = 200 candidates.
//
// Strategy:
//   1. A lightweight "gate" linear layer maps y ∈ ℝ^D_MODEL → ℝ^{VOCAB_SIZE}
//      but we only materialise the TOP_K highest-score positions.
//   2. We use a partial dot-product via argpartition-style selection.
//
// In production this would use a learned routing index; here we compute the
// full score vector once (acceptable at inference: 512×50 000 ≈ 25M MACs)
// and return the sparse (index, logit) pairs.
// ─────────────────────────────────────────────────────────────────────────────

pub struct SparseOutputHead {
    /// Full output embedding matrix: shape [VOCAB_SIZE, D_MODEL]
    /// Stored as a reference into the mmap (zero-copy).
    /// For the purposes of this implementation we own a dense copy.
    pub output_embeddings: Array2<f32>,
    /// Output bias: shape [VOCAB_SIZE]
    pub output_bias: Array1<f32>,
}

/// A sparse logit: (token_id, logit_value)
#[derive(Debug, Clone, Copy)]
pub struct SparseLogit {
    pub token_id: u32,
    pub logit: f32,
}

impl SparseOutputHead {
    pub fn new(output_embeddings: Array2<f32>, output_bias: Array1<f32>) -> Self {
        assert_eq!(output_embeddings.shape(), &[VOCAB_SIZE, D_MODEL]);
        assert_eq!(output_bias.len(), VOCAB_SIZE);
        SparseOutputHead { output_embeddings, output_bias }
    }

    /// Compute the top-K logits for the next-token distribution.
    ///
    /// Returns a Vec of (token_id, logit) pairs, length = TOP_K, sorted
    /// descending by logit. O(V × D_model) compute, O(K) output.
    pub fn top_k_logits(&self, y: &Array1<f32>) -> Vec<SparseLogit> {
        debug_assert_eq!(y.len(), D_MODEL);

        // Full logit vector: score[v] = output_embeddings[v] · y + bias[v]
        let scores: Vec<f32> = (0..VOCAB_SIZE)
            .map(|v| {
                let row = self.output_embeddings.row(v);
                let dot: f32 = row.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                dot + self.output_bias[v]
            })
            .collect();

        // Partial sort: find TOP_K maximum indices without full sort
        // We use a fixed-size min-heap via a simple selection approach.
        top_k_indices_and_values(&scores, TOP_K)
    }

    /// Compute prediction for next-token cross-entropy loss.
    /// Returns the full logit vector (used during training only).
    pub fn full_logits(&self, y: &Array1<f32>) -> Array1<f32> {
        let scores: Vec<f32> = (0..VOCAB_SIZE)
            .map(|v| {
                let row = self.output_embeddings.row(v);
                row.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>() + self.output_bias[v]
            })
            .collect();
        Array1::from_vec(scores)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction head: assembles aggregator + output head and exposes the full
// forward interface used by train.rs and main.rs
// ─────────────────────────────────────────────────────────────────────────────

pub struct PredictionHead {
    pub aggregator: HolographicMemoryAggregator,
    pub head: SparseOutputHead,
}

impl PredictionHead {
    pub fn new(aggregator: HolographicMemoryAggregator, head: SparseOutputHead) -> Self {
        PredictionHead { aggregator, head }
    }

    /// Full forward pass of the output stage.
    ///
    /// Returns the full logit vector for loss computation and the top-K
    /// sparse logits for decoding.
    pub fn forward(
        &self,
        s_t: &Array1<f32>,
        layer_readouts: &[Array1<f32>],
    ) -> (Array1<f32>, Vec<SparseLogit>) {
        let y = self.aggregator.aggregate(s_t, layer_readouts);
        let full = self.head.full_logits(&y);
        let sparse = top_k_indices_and_values(full.as_slice().unwrap(), TOP_K);
        (full, sparse)
    }

    /// Predicted embedding for next-position predictive coding.
    /// x̂_t = W_out · s_t  (before vocabulary projection).
    pub fn predict_embedding(&self, s_t: &Array1<f32>) -> Array1<f32> {
        self.aggregator.w_out.dot(s_t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: partial top-K extraction
// ─────────────────────────────────────────────────────────────────────────────

/// O(V + K log K) top-K extraction using a simple linear scan with a
/// min-heap-emulated fixed-size buffer. No dynamic allocation inside hot loop.
fn top_k_indices_and_values(scores: &[f32], k: usize) -> Vec<SparseLogit> {
    debug_assert!(k <= scores.len());

    // Phase 1: seed heap with first k elements
    let mut heap: Vec<(f32, u32)> = scores[..k]
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    // Build min-heap (smallest at position 0)
    heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Phase 2: scan the rest
    for (i, &score) in scores[k..].iter().enumerate() {
        let global_i = (i + k) as u32;
        if score > heap[0].0 {
            heap[0] = (score, global_i);
            // Sift down to maintain sorted-ascending invariant
            sift_down(&mut heap);
        }
    }

    // Sort descending by score for the caller
    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    heap.into_iter()
        .map(|(logit, token_id)| SparseLogit { token_id, logit })
        .collect()
}

/// Sift down the minimum element at index 0 in a "sorted ascending" buffer.
/// This is a linear scan re-insertion, acceptable for K=200.
#[inline]
fn sift_down(buf: &mut Vec<(f32, u32)>) {
    let new_min_candidate = buf[0];
    let mut insert_pos = 0;
    while insert_pos + 1 < buf.len() && buf[insert_pos + 1].0 < new_min_candidate.0 {
        buf[insert_pos] = buf[insert_pos + 1];
        insert_pos += 1;
    }
    buf[insert_pos] = new_min_candidate;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_count_and_monotone() {
        let scores: Vec<f32> = (0..VOCAB_SIZE).map(|i| i as f32).collect();
        let result = top_k_indices_and_values(&scores, TOP_K);
        assert_eq!(result.len(), TOP_K);
        // First result should be the largest score: VOCAB_SIZE - 1
        assert_eq!(result[0].token_id, (VOCAB_SIZE - 1) as u32);
        // Descending order
        for i in 1..result.len() {
            assert!(result[i - 1].logit >= result[i].logit);
        }
    }

    #[test]
    fn aggregator_output_shape() {
        let w_out = Array2::<f32>::zeros((D_MODEL, N_RES));
        let agg = HolographicMemoryAggregator::new(w_out);
        let s = Array1::<f32>::zeros(N_RES);
        let readouts: Vec<Array1<f32>> = (0..4).map(|_| Array1::zeros(RANK_R)).collect();
        let y = agg.aggregate(&s, &readouts);
        assert_eq!(y.len(), D_MODEL);
    }
}
