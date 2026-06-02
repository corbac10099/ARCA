/// Holographic associative memory and GPU-accelerated sparse Top-K output head.
///
/// # Module structure
///
/// ```text
/// HolographicMemoryAggregator   — CPU-only (small matrices, fast)
/// SparseOutputHead              — GPU GEMV + CPU top-K selection
/// PredictionHead                — composes aggregator + output head
/// ```
///
/// # GPU offload strategy
///
/// `SparseOutputHead::top_k_logits` previously ran a 50 000×512 dot-product
/// loop on the CPU (O(V × D_model) = 25 M MACs per step).  This has been
/// replaced by:
///
///   1. `GpuContext::upload_y_hidden`  — 2 KiB PCIe upload.
///   2. `GpuContext::dispatch_logits`  — runs `logit_compute.wgsl` which
///      computes all 50 000 logits in parallel on the GPU (workgroup 256 =
///      4 RDNA wavefronts per CU).
///   3. `GpuContext::readback_logits`  — 200 KiB PCIe download (negligible
///      compared to the 25 M MACs saved).
///   4. `top_k_indices_and_values`     — the existing O(V + K log K) CPU
///      sort still runs, but on a flat `Vec<f32>` already in L1/L2.
///
/// The output embedding table (50 000 × 512 ≈ 97 MiB) and bias (200 KB)
/// live permanently in GPU VRAM.  The CPU uploads only `y_hidden` (2 KB)
/// per step.
///
/// # Interaction with GpuContext
///
/// `PredictionHead::forward_gpu` accepts a `&mut GpuContext` reference,
/// uploads `y_hidden`, dispatches the logit shader, reads back, and then
/// calls the CPU top-K routine.  The full CPU path (`forward`) is preserved
/// for the no-GPU build and benchmarking.
///
/// # Feature flag
///
/// All GPU paths are gated on `#[cfg(feature = "gpu")]`.

use ndarray::{Array1, Array2};

use crate::metabolic_core::{RANK_R, N_RES, D_MODEL};

#[cfg(feature = "gpu")]
use crate::gpu_context::GpuContext;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
pub const VOCAB_SIZE: usize = 50_000;
pub const TOP_K:      usize = 200;

// ─────────────────────────────────────────────────────────────────────────────
// Holographic memory aggregator (CPU — small matrices)
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates per-layer memory read-outs into a single representation vector
/// `y ∈ ℝ^{D_MODEL}` via:
///
///   1. Sum layer read-outs → `modulation ∈ ℝ^{RANK_R}`.
///   2. Broadcast-add modulation onto the first `RANK_R` dims of `s_t`
///      (holographic interference in the reservoir sub-space).
///   3. `y = W_out · s_modulated` ∈ ℝ^{D_MODEL}.
///
/// W_out is [D_MODEL × N_RES] — the largest CPU-side matrix in the pipeline,
/// but it is invoked only once per step at the aggregation stage.
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
    /// Parameters
    /// ----------
    /// - `s_t_cpu`       — CPU shadow of the reservoir state ∈ ℝ^{N_RES}.
    ///   (Obtained from `GpuContext::readback_s` in the GPU path.)
    /// - `layer_readouts` — per-layer read-out signals, each ∈ ℝ^{RANK_R}.
    ///
    /// Returns `y ∈ ℝ^{D_MODEL}`.
    pub fn aggregate(
        &self,
        s_t_cpu:       &Array1<f32>,
        layer_readouts: &[Array1<f32>],
    ) -> Array1<f32> {
        debug_assert_eq!(s_t_cpu.len(), N_RES);

        let mut modulation = Array1::<f32>::zeros(RANK_R);
        for ro in layer_readouts {
            debug_assert_eq!(ro.len(), RANK_R);
            modulation = modulation + ro;
        }

        // Modulate the first RANK_R dimensions of s_t (holographic interference)
        let mut s_mod = s_t_cpu.to_owned();
        for i in 0..RANK_R {
            s_mod[i] += modulation[i];
        }

        self.w_out.dot(&s_mod)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse Top-K output head
// ─────────────────────────────────────────────────────────────────────────────

/// Vocabulary output head.
///
/// Holds CPU copies of the embedding table and bias for serialisation and the
/// CPU fallback path.  In the GPU path these matrices live in VRAM; the CPU
/// copies are only written to disk at checkpoint time.
pub struct SparseOutputHead {
    /// Full output embedding matrix: shape [VOCAB_SIZE, D_MODEL].
    /// CPU copy retained for serialisation; hot path uses the GPU buffer.
    pub output_embeddings: Array2<f32>,
    /// Output bias: shape [VOCAB_SIZE].
    pub output_bias: Array1<f32>,
}

/// A sparse logit: `(token_id, logit_value)`.
#[derive(Debug, Clone, Copy)]
pub struct SparseLogit {
    pub token_id: u32,
    pub logit:    f32,
}

impl SparseOutputHead {
    pub fn new(output_embeddings: Array2<f32>, output_bias: Array1<f32>) -> Self {
        assert_eq!(output_embeddings.shape(), &[VOCAB_SIZE, D_MODEL]);
        assert_eq!(output_bias.len(), VOCAB_SIZE);
        SparseOutputHead { output_embeddings, output_bias }
    }

    // ── GPU path ──────────────────────────────────────────────────────────

    /// GPU-accelerated top-K logit computation.
    ///
    /// 1. Upload `y` (2 KiB) to VRAM via `gpu.upload_y_hidden`.
    /// 2. Dispatch the logit GEMV shader (50 000 × 512 parallel).
    /// 3. Read back the 200 KiB logit vector.
    /// 4. Select top-K on CPU (O(V + K log K)).
    ///
    /// PCIe per call: **2 KiB up + 200 KiB down**.
    /// GPU wall-time: < 0.5 ms on a mid-range AMD RDNA2+ dGPU.
    #[cfg(feature = "gpu")]
    pub fn top_k_logits_gpu(
        &self,
        gpu: &mut GpuContext,
        y:   &Array1<f32>,
    ) -> Vec<SparseLogit> {
        debug_assert_eq!(y.len(), D_MODEL);
        let y_flat: Vec<f32> = y.iter().cloned().collect();
        gpu.upload_y_hidden(&y_flat);
        gpu.dispatch_logits();
        let scores = gpu.readback_logits();
        top_k_indices_and_values(&scores, TOP_K)
    }

    /// GPU-accelerated full logit vector (for training cross-entropy loss).
    ///
    /// Same path as `top_k_logits_gpu` but returns the complete 50 000-element
    /// vector instead of just the top-K.
    #[cfg(feature = "gpu")]
    pub fn full_logits_gpu(
        &self,
        gpu: &mut GpuContext,
        y:   &Array1<f32>,
    ) -> Array1<f32> {
        debug_assert_eq!(y.len(), D_MODEL);
        let y_flat: Vec<f32> = y.iter().cloned().collect();
        gpu.upload_y_hidden(&y_flat);
        gpu.dispatch_logits();
        Array1::from_vec(gpu.readback_logits())
    }

    // ── CPU fallback path ─────────────────────────────────────────────────

    /// CPU top-K logit computation — O(V × D_model) MACs.
    ///
    /// Used when the `gpu` feature is disabled or for benchmark comparisons.
    pub fn top_k_logits(&self, y: &Array1<f32>) -> Vec<SparseLogit> {
        debug_assert_eq!(y.len(), D_MODEL);
        let scores = self.compute_scores_cpu(y);
        top_k_indices_and_values(&scores, TOP_K)
    }

    /// CPU full logit vector.
    pub fn full_logits(&self, y: &Array1<f32>) -> Array1<f32> {
        Array1::from_vec(self.compute_scores_cpu(y))
    }

    /// Inner loop: `scores[v] = output_embeddings[v,:] · y + bias[v]`.
    #[inline]
    fn compute_scores_cpu(&self, y: &Array1<f32>) -> Vec<f32> {
        (0..VOCAB_SIZE)
            .map(|v| {
                let row = self.output_embeddings.row(v);
                let dot: f32 = row.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                dot + self.output_bias[v]
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction head — aggregator + output head
// ─────────────────────────────────────────────────────────────────────────────

/// Composes `HolographicMemoryAggregator` and `SparseOutputHead` into the
/// unified output interface used by `main.rs` and `train.rs`.
pub struct PredictionHead {
    pub aggregator: HolographicMemoryAggregator,
    pub head:       SparseOutputHead,
}

impl PredictionHead {
    pub fn new(aggregator: HolographicMemoryAggregator, head: SparseOutputHead) -> Self {
        PredictionHead { aggregator, head }
    }

    // ── GPU path ──────────────────────────────────────────────────────────

    /// GPU-accelerated full forward pass.
    ///
    /// Parameters
    /// ----------
    /// - `gpu`            — mutable GPU context.
    /// - `s_t_cpu`        — CPU shadow of the reservoir state (from `GpuContext::readback_s`).
    /// - `layer_readouts` — per-layer `BioInspiredLayer::read_out_cpu` results.
    ///
    /// Returns `(full_logits, sparse_top_k)`.
    ///
    /// PCIe traffic: 2 KiB up (y_hidden) + 200 KiB down (logits) per call.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        gpu:            &mut GpuContext,
        s_t_cpu:        &Array1<f32>,
        layer_readouts: &[Array1<f32>],
    ) -> (Array1<f32>, Vec<SparseLogit>) {
        let y    = self.aggregator.aggregate(s_t_cpu, layer_readouts);
        let full = self.head.full_logits_gpu(gpu, &y);
        let sparse = top_k_indices_and_values(
            full.as_slice().expect("full_logits_gpu: non-contiguous Array1"),
            TOP_K,
        );
        (full, sparse)
    }

    // ── CPU fallback ──────────────────────────────────────────────────────

    /// CPU forward pass (original, unchanged).
    ///
    /// Returns `(full_logits, sparse_top_k)`.
    pub fn forward(
        &self,
        s_t:            &Array1<f32>,
        layer_readouts: &[Array1<f32>],
    ) -> (Array1<f32>, Vec<SparseLogit>) {
        let y      = self.aggregator.aggregate(s_t, layer_readouts);
        let full   = self.head.full_logits(&y);
        let sparse = top_k_indices_and_values(
            full.as_slice().expect("full_logits: non-contiguous Array1"),
            TOP_K,
        );
        (full, sparse)
    }

    /// Predicted embedding for next-position predictive coding.
    ///   x̂_t = W_out · s_t  (before vocabulary projection)
    pub fn predict_embedding(&self, s_t: &Array1<f32>) -> Array1<f32> {
        self.aggregator.w_out.dot(s_t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: O(V + K log K) partial top-K extraction (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────

/// O(V + K log K) top-K extraction.
///
/// Works on a plain `&[f32]` — the same logic runs whether the scores came
/// from the GPU readback or the CPU fallback, so no code duplication.
///
/// Algorithm
/// ---------
/// Phase 1: Seed a fixed-size min-heap with the first `k` scores.
/// Phase 2: Linear scan of the remaining V − k scores; replace the minimum
///          when a larger value is found.
/// Phase 3: Sort the heap descending.
///
/// The min-heap is kept in sorted-ascending order so that index 0 always
/// holds the current minimum (fast replacement without a full heap).  For
/// K = 200 this is more cache-friendly than a proper binary-heap.
fn top_k_indices_and_values(scores: &[f32], k: usize) -> Vec<SparseLogit> {
    debug_assert!(k > 0);
    debug_assert!(k <= scores.len());

    // Phase 1: seed heap with first k elements (sorted ascending, min at [0])
    let mut heap: Vec<(f32, u32)> = scores[..k]
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Phase 2: scan the rest
    for (i, &score) in scores[k..].iter().enumerate() {
        if score > heap[0].0 {
            heap[0] = (score, (i + k) as u32);
            sift_down(&mut heap);
        }
    }

    // Phase 3: sort descending for the caller
    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    heap.into_iter()
        .map(|(logit, token_id)| SparseLogit { token_id, logit })
        .collect()
}

/// Re-insert the element at index 0 into a sorted-ascending `buf` by
/// shifting smaller elements left until the correct position is found.
/// O(K) but K = 200, so this is negligible.
#[inline]
fn sift_down(buf: &mut Vec<(f32, u32)>) {
    let candidate = buf[0];
    let mut pos = 0;
    while pos + 1 < buf.len()
        && buf[pos + 1].0 < candidate.0
    {
        buf[pos] = buf[pos + 1];
        pos += 1;
    }
    buf[pos] = candidate;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── top-K correctness ─────────────────────────────────────────────────

    #[test]
    fn top_k_count_and_monotone_ascending_input() {
        // scores = [0, 1, 2, ..., VOCAB_SIZE-1]
        let scores: Vec<f32> = (0..VOCAB_SIZE).map(|i| i as f32).collect();
        let result = top_k_indices_and_values(&scores, TOP_K);
        assert_eq!(result.len(), TOP_K, "must return exactly TOP_K entries");
        // Largest score is VOCAB_SIZE-1
        assert_eq!(result[0].token_id, (VOCAB_SIZE - 1) as u32);
        // Descending order
        for i in 1..result.len() {
            assert!(
                result[i - 1].logit >= result[i].logit,
                "not descending at index {}", i
            );
        }
    }

    #[test]
    fn top_k_count_and_monotone_descending_input() {
        // scores = [VOCAB_SIZE-1, VOCAB_SIZE-2, ..., 0]
        let scores: Vec<f32> = (0..VOCAB_SIZE).rev().map(|i| i as f32).collect();
        let result = top_k_indices_and_values(&scores, TOP_K);
        assert_eq!(result.len(), TOP_K);
        // Token 0 has the highest score (VOCAB_SIZE-1)
        assert_eq!(result[0].token_id, 0u32);
    }

    #[test]
    fn top_k_all_equal_scores() {
        let scores = vec![1.0f32; VOCAB_SIZE];
        let result = top_k_indices_and_values(&scores, TOP_K);
        assert_eq!(result.len(), TOP_K);
        for sl in &result {
            assert!((sl.logit - 1.0).abs() < 1e-6);
        }
    }

    // ── Aggregator ────────────────────────────────────────────────────────

    #[test]
    fn aggregator_output_shape_zero_input() {
        let w_out = Array2::<f32>::zeros((D_MODEL, N_RES));
        let agg   = HolographicMemoryAggregator::new(w_out);
        let s     = Array1::<f32>::zeros(N_RES);
        let readouts: Vec<Array1<f32>> =
            (0..4).map(|_| Array1::zeros(RANK_R)).collect();
        let y = agg.aggregate(&s, &readouts);
        assert_eq!(y.len(), D_MODEL);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn aggregator_modulation_affects_output() {
        use ndarray::Array2;
        // W_out = identity (first D_MODEL rows of an N_RES-wide identity)
        // Use a [D_MODEL, N_RES] matrix where W_out[i,i] = 1
        let mut w_out = Array2::<f32>::zeros((D_MODEL, N_RES));
        for i in 0..D_MODEL {
            w_out[[i, i]] = 1.0;
        }
        let agg = HolographicMemoryAggregator::new(w_out);

        let s = Array1::<f32>::zeros(N_RES);
        // Single layer read-out: e_0 = 1.0, rest = 0
        let mut ro = Array1::<f32>::zeros(RANK_R);
        ro[0] = 1.0;
        let readouts = vec![ro];
        let y = agg.aggregate(&s, &readouts);
        // After modulation s_mod[0] = 1.0; W_out[0,0]=1 → y[0]=1.0
        assert!((y[0] - 1.0).abs() < 1e-6, "y[0]={}", y[0]);
    }

    // ── SparseOutputHead CPU path ─────────────────────────────────────────

    #[test]
    fn sparse_head_cpu_top_k_shape() {
        let emb  = Array2::<f32>::zeros((VOCAB_SIZE, D_MODEL));
        let bias = Array1::<f32>::zeros(VOCAB_SIZE);
        let head = SparseOutputHead::new(emb, bias);
        let y    = Array1::<f32>::zeros(D_MODEL);
        let top  = head.top_k_logits(&y);
        assert_eq!(top.len(), TOP_K);
    }

    // ── PredictionHead CPU path ───────────────────────────────────────────

    #[test]
    fn prediction_head_forward_shapes() {
        let w_out = Array2::<f32>::zeros((D_MODEL, N_RES));
        let agg   = HolographicMemoryAggregator::new(w_out);
        let emb   = Array2::<f32>::zeros((VOCAB_SIZE, D_MODEL));
        let bias  = Array1::<f32>::zeros(VOCAB_SIZE);
        let head  = SparseOutputHead::new(emb, bias);
        let ph    = PredictionHead::new(agg, head);

        let s_t      = Array1::<f32>::zeros(N_RES);
        let readouts: Vec<Array1<f32>> =
            (0..4).map(|_| Array1::zeros(RANK_R)).collect();
        let (full, sparse) = ph.forward(&s_t, &readouts);
        assert_eq!(full.len(), VOCAB_SIZE);
        assert_eq!(sparse.len(), TOP_K);
    }

    #[test]
    fn predict_embedding_shape() {
        let w_out = Array2::<f32>::zeros((D_MODEL, N_RES));
        let agg   = HolographicMemoryAggregator::new(w_out);
        let emb   = Array2::<f32>::zeros((VOCAB_SIZE, D_MODEL));
        let bias  = Array1::<f32>::zeros(VOCAB_SIZE);
        let ph    = PredictionHead::new(agg, SparseOutputHead::new(emb, bias));

        let s_t   = Array1::<f32>::zeros(N_RES);
        let x_hat = ph.predict_embedding(&s_t);
        assert_eq!(x_hat.len(), D_MODEL);
    }
}
