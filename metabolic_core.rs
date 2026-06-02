/// Metabolic core — GPU-accelerated Liquid State Machine, LSH routing, and
/// bio-inspired plastic layers with homeostatic control.
///
/// # GPU offload strategy
///
/// Two operations dominate the hot inference/training loop:
///
/// 1. **Reservoir step** — the 4096×4096 GEMV `R·s_{t-1} + W_in·x_t` plus
///    in-place `tanh`.  Cost on CPU: ~64 M MACs ≈ 32 ms on a 4 GHz core.
///    Offloaded to `GpuContext::dispatch_reservoir` which runs
///    `reservoir_update.wgsl` (workgroup 64 = one RDNA wavefront per row).
///
/// 2. **Hebbian outer-product update** — for each of the L=4 layers:
///    compute `delta_M = local_e ⊗ local_s` (RANK_R² = 1 024 MACs),
///    apply decay, fatigue clamping, and soft saturation.  While small
///    individually, the L × RANK_R² = 4 096 element-wise ops accumulate.
///    Offloaded to `GpuContext::dispatch_hebbian` which runs
///    `hebbian_plasticity.wgsl`.  The CPU only produces two RANK_R = 32
///    element vectors per layer (via small matrix-vector products on W_down
///    and W_up, whose dimensions are [32×512] and [32×4096] respectively).
///
/// # CPU responsibilities that remain on-chip
///
/// - `MultiScaleEncoder` (small projections, L1-friendly).
/// - `LshRouter::hash` (32×4096 dot product on the CPU shadow of s_t).
/// - `GlobalMetabolicController::compute_climate` (scalar arithmetic).
/// - `BioInspiredLayer::read_out_cpu` (M·(W_up·s_t), 1 KiB matrix).
/// - `HolographicMemoryAggregator::aggregate` (the CPU aggregates layer
///   read-outs and produces y_hidden before calling `upload_y_hidden`).
/// - Slow-learning SGD (touches the same tiny matrices; negligible).
///
/// # CPU shadow of s_t
///
/// After `dispatch_reservoir` the reservoir state lives exclusively in VRAM.
/// `ArcaSystem::forward_step_gpu` calls `GpuContext::readback_s` once per
/// step (16 KiB over PCIe ≈ 1 µs) to maintain a CPU-side shadow used by
/// the LSH router and the Hebbian layer projections.
///
/// # Feature flag
///
/// The `gpu` Cargo feature gates all GPU paths.  When the feature is absent
/// the file compiles to pure-CPU Rust identical to the original implementation.

use ndarray::{Array1, Array2};

#[cfg(feature = "gpu")]
use crate::gpu_context::GpuContext;

// ─────────────────────────────────────────────────────────────────────────────
// Public constants
// ─────────────────────────────────────────────────────────────────────────────
pub const N_RES:   usize = 4096;
pub const D_MODEL: usize = 512;
pub const RANK_R:  usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// LSH routing (CPU — tiny 32×4096 matrix, always L1-resident)
// ─────────────────────────────────────────────────────────────────────────────

/// LSH signature: h(s_t) = sign(W_lsh · s_t) ∈ {-1, +1}^k
pub struct LshRouter {
    /// W_lsh: shape [k, N_RES]
    pub w_lsh: Array2<f32>,
}

impl LshRouter {
    pub fn new(w_lsh: Array2<f32>) -> Self {
        assert_eq!(w_lsh.shape()[1], N_RES);
        LshRouter { w_lsh }
    }

    /// Returns the binary hash vector as `Vec<bool>` (true ≡ +1, false ≡ −1).
    ///
    /// `s_t` must be the **CPU shadow** of the current reservoir state
    /// (maintained in `ArcaSystem`).
    pub fn hash(&self, s_t: &Array1<f32>) -> Vec<bool> {
        debug_assert_eq!(s_t.len(), N_RES);
        self.w_lsh.dot(s_t).iter().map(|&v| v >= 0.0).collect()
    }

    /// Hamming distance between two binary hash signatures.
    #[inline]
    pub fn hamming(a: &[bool], b: &[bool]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LSM Reservoir — GPU execution + CPU fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Liquid State Machine reservoir.
///
/// The recurrent matrix `R` (4096×4096, ≈ 64 MiB) and the input projection
/// `W_in` (4096×512, ≈ 8 MiB) are uploaded to VRAM once at init and never
/// touched again during the hot loop.  CPU copies are retained **only** for
/// checkpoint serialisation.
pub struct LiquidReservoir {
    /// R matrix — CPU copy for serialisation only.
    pub r_matrix: Array2<f32>,
    /// W_in — CPU copy; also re-uploaded after slow-learning SGD updates.
    pub w_in:     Array2<f32>,
}

impl LiquidReservoir {
    pub fn new(r_matrix: Array2<f32>, w_in: Array2<f32>) -> Self {
        assert_eq!(r_matrix.shape(), &[N_RES, N_RES]);
        assert_eq!(w_in.shape(),     &[N_RES, D_MODEL]);
        LiquidReservoir { r_matrix, w_in }
    }

    // ── GPU path ──────────────────────────────────────────────────────────

    /// GPU-accelerated step.
    ///
    /// Uploads `x_t` (D_MODEL × 4 = 2 KiB) over PCIe and dispatches the
    /// `reservoir_update.wgsl` compute shader.  The new state is written into
    /// the ping-pong VRAM buffer; `gpu.s_ping` is flipped on return.
    ///
    /// **The caller must obtain the CPU shadow via `gpu.readback_s()` after
    /// this call if s_t is needed for LSH or Hebbian projections.**
    ///
    /// PCIe traffic: 2 KiB upload.  No download in the hot loop.
    #[cfg(feature = "gpu")]
    pub fn step_gpu(&self, gpu: &mut GpuContext, x_t: &Array1<f32>) {
        debug_assert_eq!(x_t.len(), D_MODEL);
        let x_flat: Vec<f32> = x_t.iter().cloned().collect();
        gpu.dispatch_reservoir(&x_flat);
    }

    // ── CPU fallback ──────────────────────────────────────────────────────

    /// Pure-CPU LSM step.
    ///   s_t = tanh(R · s_{t-1} + W_in · x_t)
    ///
    /// Used when the `gpu` feature is disabled (CPU-only build) or for unit
    /// testing without a GPU present.
    #[cfg(not(feature = "gpu"))]
    pub fn step(&self, s_prev: &Array1<f32>, x_t: &Array1<f32>) -> Array1<f32> {
        debug_assert_eq!(s_prev.len(), N_RES);
        debug_assert_eq!(x_t.len(),    D_MODEL);
        let recurrent  = self.r_matrix.dot(s_prev);
        let input_proj = self.w_in.dot(x_t);
        (recurrent + input_proj).mapv(|v| v.tanh())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Homeostatic Controller (§1.3.1) — pure scalar CPU
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the macro "climate" variables that modulate all layers globally.
pub struct GlobalMetabolicController {
    pub num_layers:  usize,
    pub gamma_macro: f32,
    pub tau_sleep:   f32,
    pub lambda_min:  f32,
    pub lambda_max:  f32,
}

impl GlobalMetabolicController {
    pub fn new(
        num_layers:  usize,
        gamma_macro: f32,
        tau_sleep:   f32,
        lambda_min:  f32,
        lambda_max:  f32,
    ) -> Self {
        GlobalMetabolicController { num_layers, gamma_macro, tau_sleep, lambda_min, lambda_max }
    }

    /// Compute macro climate variables from prediction error `e_t` and the
    /// previous tension scalar `T_{t-1}`.
    ///
    /// Returns `(T_t, β_global, λ_global, σ_global)`.
    pub fn compute_climate(
        &self,
        e_t:          &Array1<f32>,
        tension_prev: f32,
    ) -> (f32, f32, f32, f32) {
        // Surprise: S_t = ‖e_t‖₂
        let s_score: f32 = e_t.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Macro tension: T_t = (1-γ_m)·T_{t-1} + γ_m·tanh(S_t)
        let tension = (1.0 - self.gamma_macro) * tension_prev
            + self.gamma_macro * s_score.tanh();

        let ratio  = tension / self.tau_sleep;
        let beta   = 0.8 * ratio.tanh();
        let lambda = self.lambda_min + (self.lambda_max - self.lambda_min) * ratio.tanh();
        let sigma  = 0.25 * (1.0 - ratio.tanh());

        (tension, beta, lambda, sigma)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bio-inspired Plastic Layer (§1.3.2)
// ─────────────────────────────────────────────────────────────────────────────

/// One bio-inspired layer.
///
/// The CPU-side matrices `w_down` [RANK_R × D_MODEL] and `w_up` [RANK_R × N_RES]
/// are used in the hot loop only for the two small GEMV projections that
/// produce `local_e` and `local_s` (RANK_R-dimensional each).  The expensive
/// outer-product update and homeostatic blending happen in VRAM via the
/// Hebbian shader.
///
/// `m_base` [RANK_R × RANK_R] is uploaded to VRAM once at init and only
/// changed during slow-learning (rare); it acts as the resting-state
/// attractor for the homeostatic reset.
pub struct BioInspiredLayer {
    /// W_down: [RANK_R, D_MODEL]  — maps prediction error to rank-R subspace.
    pub w_down: Array2<f32>,
    /// W_up:   [RANK_R, N_RES]   — maps reservoir state to rank-R subspace.
    pub w_up:   Array2<f32>,
    /// M_base: [RANK_R, RANK_R]  — homeostatic resting-state attractor.
    pub m_base: Array2<f32>,
    /// γ — raw conductance gate parameter (conductance κ = sigmoid(γ)).
    pub gamma:  f32,
}

impl BioInspiredLayer {
    pub fn new(
        w_down: Array2<f32>,
        w_up:   Array2<f32>,
        m_base: Array2<f32>,
        gamma:  f32,
    ) -> Self {
        assert_eq!(w_down.shape(), &[RANK_R, D_MODEL]);
        assert_eq!(w_up.shape(),   &[RANK_R, N_RES]);
        assert_eq!(m_base.shape(), &[RANK_R, RANK_R]);
        BioInspiredLayer { w_down, w_up, m_base, gamma }
    }

    // ── GPU path ──────────────────────────────────────────────────────────

    /// GPU-accelerated Hebbian forward pass.
    ///
    /// Process
    /// -------
    /// 1. **CPU** computes `local_e = W_down · e_t` (RANK_R × D_MODEL GEMV,
    ///    tiny: 32×512 = 16 384 MACs).
    /// 2. **CPU** computes `local_s = W_up · s_t_cpu` (RANK_R × N_RES GEMV,
    ///    32×4096 = 131 072 MACs — still CPU-friendly due to cache reuse).
    /// 3. **GPU** receives the two RANK_R-vectors (128 B each) and the six
    ///    scalar parameters, then executes the full outer-product update,
    ///    Frobenius fatigue, homeostatic clamping, and soft saturation for
    ///    the RANK_R×RANK_R M matrix in parallel.
    ///
    /// Returns `κ_local = sigmoid(γ)` — still needed on CPU for the loss
    /// budget term.
    ///
    /// `s_t_cpu` must be the CPU shadow of the current reservoir state.
    ///
    /// PCIe traffic per layer per step: 2 × 128 B upload + 0 download.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        gpu:            &mut GpuContext,
        layer_idx:      usize,
        e_t:            &Array1<f32>,
        s_t_cpu:        &Array1<f32>,
        beta_global:    f32,
        lambda_global:  f32,
        sigma_global:   f32,
        eta_lr:         f32,
        alpha_fatigue:  f32,
        tau_saturation: f32,
    ) -> f32 {
        debug_assert_eq!(e_t.len(),     D_MODEL);
        debug_assert_eq!(s_t_cpu.len(), N_RES);

        let kappa = sigmoid(self.gamma);

        // Small CPU projections — RANK_R=32 elements each
        let local_e: Vec<f32> = self.w_down.dot(e_t).into_raw_vec();
        let local_s: Vec<f32> = self.w_up.dot(s_t_cpu).into_raw_vec();

        // β³ · η is pre-multiplied on CPU to save a few shader instructions
        let beta3_eta = beta_global.powi(3) * eta_lr;

        gpu.dispatch_hebbian(
            layer_idx,
            &local_e,
            &local_s,
            lambda_global,
            kappa,
            beta3_eta,
            sigma_global,
            alpha_fatigue,
            tau_saturation,
        );

        kappa
    }

    /// Read-out using a CPU-side M shadow matrix.
    ///
    /// Because M lives in VRAM, `ArcaSystem` maintains a CPU shadow
    /// `m_shadow[l]` updated by a lightweight per-step readback of the
    /// 32×32 = 1 KiB matrix.  This method accepts that shadow.
    ///
    ///   y_mem = M_t · (W_up · s_t)  ∈ ℝ^RANK_R
    #[cfg(feature = "gpu")]
    pub fn read_out_cpu(
        &self,
        m_t_cpu: &Array2<f32>,
        s_t_cpu: &Array1<f32>,
    ) -> Array1<f32> {
        debug_assert_eq!(m_t_cpu.shape(), &[RANK_R, RANK_R]);
        debug_assert_eq!(s_t_cpu.len(),   N_RES);
        let local_s = self.w_up.dot(s_t_cpu);
        m_t_cpu.dot(&local_s)
    }

    // ── CPU fallback ──────────────────────────────────────────────────────

    /// Pure-CPU Hebbian forward pass (original implementation, unchanged).
    ///
    /// Returns `(M_t_updated, κ_local)`.
    ///
    /// Equations:
    ///   κ       = sigmoid(γ)
    ///   ΔM      = (W_down · e_t) ⊗ (W_up · s_t)
    ///   M_next  = λ · M_{t-1} + κ · (β³ · η · ΔM)
    ///   E       = ‖M_next‖_F²
    ///   σ_local = clip(σ_g + α_f · tanh(E / τ), 0, 0.95)
    ///   M_t     = tanh( (1−σ)·M_next + σ·M_base ) × 1.5
    #[cfg(not(feature = "gpu"))]
    pub fn forward_and_adapt(
        &self,
        m_prev:         &Array2<f32>,
        e_t:            &Array1<f32>,
        s_t:            &Array1<f32>,
        beta_global:    f32,
        lambda_global:  f32,
        sigma_global:   f32,
        eta_lr:         f32,
        alpha_fatigue:  f32,
        tau_saturation: f32,
    ) -> (Array2<f32>, f32) {
        debug_assert_eq!(m_prev.shape(), &[RANK_R, RANK_R]);
        debug_assert_eq!(e_t.len(),      D_MODEL);
        debug_assert_eq!(s_t.len(),      N_RES);

        let kappa    = sigmoid(self.gamma);
        let local_e  = self.w_down.dot(e_t);
        let local_s  = self.w_up.dot(s_t);
        let delta_m  = outer_product_r32(&local_e, &local_s);
        let m_next   = lambda_global * m_prev + kappa * (beta_global.powi(3) * eta_lr * delta_m);

        let frobenius_sq: f32 = m_next.iter().map(|v| v * v).sum();
        let fatigue     = alpha_fatigue * (frobenius_sq / tau_saturation).tanh();
        let sigma_local = (sigma_global + fatigue).clamp(0.0, 0.95);

        let m_t = (1.0 - sigma_local) * &m_next + sigma_local * &self.m_base;
        let m_t = m_t.mapv(|v| v.tanh() * 1.5);
        (m_t, kappa)
    }

    /// CPU read-out (unchanged, used in the CPU fallback path).
    #[cfg(not(feature = "gpu"))]
    pub fn read_out(&self, m_t: &Array2<f32>, s_t: &Array1<f32>) -> Array1<f32> {
        let local_s = self.w_up.dot(s_t);
        m_t.dot(&local_s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// κ = sigmoid(γ)
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Outer product of two RANK_R vectors → Array2<f32> [RANK_R × RANK_R].
///
/// Used only in the CPU fallback path; the GPU path computes the outer
/// product directly inside the Hebbian shader.
#[cfg(not(feature = "gpu"))]
#[inline]
fn outer_product_r32(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    debug_assert_eq!(a.len(), RANK_R);
    debug_assert_eq!(b.len(), RANK_R);

    // MaybeUninit approach: initialise every element exactly once to avoid
    // redundant zeroing from `Array2::zeros`.
    let mut out = Array2::<f32>::uninit((RANK_R, RANK_R));
    // SAFETY: we write every element before calling `assume_init`.
    let out_slice = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut f32,
            RANK_R * RANK_R,
        )
    };
    for i in 0..RANK_R {
        for j in 0..RANK_R {
            out_slice[i * RANK_R + j] = a[i] * b[j];
        }
    }
    // SAFETY: all elements have been initialised above.
    unsafe { out.assume_init() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── Reservoir CPU fallback ────────────────────────────────────────────

    #[test]
    fn reservoir_struct_construction() {
        let r    = Array2::<f32>::zeros((N_RES, N_RES));
        let w_in = Array2::<f32>::zeros((N_RES, D_MODEL));
        let _res = LiquidReservoir::new(r, w_in);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn reservoir_cpu_output_shape_and_zero_input() {
        let r    = Array2::<f32>::zeros((N_RES, N_RES));
        let w_in = Array2::<f32>::zeros((N_RES, D_MODEL));
        let res  = LiquidReservoir::new(r, w_in);
        let s    = Array1::<f32>::zeros(N_RES);
        let x    = Array1::<f32>::zeros(D_MODEL);
        let s_new = res.step(&s, &x);
        assert_eq!(s_new.len(), N_RES);
        // tanh(0) = 0 for all elements
        assert!(s_new.iter().all(|&v| v == 0.0));
    }

    // ── Layer CPU fallback ────────────────────────────────────────────────

    #[test]
    fn layer_struct_construction() {
        let w_down = Array2::<f32>::zeros((RANK_R, D_MODEL));
        let w_up   = Array2::<f32>::zeros((RANK_R, N_RES));
        let m_base = Array2::<f32>::zeros((RANK_R, RANK_R));
        let _layer = BioInspiredLayer::new(w_down, w_up, m_base, 0.0);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn layer_cpu_output_shape_and_kappa() {
        let w_down = Array2::<f32>::zeros((RANK_R, D_MODEL));
        let w_up   = Array2::<f32>::zeros((RANK_R, N_RES));
        let m_base = Array2::<f32>::zeros((RANK_R, RANK_R));
        let layer  = BioInspiredLayer::new(w_down, w_up, m_base, 0.0);

        let m_prev = Array2::<f32>::zeros((RANK_R, RANK_R));
        let e_t    = Array1::<f32>::zeros(D_MODEL);
        let s_t    = Array1::<f32>::zeros(N_RES);
        let (m_new, kappa) =
            layer.forward_and_adapt(&m_prev, &e_t, &s_t, 0.5, 0.9, 0.1, 0.01, 0.1, 100.0);
        assert_eq!(m_new.shape(), &[RANK_R, RANK_R]);
        // sigmoid(0) = 0.5
        assert!((kappa - 0.5).abs() < 1e-5, "kappa={}", kappa);
    }

    // ── Climate controller ────────────────────────────────────────────────

    #[test]
    fn climate_controller_output_ranges() {
        let ctrl = GlobalMetabolicController::new(4, 0.01, 1.0, 0.8, 0.999);
        let e    = Array1::<f32>::from_vec(vec![1.0_f32; D_MODEL]);
        let (t, beta, lambda, sigma) = ctrl.compute_climate(&e, 0.5);

        assert!(t      >= 0.0 && t      <= 1.0 + 1e-5, "tension={}", t);
        assert!(beta   >= 0.0 && beta   <= 0.8 + 1e-5, "beta={}",    beta);
        assert!(lambda >= 0.8 && lambda <= 1.0 + 1e-5, "lambda={}",  lambda);
        assert!(sigma  >= 0.0 && sigma  <= 0.25 + 1e-5,"sigma={}",   sigma);
    }

    // ── LSH router ────────────────────────────────────────────────────────

    #[test]
    fn lsh_hash_length_and_hamming() {
        let k      = 32usize;
        let w_lsh  = Array2::<f32>::zeros((k, N_RES));
        let router = LshRouter::new(w_lsh);
        let s      = Array1::<f32>::zeros(N_RES);
        let hash   = router.hash(&s);
        assert_eq!(hash.len(), k);

        let all_true  = vec![true;  k];
        let all_false = vec![false; k];
        assert_eq!(LshRouter::hamming(&all_true, &all_false), k);
        assert_eq!(LshRouter::hamming(&all_true, &all_true),  0);
    }

    // ── Sigmoid helper ────────────────────────────────────────────────────

    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs()     < 1e-6);
        assert!((sigmoid(f32::INFINITY) - 1.0) < 1e-6);
        assert!((sigmoid(f32::NEG_INFINITY)).abs() < 1e-6);
    }
}
