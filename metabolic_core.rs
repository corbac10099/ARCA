/// Metabolic core: Liquid State Machine reservoir, LSH routing, and
/// bio-inspired plastic layers with homeostatic control.
///
/// Implements sections 1.2, 1.3 of the formal specification.

use ndarray::{Array1, Array2, Axis, s};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
pub const N_RES: usize = 4096;
pub const D_MODEL: usize = 512;
pub const RANK_R: usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// LSH routing
// ─────────────────────────────────────────────────────────────────────────────

/// LSH signature: h(s_t) = sign(W_lsh · s_t) ∈ {-1,+1}^k
pub struct LshRouter {
    /// W_lsh: shape [k, N_RES]
    pub w_lsh: Array2<f32>,
}

impl LshRouter {
    pub fn new(w_lsh: Array2<f32>) -> Self {
        assert_eq!(w_lsh.shape()[1], N_RES);
        LshRouter { w_lsh }
    }

    /// Returns the binary hash vector as a Vec<bool> (true ≡ +1, false ≡ -1).
    pub fn hash(&self, s_t: &Array1<f32>) -> Vec<bool> {
        let projected = self.w_lsh.dot(s_t);
        projected.iter().map(|&v| v >= 0.0).collect()
    }

    /// Hamming distance between two hash signatures.
    #[inline]
    pub fn hamming(a: &[bool], b: &[bool]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LSM Reservoir
//   s_t = tanh(R · s_{t-1} + W_in · x_t)
// ─────────────────────────────────────────────────────────────────────────────

pub struct LiquidReservoir {
    /// Sparse R matrix reconstructed on-the-fly by SovereignModel::generate_sparse_lsm().
    /// Stored as a dense Array2 (sparsity is in values, not storage for now).
    pub r_matrix: Array2<f32>,
    /// W_in: shape [N_RES, D_MODEL] — learned projection.
    pub w_in: Array2<f32>,
}

impl LiquidReservoir {
    pub fn new(r_matrix: Array2<f32>, w_in: Array2<f32>) -> Self {
        assert_eq!(r_matrix.shape(), &[N_RES, N_RES]);
        assert_eq!(w_in.shape(), &[N_RES, D_MODEL]);
        LiquidReservoir { r_matrix, w_in }
    }

    /// One-step LSM transition.
    ///   s_t = tanh(R · s_{t-1} + W_in · x_t)
    /// Returns new state vector ∈ ℝ^{N_RES}.
    /// No heap allocation beyond the returned Array1.
    pub fn step(&self, s_prev: &Array1<f32>, x_t: &Array1<f32>) -> Array1<f32> {
        debug_assert_eq!(s_prev.len(), N_RES);
        debug_assert_eq!(x_t.len(), D_MODEL);

        let recurrent = self.r_matrix.dot(s_prev);
        let input_proj = self.w_in.dot(x_t);
        let pre_act = recurrent + input_proj;
        pre_act.mapv(|v| v.tanh())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Homeostatic Controller (§1.3.1)
// ─────────────────────────────────────────────────────────────────────────────

pub struct GlobalMetabolicController {
    pub num_layers: usize,
    pub gamma_macro: f32,
    pub tau_sleep: f32,
    pub lambda_min: f32,
    pub lambda_max: f32,
}

impl GlobalMetabolicController {
    pub fn new(
        num_layers: usize,
        gamma_macro: f32,
        tau_sleep: f32,
        lambda_min: f32,
        lambda_max: f32,
    ) -> Self {
        GlobalMetabolicController { num_layers, gamma_macro, tau_sleep, lambda_min, lambda_max }
    }

    /// Compute macro climate variables from prediction error e_t and previous tension T_{t-1}.
    ///
    /// Returns (T_t, β_global, λ_global, σ_global)
    pub fn compute_climate(
        &self,
        e_t: &Array1<f32>,
        tension_prev: f32,
    ) -> (f32, f32, f32, f32) {
        // Surprise score: S_t = ‖e_t‖₂
        let s_score: f32 = e_t.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Macro tension: T_t = (1-γ_macro)·T_{t-1} + γ_macro·tanh(S_t)
        let tension = (1.0 - self.gamma_macro) * tension_prev
            + self.gamma_macro * s_score.tanh();

        let ratio = tension / self.tau_sleep;

        // β_global = 0.8 · tanh(T_t / τ_sleep)
        let beta = 0.8 * ratio.tanh();

        // λ_global = λ_min + (λ_max - λ_min) · tanh(T_t / τ_sleep)
        let lambda =
            self.lambda_min + (self.lambda_max - self.lambda_min) * ratio.tanh();

        // σ_global = 0.25 · (1 - tanh(T_t / τ_sleep))
        let sigma = 0.25 * (1.0 - ratio.tanh());

        (tension, beta, lambda, sigma)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bio-inspired Plastic Layer (§1.3.2)
// ─────────────────────────────────────────────────────────────────────────────

pub struct BioInspiredLayer {
    /// W_down: shape [RANK_R, D_MODEL]
    pub w_down: Array2<f32>,
    /// W_up:   shape [RANK_R, N_RES]
    pub w_up: Array2<f32>,
    /// M_base: shape [RANK_R, RANK_R] — resting-state attractor for homeostatic reset.
    pub m_base: Array2<f32>,
    /// Conductance gate parameter (raw, before sigmoid).
    pub gamma: f32,
}

impl BioInspiredLayer {
    pub fn new(
        w_down: Array2<f32>,
        w_up: Array2<f32>,
        m_base: Array2<f32>,
        gamma: f32,
    ) -> Self {
        assert_eq!(w_down.shape(), &[RANK_R, D_MODEL]);
        assert_eq!(w_up.shape(), &[RANK_R, N_RES]);
        assert_eq!(m_base.shape(), &[RANK_R, RANK_R]);
        BioInspiredLayer { w_down, w_up, m_base, gamma }
    }

    /// Local Hebbian update — gradient is detached (stop-grad) on M.
    ///
    /// Returns (M_t_updated, κ_local).
    ///
    /// Equations:
    ///   κ       = sigmoid(γ)
    ///   ΔM      = (W_down · e_t) ⊗ (W_up · s_t)   [rank-1 outer product]
    ///   M_next  = λ · M_{t-1} + κ · (β³ · η · ΔM)
    ///   E       = ‖M_next‖_F²
    ///   σ_local = clip(σ_global + α_fatigue · tanh(E / τ_sat), 0, 0.95)
    ///   M_t     = (1-σ_local)·M_next + σ_local·M_base
    ///   M_t     = tanh(M_t) * 1.5     [soft saturation]
    pub fn forward_and_adapt(
        &self,
        m_prev: &Array2<f32>,
        e_t: &Array1<f32>,
        s_t: &Array1<f32>,
        beta_global: f32,
        lambda_global: f32,
        sigma_global: f32,
        eta_lr: f32,
        alpha_fatigue: f32,
        tau_saturation: f32,
    ) -> (Array2<f32>, f32) {
        debug_assert_eq!(m_prev.shape(), &[RANK_R, RANK_R]);
        debug_assert_eq!(e_t.len(), D_MODEL);
        debug_assert_eq!(s_t.len(), N_RES);

        let kappa = sigmoid(self.gamma);

        // Local projections (no gradient flows back into w_down / w_up here;
        // that happens in the slow-learning backward pass).
        let local_e: Array1<f32> = self.w_down.dot(e_t);   // ℝ^RANK_R
        let local_s: Array1<f32> = self.w_up.dot(s_t);     // ℝ^RANK_R

        // ΔM = local_e ⊗ local_s  (rank-1 outer product, RANK_R × RANK_R)
        let delta_m = outer_product_r32(&local_e, &local_s);

        // M_next = λ · M_{t-1} + κ · (β³ · η · ΔM)
        let innovation_gate = beta_global.powi(3);
        let m_next = lambda_global * m_prev + kappa * (innovation_gate * eta_lr * delta_m);

        // Frobenius energy: E = ‖M_next‖_F²  (sum of squared elements)
        let frobenius_sq: f32 = m_next.iter().map(|v| v * v).sum();

        // σ_local = clip(σ_global + α_fatigue · tanh(E / τ_sat), 0, 0.95)
        let fatigue = alpha_fatigue * (frobenius_sq / tau_saturation).tanh();
        let sigma_local = (sigma_global + fatigue).clamp(0.0, 0.95);

        // Homeostatic relaxation: M_t = (1-σ)·M_next + σ·M_base
        let m_t = (1.0 - sigma_local) * &m_next + sigma_local * &self.m_base;

        // Soft saturation
        let m_t = m_t.mapv(|v| v.tanh() * 1.5);

        (m_t, kappa)
    }

    /// Read-out: project the memory matrix onto the reservoir state to produce
    /// a modulation signal injected into the output head.
    ///   y_mem = M_t · (W_up · s_t)  ∈ ℝ^RANK_R
    pub fn read_out(&self, m_t: &Array2<f32>, s_t: &Array1<f32>) -> Array1<f32> {
        let local_s = self.w_up.dot(s_t);
        m_t.dot(&local_s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Outer product of two RANK_R vectors, returned as Array2<f32> [RANK_R × RANK_R].
/// Avoids any heap allocation beyond the single result Array.
#[inline]
fn outer_product_r32(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    debug_assert_eq!(a.len(), RANK_R);
    debug_assert_eq!(b.len(), RANK_R);
    let mut out = Array2::<f32>::uninit((RANK_R, RANK_R));
    // SAFETY: We initialise every element exactly once below.
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
    // SAFETY: all elements initialised.
    unsafe { out.assume_init() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_output_shape() {
        let r = Array2::<f32>::zeros((N_RES, N_RES));
        let w_in = Array2::<f32>::zeros((N_RES, D_MODEL));
        let res = LiquidReservoir::new(r, w_in);
        let s = Array1::<f32>::zeros(N_RES);
        let x = Array1::<f32>::zeros(D_MODEL);
        let s_new = res.step(&s, &x);
        assert_eq!(s_new.len(), N_RES);
        // tanh(0) = 0
        assert!(s_new.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn layer_adapt_output_shape() {
        let w_down = Array2::<f32>::zeros((RANK_R, D_MODEL));
        let w_up = Array2::<f32>::zeros((RANK_R, N_RES));
        let m_base = Array2::<f32>::zeros((RANK_R, RANK_R));
        let layer = BioInspiredLayer::new(w_down, w_up, m_base, 0.0);

        let m_prev = Array2::<f32>::zeros((RANK_R, RANK_R));
        let e_t = Array1::<f32>::zeros(D_MODEL);
        let s_t = Array1::<f32>::zeros(N_RES);
        let (m_new, kappa) = layer.forward_and_adapt(&m_prev, &e_t, &s_t, 0.5, 0.9, 0.1, 0.01, 0.1, 100.0);
        assert_eq!(m_new.shape(), &[RANK_R, RANK_R]);
        assert!((kappa - 0.5).abs() < 1e-5);
    }

    #[test]
    fn climate_controller_bounds() {
        let ctrl = GlobalMetabolicController::new(4, 0.01, 1.0, 0.8, 0.999);
        let e = Array1::<f32>::from_vec(vec![1.0; D_MODEL]);
        let (t, beta, lambda, sigma) = ctrl.compute_climate(&e, 0.5);
        assert!(t >= 0.0 && t <= 1.0 + 1e-5);
        assert!(beta >= 0.0 && beta <= 0.8 + 1e-5);
        assert!(lambda >= 0.8 && lambda <= 1.0 + 1e-5);
        assert!(sigma >= 0.0 && sigma <= 0.25 + 1e-5);
    }
}
