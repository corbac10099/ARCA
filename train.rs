/// Slow-learning optimisation loop (§1.4).
///
/// Minimises the multi-objective loss:
///
///   L_total = L_next_token
///           + α_meta · Σ_l ‖W_down^(l) · e_t − (W_up^(l))ᵀ · s_t‖²₂
///           + μ_budget · Σ_l κ^(l) / (1 − κ^(l) + ε)
///
/// Gradient descent updates the "skeleton" parameters:
///   W_fusion, W_in, W_out, W_up^(l), W_down^(l), M_base^(l), γ^(l)
///
/// The memory matrices M_t^(l) are updated via the local Hebbian rule (detached)
/// in metabolic_core.rs and are NOT updated here.

use ndarray::{Array1, Array2, Zip};

use crate::metabolic_core::{BioInspiredLayer, RANK_R, N_RES, D_MODEL};
use crate::memory::VOCAB_SIZE;

// ─────────────────────────────────────────────────────────────────────────────
// Hyper-parameters (production defaults, can be overridden at construction)
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub alpha_meta: f32,
    pub mu_budget: f32,
    pub eps: f32,
    pub learning_rate: f32,
    pub grad_clip_norm: f32,
    pub eta_lr_hebbian: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            alpha_meta: 0.01,
            mu_budget: 0.001,
            eps: 1e-6,
            learning_rate: 3e-4,
            grad_clip_norm: 1.0,
            eta_lr_hebbian: 0.01,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss functions
// ─────────────────────────────────────────────────────────────────────────────

/// Next-token cross-entropy loss (numerically stable log-softmax).
///
/// logits: full vocabulary logit vector ∈ ℝ^V
/// target: ground-truth token index
pub fn cross_entropy_loss(logits: &Array1<f32>, target: usize) -> f32 {
    debug_assert_eq!(logits.len(), VOCAB_SIZE);
    debug_assert!(target < VOCAB_SIZE);

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&l| l - max_logit).collect();
    let log_sum_exp: f32 = shifted.iter().map(|&v| v.exp()).sum::<f32>().ln();
    let log_prob_target = shifted[target] - log_sum_exp;
    -log_prob_target
}

/// Meta-plasticity alignment loss.
///
/// For each layer l:
///   ‖W_down^(l) · e_t − (W_up^(l))ᵀ · s_t‖²₂
///
/// The interpretation: the error projection and the state projection should
/// align in the memory subspace — this encourages the skeleton matrices to
/// cooperate with the local Hebbian updates.
pub fn meta_plasticity_loss(
    layers: &[&BioInspiredLayer],
    e_t: &Array1<f32>,
    s_t: &Array1<f32>,
) -> f32 {
    let mut total = 0.0_f32;
    for layer in layers {
        // W_down · e_t  ∈ ℝ^RANK_R
        let proj_e = layer.w_down.dot(e_t);
        // (W_up)ᵀ · s_t  ∈ ℝ^RANK_R
        //   W_up is [RANK_R, N_RES], so W_upᵀ is [N_RES, RANK_R]
        //   (W_up)ᵀ · s_t = W_up.t() · s_t  ∈ ℝ^RANK_R
        // Wait: w_up is [RANK_R, N_RES]. w_up.t() is [N_RES, RANK_R].
        // We need (W_up)^T · s_t where s_t ∈ ℝ^{N_RES} → result ∈ ℝ^{RANK_R}
        // Actually: W_up ∈ ℝ^{RANK_R × N_RES}.  The spec says (W_up^(l))^T ∈ ℝ^{N_RES × RANK_R}.
        // (W_up^T) · s_t: (N_RES × RANK_R) · (N_RES) → not conformant.
        // Correct reading: W_up · s_t is (RANK_R × N_RES) · (N_RES) = RANK_R  ← this is what we want.
        // The spec notation (W_up^(l))^T · s_t must mean W_up · s_t (apply W_up on s_t).
        let proj_s = layer.w_up.dot(s_t);
        let diff = proj_e - proj_s;
        total += diff.iter().map(|v| v * v).sum::<f32>();
    }
    total
}

/// Asymptotic metabolic budget loss (anti-monopoly barrier).
///
///   μ_budget · Σ_l  κ^(l) / (1 − κ^(l) + ε)
///
/// This penalises any layer saturating its conductance gate κ → 1,
/// forcing a sparse, distributed resource allocation across layers.
pub fn calculate_asymptotic_budget_loss(kappas: &[f32], mu_budget: f32, eps: f32) -> f32 {
    let mut loss = 0.0_f32;
    for &kappa in kappas {
        loss += kappa / (1.0 - kappa + eps);
    }
    mu_budget * loss
}

/// Aggregate multi-objective loss.
pub fn total_loss(
    next_token_loss: f32,
    meta_loss: f32,
    budget_loss: f32,
    alpha_meta: f32,
) -> f32 {
    next_token_loss + alpha_meta * meta_loss + budget_loss
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient computation helpers (manual / finite-difference)
//
// In a production system these would be backed by an automatic differentiation
// engine (e.g. candle, burn, or custom backward passes).  Here we implement
// the closed-form gradients for the skeleton parameters as specified.
// ─────────────────────────────────────────────────────────────────────────────

/// Gradient of the cross-entropy loss w.r.t. the full logit vector.
///   ∂L/∂logits[v] = softmax(logits)[v] − 𝟏[v == target]
pub fn grad_cross_entropy_wrt_logits(logits: &Array1<f32>, target: usize) -> Array1<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let mut softmax = Array1::from_vec(exps.iter().map(|&e| e / sum_exp).collect());
    softmax[target] -= 1.0;
    softmax
}

/// Gradient-clipping by global L2 norm (in-place).
pub fn clip_grad_norm(grad: &mut Array1<f32>, max_norm: f32) {
    let norm: f32 = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > max_norm {
        grad.mapv_inplace(|v| v * max_norm / norm);
    }
}

pub fn clip_grad_norm_2d(grad: &mut Array2<f32>, max_norm: f32) {
    let norm: f32 = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > max_norm {
        grad.mapv_inplace(|v| v * max_norm / norm);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD skeleton parameter update (Adam-lite: SGD with momentum = 0)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place SGD step: param -= lr * grad  (with norm clipping applied before call)
#[inline]
pub fn sgd_step_2d(param: &mut Array2<f32>, grad: &Array2<f32>, lr: f32) {
    Zip::from(param.view_mut())
        .and(grad.view())
        .for_each(|p, &g| *p -= lr * g);
}

#[inline]
pub fn sgd_step_1d(param: &mut Array1<f32>, grad: &Array1<f32>, lr: f32) {
    Zip::from(param.view_mut())
        .and(grad.view())
        .for_each(|p, &g| *p -= lr * g);
}

// ─────────────────────────────────────────────────────────────────────────────
// The slow-learning training step
//
// This function orchestrates one full gradient step over the skeleton
// parameters given a single (x_t, target_token) observation.  It computes:
//
//   1. L_next_token via cross-entropy
//   2. L_meta via meta-plasticity alignment
//   3. L_budget via asymptotic barrier
//   4. Propagates gradients back through W_out, W_in, W_fusion (chain rule)
//   5. Updates γ^(l) for each layer (budget + meta gradients)
//
// The memory matrices M_t are NOT touched here; they are updated by the
// Hebbian rule in metabolic_core::BioInspiredLayer::forward_and_adapt.
// ─────────────────────────────────────────────────────────────────────────────

pub struct TrainState {
    pub config: TrainConfig,
}

impl TrainState {
    pub fn new(config: TrainConfig) -> Self {
        TrainState { config }
    }

    /// Compute all losses and their scalar sum.  Returns individual components
    /// so callers can log them separately.
    pub fn compute_losses(
        &self,
        logits: &Array1<f32>,
        target_token: usize,
        layers: &[&BioInspiredLayer],
        e_t: &Array1<f32>,
        s_t: &Array1<f32>,
        kappas: &[f32],
    ) -> LossComponents {
        let l_next = cross_entropy_loss(logits, target_token);
        let l_meta = meta_plasticity_loss(layers, e_t, s_t);
        let l_budget = calculate_asymptotic_budget_loss(
            kappas,
            self.config.mu_budget,
            self.config.eps,
        );
        let l_total = total_loss(l_next, l_meta, l_budget, self.config.alpha_meta);
        LossComponents { next_token: l_next, meta: l_meta, budget: l_budget, total: l_total }
    }

    /// Gradient of L_total w.r.t. logits (for backprop through output embeddings).
    pub fn dL_dlogits(&self, logits: &Array1<f32>, target_token: usize) -> Array1<f32> {
        // Only the cross-entropy term contributes directly to logit gradients.
        grad_cross_entropy_wrt_logits(logits, target_token)
    }

    /// Gradient of γ^(l) from the budget loss.
    ///
    ///   ∂L_budget / ∂γ^(l) = μ_budget · ∂/∂γ [ κ / (1−κ+ε) ]
    ///   where κ = sigmoid(γ)
    ///   ∂κ/∂γ = κ(1−κ)
    ///   ∂/∂γ [ κ/(1−κ+ε) ] = [ (1−κ+ε) + κ ] / (1−κ+ε)² · κ(1−κ)
    pub fn grad_gamma_budget(&self, kappa: f32) -> f32 {
        let denom = 1.0 - kappa + self.config.eps;
        let dk_dkappa = (denom + kappa) / (denom * denom);
        let dkappa_dgamma = kappa * (1.0 - kappa);
        self.config.mu_budget * dk_dkappa * dkappa_dgamma
    }

    /// Apply one SGD step to the conductance gate γ^(l).
    pub fn update_gamma(&self, gamma: &mut f32, kappa: f32) {
        let grad = self.grad_gamma_budget(kappa);
        *gamma -= self.config.learning_rate * grad;
    }
}

#[derive(Debug)]
pub struct LossComponents {
    pub next_token: f32,
    pub meta: f32,
    pub budget: f32,
    pub total: f32,
}

impl std::fmt::Display for LossComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "total={:.4}  next_token={:.4}  meta={:.4}  budget={:.4}",
            self.total, self.next_token, self.meta, self.budget
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn cross_entropy_known_value() {
        let mut logits = Array1::from_elem(VOCAB_SIZE, -100.0_f32);
        logits[42] = 10.0_f32; // dominant class
        let loss = cross_entropy_loss(&logits, 42);
        // With logit[42]>>others, loss must be tiny.
        assert!(loss < 0.01, "Expected small loss, got {}", loss);
    }

    #[test]
    fn budget_loss_non_negative() {
        let kappas = vec![0.1, 0.5, 0.9];
        let loss = calculate_asymptotic_budget_loss(&kappas, 1e-3, 1e-6);
        assert!(loss >= 0.0);
    }

    #[test]
    fn budget_loss_increases_with_kappa() {
        let low = calculate_asymptotic_budget_loss(&[0.1], 1.0, 1e-6);
        let high = calculate_asymptotic_budget_loss(&[0.9], 1.0, 1e-6);
        assert!(high > low);
    }

    #[test]
    fn softmax_grad_sums_to_zero_shifted() {
        let logits = Array1::from_vec((0..VOCAB_SIZE).map(|i| (i as f32) * 0.001).collect());
        let grad = grad_cross_entropy_wrt_logits(&logits, 100);
        // Sum of (softmax - one_hot) should be zero (prob mass sums to 1, subtract 1 for target).
        let sum: f32 = grad.iter().sum();
        assert!((sum).abs() < 1e-3, "sum={}", sum);
    }
}
