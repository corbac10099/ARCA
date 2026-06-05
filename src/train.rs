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
    /// Save a checkpoint every this many steps (0 = disabled).
    pub checkpoint_every: usize,
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
            checkpoint_every: 500,
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
pub fn meta_plasticity_loss(
    layers: &[&BioInspiredLayer],
    e_t: &Array1<f32>,
    s_t: &Array1<f32>,
) -> f32 {
    let mut total = 0.0_f32;
    for layer in layers {
        let proj_e = layer.w_down.dot(e_t);
        let proj_s = layer.w_up.dot(s_t);
        let diff = proj_e - proj_s;
        total += diff.iter().map(|v| v * v).sum::<f32>();
    }
    total
}

/// Asymptotic metabolic budget loss (anti-monopoly barrier).
///
///   μ_budget · Σ_l  κ^(l) / (1 − κ^(l) + ε)
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
// Gradient computation helpers
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
// SGD skeleton parameter update
// ─────────────────────────────────────────────────────────────────────────────

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
// ─────────────────────────────────────────────────────────────────────────────

pub struct TrainState {
    pub config: TrainConfig,
    /// Total number of parameter update steps taken (for checkpointing).
    pub step: usize,
}

impl TrainState {
    pub fn new(config: TrainConfig) -> Self {
        TrainState { config, step: 0 }
    }

    /// Compute all losses and their scalar sum.
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

    /// Gradient of L_total w.r.t. logits.
    pub fn dL_dlogits(&self, logits: &Array1<f32>, target_token: usize) -> Array1<f32> {
        grad_cross_entropy_wrt_logits(logits, target_token)
    }

    /// Gradient of γ^(l) from the budget loss.
    pub fn grad_gamma_budget(&self, kappa: f32) -> f32 {
        let denom = 1.0 - kappa + self.config.eps;
        let dk_dkappa = (denom + kappa) / (denom * denom);
        let dkappa_dgamma = kappa * (1.0 - kappa);
        self.config.mu_budget * dk_dkappa * dkappa_dgamma
    }

    /// Apply one SGD step to the conductance gate γ^(l) and increment step counter.
    pub fn update_gamma(&mut self, gamma: &mut f32, kappa: f32) {
        let grad = self.grad_gamma_budget(kappa);
        *gamma -= self.config.learning_rate * grad;
        self.step += 1;
    }

    /// Returns true if a checkpoint should be saved at the current step.
    pub fn should_checkpoint(&self) -> bool {
        self.config.checkpoint_every > 0 && self.step > 0 && self.step % self.config.checkpoint_every == 0
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
// Batch training helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A single training sample: byte sequence with its aligned BPE ids.
pub struct TrainSample {
    pub bytes: Vec<u8>,
    pub bpe_ids: Vec<u32>,
}

/// Slice a corpus into non-overlapping windows of `window_size` bytes.
///
/// Each window becomes one `TrainSample`.  The last incomplete window is
/// dropped so every sample has exactly `window_size` bytes.
pub fn build_batches(corpus: &[u8], bpe_ids: &[u32], window_size: usize) -> Vec<TrainSample> {
    assert_eq!(corpus.len(), bpe_ids.len());
    corpus
        .windows(window_size)
        .zip(bpe_ids.windows(window_size))
        .step_by(window_size)          // non-overlapping
        .map(|(b, ids)| TrainSample {
            bytes: b.to_vec(),
            bpe_ids: ids.to_vec(),
        })
        .collect()
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
        logits[42] = 10.0_f32;
        let loss = cross_entropy_loss(&logits, 42);
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
        let sum: f32 = grad.iter().sum();
        assert!((sum).abs() < 1e-3, "sum={}", sum);
    }

    #[test]
    fn build_batches_count() {
        let corpus: Vec<u8> = (0u8..100).collect();
        let bpe_ids: Vec<u32> = corpus.iter().map(|&b| b as u32).collect();
        let batches = build_batches(&corpus, &bpe_ids, 32);
        assert_eq!(batches.len(), 3); // floor(100/32) = 3
        assert_eq!(batches[0].bytes.len(), 32);
    }

    #[test]
    fn should_checkpoint_fires_correctly() {
        let cfg = TrainConfig { checkpoint_every: 3, ..Default::default() };
        let mut state = TrainState::new(cfg);
        assert!(!state.should_checkpoint());
        let mut gamma = 0.0f32;
        for _ in 0..3 {
            state.update_gamma(&mut gamma, 0.5);
        }
        assert!(state.should_checkpoint());
    }
}
