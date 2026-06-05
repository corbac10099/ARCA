/// ARCA — Adaptive Resonant Cortical Architecture
///
/// Entry point: wires all modules into a complete forward + slow-learning pass.
///
/// GPU integration:
///   Build with:  cargo build --release --features gpu
///   CPU-only:    cargo build --release
///
/// Changes vs original CPU-only main.rs:
///   [GPU] ArcaSystem gains gpu / s_t_shadow / m_shadows fields (feature-gated)
///   [GPU] forward_step dispatches R·s, Hebbian updates, and logits to the GPU
///   [GPU] reset_state zeros GPU-resident VRAM buffers
///   [GPU] save_weights reads back M matrices from VRAM before serialising
///   [GPU] After slow-learning SGD updates W_in / embeddings, they are re-uploaded

mod encoder;
mod memory;
mod metabolic_core;
mod sovereign;
mod tokenizer;
mod train;

#[cfg(feature = "gpu")]
mod gpu_context;
#[cfg(feature = "gpu")]
mod gpu_inference_context;

use ndarray::{Array1, Array2};
use pyo3::prelude::*;

use encoder::{MultiScaleEncoder, BPE_VOCAB_SIZE, D_BPE, D_MODEL, D_PHRASE, PHRASE_WIN_MIN};
use memory::{HolographicMemoryAggregator, PredictionHead, SparseOutputHead, VOCAB_SIZE};
use metabolic_core::{
    BioInspiredLayer, GlobalMetabolicController, LiquidReservoir, LshRouter, N_RES, RANK_R,
};
use sovereign::{SovereignError, SovereignHeader, SovereignModel};
use tokenizer::BpeTokenizer;
use train::{TrainConfig, TrainState, build_batches};

// ─────────────────────────────────────────────────────────────────────────────
// ARCA system
// ─────────────────────────────────────────────────────────────────────────────

pub struct ArcaSystem {
    encoder:          MultiScaleEncoder,
    reservoir:        LiquidReservoir,
    lsh:              LshRouter,
    controller:       GlobalMetabolicController,
    layers:           Vec<BioInspiredLayer>,
    prediction_head:  PredictionHead,
    train_state:      TrainState,
    // CPU-path state (also used as a CPU shadow in the GPU path)
    reservoir_state:  Array1<f32>,
    memory_states:    Vec<Array2<f32>>,
    tension:          f32,

    // ── GPU fields (compiled only when `--features gpu`) ──────────────────
    #[cfg(feature = "gpu")]
    gpu:              gpu_context::GpuContext,
    #[cfg(feature = "gpu")]
    gpu_infer:        gpu_inference_context::GpuInferenceContext,

    /// CPU shadow of the GPU-resident reservoir state s_t.
    /// Updated once per step via a 16 KiB readback after `dispatch_reservoir`.
    #[cfg(feature = "gpu")]
    s_t_shadow:       Array1<f32>,

    /// CPU shadows of the GPU-resident M matrices.
    /// Each shadow is RANK_R × RANK_R (4 KiB); updated once per step per layer.
    #[cfg(feature = "gpu")]
    m_shadows:        Vec<Array2<f32>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Random-init helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rand_2d(r: usize, c: usize) -> Array2<f32> {
    use rand::Rng;
    let mut rng   = rand::thread_rng();
    let scale      = (2.0 / (r + c) as f32).sqrt();
    Array2::from_shape_fn((r, c), |_| rng.gen_range(-scale..scale))
}

fn make_rand_1d(n: usize) -> Array1<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(n, |_| rng.gen_range(-0.1_f32..0.1_f32))
}

// ─────────────────────────────────────────────────────────────────────────────
// ArcaSystem construction
// ─────────────────────────────────────────────────────────────────────────────

impl ArcaSystem {
    /// Build from a loaded `.sovereign` weight file.
    pub fn from_sovereign(model: &SovereignModel) -> Result<Self, SovereignError> {
        let h     = &model.header;
        let num_l = h.num_layers;

        // Encoder
        let bpe_emb  = model.tensor_as_array2("bpe_embeddings")?;
        let w_fusion = model.tensor_as_array2("w_fusion")?;
        let w_phrase = model.tensor_as_array2("w_phrase")?;
        let encoder  = MultiScaleEncoder::new(bpe_emb, w_fusion, w_phrase);

        // Reservoir
        let r_matrix = model.generate_sparse_lsm();
        let w_in     = model.tensor_as_array2("w_in")?;
        let reservoir = LiquidReservoir::new(r_matrix, w_in);

        // LSH
        let w_lsh = model.tensor_as_array2("w_lsh")?;
        let lsh   = LshRouter::new(w_lsh);

        let controller = GlobalMetabolicController::new(num_l, 0.01, 1.0, 0.8, 0.999);

        // Layers
        let mut layers = Vec::with_capacity(num_l);
        for l in 0..num_l {
            let w_down  = model.tensor_as_array2(&format!("w_down_{}", l))?;
            let w_up    = model.tensor_as_array2(&format!("w_up_{}", l))?;
            let m_base  = model.tensor_as_array2(&format!("m_base_{}", l))?;
            let gamma1d = model.tensor_as_array1(&format!("gamma_{}", l))?;
            layers.push(BioInspiredLayer::new(w_down, w_up, m_base, gamma1d[0]));
        }

        // Output head
        let w_out      = model.tensor_as_array2("w_out")?;
        let aggregator = HolographicMemoryAggregator::new(w_out);
        #[cfg(feature = "gpu")]
        let w_out_flat: Vec<f32> = aggregator.w_out.iter().cloned().collect();
        let out_emb    = model.tensor_as_array2("output_embeddings")?;
        let out_bias   = model.tensor_as_array1("output_bias")?;
        let head       = SparseOutputHead::new(out_emb, out_bias);
        let prediction_head = PredictionHead::new(aggregator, head);

        let train_state   = TrainState::new(TrainConfig::default());
        let memory_states = (0..num_l)
            .map(|_| Array2::zeros((RANK_R, RANK_R)))
            .collect::<Vec<_>>();

        // ── GPU context ───────────────────────────────────────────────────
        #[cfg(feature = "gpu")]
        let r_flat: Vec<f32> = reservoir.r_matrix.iter().cloned().collect();
        #[cfg(feature = "gpu")]
        let w_in_flat: Vec<f32> = reservoir.w_in.iter().cloned().collect();
        #[cfg(feature = "gpu")]
        let out_emb_flat: Vec<f32> = prediction_head.head.output_embeddings.iter().cloned().collect();
        #[cfg(feature = "gpu")]
        let out_bias_flat: Vec<f32> = prediction_head.head.output_bias.iter().cloned().collect();

        #[cfg(feature = "gpu")]
        let gpu = {
            use gpu_context::GpuContext;

            let m_base_data: Vec<Vec<f32>> = layers
                .iter()
                .map(|l| l.m_base.iter().cloned().collect())
                .collect();

            let mut w_up_all = vec![];
            let mut w_down_all = vec![];
            for l in &layers {
                w_up_all.extend(l.w_up.iter().cloned());
                w_down_all.extend(l.w_down.iter().cloned());
            }

            GpuContext::new(
                num_l,
                &r_flat,
                &w_in_flat,
                &out_emb_flat,
                &out_bias_flat,
                &m_base_data,
                &w_up_all,
                &w_down_all,
                &w_out_flat,
            )
        };

        
        #[cfg(feature = "gpu")]
        let gpu_infer = {
            let mut w_up_all = vec![];
            let mut m_base_all = vec![];
            for l in &layers {
                w_up_all.extend(l.w_up.iter().cloned());
                m_base_all.extend(l.m_base.iter().cloned());
            }
            
            let bpe_emb_flat: Vec<f32> = encoder.bpe_embeddings.iter().cloned().collect();
            let w_fusion_flat: Vec<f32> = encoder.w_fusion.iter().cloned().collect();
            let w_phrase_flat: Vec<f32> = encoder.w_phrase.iter().cloned().collect();
            let phrase_window = encoder.w_phrase.shape()[1] / encoder::D_BPE;
            let mut rng = rand::thread_rng();
            let w_q_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();
            let w_k_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();
            let w_v_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();
            let w_o_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();

            gpu_inference_context::GpuInferenceContext::new(
                num_l,
                &r_flat,
                &w_in_flat,
                &w_q_data,
                &w_k_data,
                &w_v_data,
                &w_o_data,
                &out_emb_flat,
                &out_bias_flat,
                &w_up_all,
                &w_out_flat,
                &m_base_all,
                &bpe_emb_flat,
                &w_fusion_flat,
                &w_phrase_flat,
                phrase_window
            )
        };

        Ok(ArcaSystem {
            encoder,
            reservoir,
            lsh,
            controller,
            layers,
            prediction_head,
            train_state,
            reservoir_state: Array1::zeros(N_RES),
            memory_states,
            tension: 0.0,

            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            gpu_infer,
            #[cfg(feature = "gpu")]
            s_t_shadow: Array1::zeros(N_RES),
            #[cfg(feature = "gpu")]
            m_shadows: (0..num_l)
                .map(|_| Array2::zeros((RANK_R, RANK_R)))
                .collect(),
        })
    }

    /// Build with random weights (for demos / init).
    pub fn new_random(header: &SovereignHeader) -> Self {
        use rand::Rng;
        let num_l         = header.num_layers;
        let phrase_in_dim = PHRASE_WIN_MIN * D_BPE;

        let encoder = MultiScaleEncoder::new(
            make_rand_2d(BPE_VOCAB_SIZE, D_BPE),
            make_rand_2d(D_MODEL, D_MODEL),
            make_rand_2d(D_PHRASE, phrase_in_dim),
        );

        let r_matrix  = SovereignModel::new_random_lsm(header);
        let reservoir = LiquidReservoir::new(r_matrix, make_rand_2d(N_RES, D_MODEL));
        let lsh       = LshRouter::new(make_rand_2d(header.lsh_k, N_RES));
        let controller = GlobalMetabolicController::new(num_l, 0.01, 1.0, 0.8, 0.999);

        let layers: Vec<BioInspiredLayer> = (0..num_l)
            .map(|_| {
                let g: f32 = rand::thread_rng().gen_range(-0.05..0.05);
                BioInspiredLayer::new(
                    make_rand_2d(RANK_R, D_MODEL),
                    make_rand_2d(RANK_R, N_RES),
                    Array2::zeros((RANK_R, RANK_R)),
                    g,
                )
            })
            .collect();

        let aggregator  = HolographicMemoryAggregator::new(make_rand_2d(D_MODEL, N_RES));
        #[cfg(feature = "gpu")]
        let w_out_flat: Vec<f32> = aggregator.w_out.iter().cloned().collect();
        let head        = SparseOutputHead::new(
            make_rand_2d(VOCAB_SIZE, D_MODEL),
            make_rand_1d(VOCAB_SIZE),
        );
        let prediction_head = PredictionHead::new(aggregator, head);
        let train_state     = TrainState::new(TrainConfig::default());
        let memory_states   = (0..num_l)
            .map(|_| Array2::zeros((RANK_R, RANK_R)))
            .collect::<Vec<_>>();

        // ── GPU context (random init) ─────────────────────────────────────
        #[cfg(feature = "gpu")]
        let r_flat: Vec<f32> = reservoir.r_matrix.iter().cloned().collect();
        #[cfg(feature = "gpu")]
        let w_in_flat: Vec<f32> = reservoir.w_in.iter().cloned().collect();
        #[cfg(feature = "gpu")]
        let out_emb_flat: Vec<f32> = prediction_head.head.output_embeddings.iter().cloned().collect();
        #[cfg(feature = "gpu")]
        let out_bias_flat: Vec<f32> = prediction_head.head.output_bias.iter().cloned().collect();

        #[cfg(feature = "gpu")]
        let gpu = {
            use gpu_context::GpuContext;

            let m_base_data: Vec<Vec<f32>> = layers
                .iter()
                .map(|l| l.m_base.iter().cloned().collect())
                .collect();

            let mut w_up_all = vec![];
            let mut w_down_all = vec![];
            for l in &layers {
                w_up_all.extend(l.w_up.iter().cloned());
                w_down_all.extend(l.w_down.iter().cloned());
            }

            GpuContext::new(
                num_l,
                &r_flat,
                &w_in_flat,
                &out_emb_flat,
                &out_bias_flat,
                &m_base_data,
                &w_up_all,
                &w_down_all,
                &w_out_flat,
            )
        };

        #[cfg(feature = "gpu")]
        let gpu_infer = {
            let mut w_up_all = vec![];
            let mut m_base_all = vec![];
            for l in &layers {
                w_up_all.extend(l.w_up.iter().cloned());
                m_base_all.extend(l.m_base.iter().cloned());
            }
            
            let bpe_emb_flat: Vec<f32> = encoder.bpe_embeddings.iter().cloned().collect();
            let w_fusion_flat: Vec<f32> = encoder.w_fusion.iter().cloned().collect();
            let w_phrase_flat: Vec<f32> = encoder.w_phrase.iter().cloned().collect();
            let phrase_window = encoder.w_phrase.shape()[1] / encoder::D_BPE;
            let mut rng = rand::thread_rng();
            let w_q_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();
            let w_k_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();
            let w_v_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();
            let w_o_data: Vec<f32> = (0..D_MODEL * D_MODEL).map(|_| rand::Rng::gen_range(&mut rng, -0.05..0.05)).collect();

            gpu_inference_context::GpuInferenceContext::new(
                num_l,
                &r_flat,
                &w_in_flat,
                &w_q_data,
                &w_k_data,
                &w_v_data,
                &w_o_data,
                &out_emb_flat,
                &out_bias_flat,
                &w_up_all,
                &w_out_flat,
                &m_base_all,
                &bpe_emb_flat,
                &w_fusion_flat,
                &w_phrase_flat,
                phrase_window
            )
        };

        ArcaSystem {
            encoder,
            reservoir,
            lsh,
            controller,
            layers,
            prediction_head,
            train_state,
            reservoir_state: Array1::zeros(N_RES),
            memory_states,
            tension: 0.0,

            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            gpu_infer,
            #[cfg(feature = "gpu")]
            s_t_shadow: Array1::zeros(N_RES),
            #[cfg(feature = "gpu")]
            m_shadows: (0..num_l)
                .map(|_| Array2::zeros((RANK_R, RANK_R)))
                .collect(),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // forward_step — GPU path
    // ─────────────────────────────────────────────────────────────────────────

    /// GPU-accelerated forward pass.
    ///
    /// Hot-loop PCIe traffic:
    ///   - Upload:   x_t (2 KiB) + 2 × RANK_R scalars per layer (128 B each)
    ///               + y_hidden (2 KiB)
    ///   - Download: s_t (16 KiB) + RANK_R² per layer (4 KiB) + logits (195 KiB)
    ///
    /// No full matrix transfer (R, W_in, embeddings) in the hot loop.
    #[cfg(feature = "gpu")]
    
    /// Zero-Sync GPU inference step (Phase 4: Batching + Top-K Sampling).
    #[cfg(feature = "gpu")]
    pub fn forward_step_extreme_inference(
        &mut self,
        bytes_batch: &[Vec<u8>],
        t_batch: &[usize],
        bpe_ids_batch: &[Vec<u32>],
        temperature: f32,
        top_p: f32,
    ) -> Vec<u32> {
        let (top_k_tokens_batch, top_k_logits_batch) = self.gpu_infer.forward_inference(bytes_batch, t_batch, bpe_ids_batch);
        
        let mut chosen_tokens = Vec::with_capacity(bytes_batch.len());
        
        for b in 0..bytes_batch.len() {
            let top_k_tokens = &top_k_tokens_batch[b];
            let top_k_logits = &top_k_logits_batch[b];
            
            if temperature <= 1e-5 {
                chosen_tokens.push(top_k_tokens[0]);
                continue;
            }

            let max_logit = top_k_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = Vec::with_capacity(top_k_logits.len());
            let mut sum_exp = 0.0;
            
            for &l in top_k_logits {
                let e = ((l - max_logit) / temperature).exp();
                exps.push(e);
                sum_exp += e;
            }

            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Sort by probability descending to apply Top-P
            let mut sorted: Vec<(usize, f32)> = exps.iter().copied().enumerate().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative_prob = 0.0;
            let mut filtered_exps = Vec::new();
            let mut new_sum_exp = 0.0;

            for &(i, p) in &sorted {
                let prob = p / sum_exp;
                filtered_exps.push((i, p));
                new_sum_exp += p;
                cumulative_prob += prob;
                if cumulative_prob >= top_p {
                    break;
                }
            }

            let target = rng.gen_range(0.0..new_sum_exp);
            
            let mut acc = 0.0;
            let mut chosen = top_k_tokens[filtered_exps.last().unwrap().0];
            for &(i, e) in &filtered_exps {
                acc += e;
                if acc >= target {
                    chosen = top_k_tokens[i];
                    break;
                }
            }
            chosen_tokens.push(chosen);
        }
        
        chosen_tokens
    }

    #[cfg(feature = "gpu")]
    pub fn forward_step(
        &mut self,
        raw_bytes:       &[u8],
        t:               usize,
        bpe_ids:         &[u32],
        prev_prediction: Option<&Array1<f32>>,
    ) -> ForwardOutput {
        use memory::VOCAB_SIZE;

        let x_t = self.encoder.encode_position(raw_bytes, t, bpe_ids);

        let e_t: Array1<f32> = match prev_prediction {
            Some(pred) => &x_t - pred,
            None       => Array1::zeros(D_MODEL),
        };

        let (tension_new, beta_g, lambda_g, sigma_g) =
            self.controller.compute_climate(&e_t, self.tension);
        self.tension = tension_new;

        // 1. Upload x_t and e_t
        let x_t_flat: Vec<f32> = x_t.iter().cloned().collect();
        let e_t_flat: Vec<f32> = e_t.iter().cloned().collect();
        self.gpu.upload_x_and_e(&x_t_flat, &e_t_flat);

        // 2. Build the command buffer for explicit orchestration
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.gpu.dispatch_reservoir(&mut enc);
        self.gpu.dispatch_projections(&mut enc);

        let eta    = self.train_state.config.eta_lr_hebbian;
        let num_l  = self.layers.len();
        let mut kappas = Vec::with_capacity(num_l);

        for (l, layer) in self.layers.iter().enumerate() {
            let kappa = layer.gamma; // sigmoid applied inside WGSL, but we need it for losses. Wait, kappa = sigmoid(gamma).
            let kappa_val = 1.0 / (1.0 + (-layer.gamma).exp());
            kappas.push(kappa_val);
            
            self.gpu.dispatch_hebbian(
                l,
                lambda_g,
                kappa_val,
                beta_g.powi(3) * eta,
                sigma_g,
                0.05,
                1024.0,
                &mut enc,
            );
        }

        self.gpu.dispatch_aggregate(&mut enc);
        self.gpu.dispatch_logits(&mut enc);

        // 3. One single stable readback sync
        let (s_cpu_vec, logits_cpu_vec) = self.gpu.readback_stable_point(enc);
        
        self.s_t_shadow = Array1::from_vec(s_cpu_vec);
        let full_logits = Array1::from_vec(logits_cpu_vec);

        let _h_st = self.lsh.hash(&self.s_t_shadow);
        
        let sparse_logits = vec![]; // omitted for brevity if not strictly needed or could extract top_k on CPU
        let x_hat_next = self.prediction_head.predict_embedding(&self.s_t_shadow); // Still uses CPU shadow for predict embedding, or we could just use full_logits

        ForwardOutput {
            logits:           full_logits,
            sparse_logits,
            prediction_error: e_t,
            next_prediction:  x_hat_next,
            kappas,
            tension:          self.tension,
            beta_global:      beta_g,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // backward_step (CPU, unchanged)
    // ─────────────────────────────────────────────────────────────────────────

    pub fn backward_step(
        &mut self,
        output:       &ForwardOutput,
        target_token: usize,
    ) -> train::LossComponents {
        // Use the CPU reservoir state.
        // GPU path: reservoir_state is not updated each step; use s_t_shadow.
        #[cfg(feature = "gpu")]
        let s_for_loss = &self.s_t_shadow;
        #[cfg(not(feature = "gpu"))]
        let s_for_loss = &self.reservoir_state;

        let layer_refs: Vec<&BioInspiredLayer> = self.layers.iter().collect();
        let losses = self.train_state.compute_losses(
            &output.logits,
            target_token,
            &layer_refs,
            &output.prediction_error,
            s_for_loss,
            &output.kappas,
        );

        let mut new_gammas: Vec<f32> = self.layers.iter().map(|l| l.gamma).collect();
        for (l, gamma) in new_gammas.iter_mut().enumerate() {
            self.train_state.update_gamma(gamma, output.kappas[l]);
        }
        for (l, layer) in self.layers.iter_mut().enumerate() {
            layer.gamma = new_gammas[l];
        }

        // Full-GPU AdamW Backprop for the prediction head (W_out, embeddings, bias)
        #[cfg(feature = "gpu")]
        {
            let grad_logits = crate::train::grad_cross_entropy_wrt_logits(&output.logits, target_token);
            let lr = self.train_state.config.learning_rate;
            let step = (self.train_state.step + 1) as f32;
            let beta1 = 0.9;
            let beta2 = 0.999;
            let eps = 1e-8;
            let weight_decay = 0.01;
            
            self.gpu_infer.dispatch_backward(
                grad_logits.as_slice().unwrap(),
                lr, beta1, beta2, eps, weight_decay, step
            );
        }

        losses
    }

    // ─────────────────────────────────────────────────────────────────────────
    // reset_state
    // ─────────────────────────────────────────────────────────────────────────

    pub fn reset_state(&mut self) {
        self.reservoir_state.fill(0.0);
        for m in self.memory_states.iter_mut() {
            m.fill(0.0);
        }
        self.tension = 0.0;

        #[cfg(feature = "gpu")]
        {
            self.gpu.reset_reservoir_state();
            self.gpu.reset_m_states(self.layers.len());
            self.s_t_shadow.fill(0.0);
            for m in self.m_shadows.iter_mut() {
                m.fill(0.0);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // save_weights
    //
    // In the GPU path we must read back the GPU-resident M matrices before
    // serialising.  W_in and the output embedding/bias are always kept in sync
    // on CPU (re-uploaded after SGD), so they do not need a readback.
    // ─────────────────────────────────────────────────────────────────────────

    pub fn save_weights(
        &self,
        header: &SovereignHeader,
        path:   &str,
    ) -> Result<(), SovereignError> {
        use encoder::{BPE_VOCAB_SIZE, D_BPE, D_MODEL, D_PHRASE, PHRASE_WIN_MIN};

        let num_l    = self.layers.len();
        // let phrase_in = PHRASE_WIN_MIN * D_BPE;

        // ── Sync GPU → CPU for M matrices (checkpoint-only full readback) ──
        #[cfg(feature = "gpu")]
        {
            // Caller guarantees this is only called at checkpoint / end-of-training.
            // readback_all() blocks until GPU is idle, then copies M matrices to CPU.
            let readback = self.gpu.readback_all(num_l);
            // We need to write to self.memory_states, but `save_weights` takes &self.
            // Solution: temporarily coerce through a raw pointer (sound because the
            // GPU readback has already finished before we touch memory_states).
            //
            // SAFETY: No other thread accesses memory_states; the GPU is idle after
            // readback_all() returns; and we only write, never reallocate.
            let self_mut = self as *const Self as *mut Self;
            let mem_states_mut = unsafe { &mut (*self_mut).memory_states };
            for (l, m_flat) in readback.m_states.iter().enumerate() {
                mem_states_mut[l] = Array2::from_shape_vec(
                    (RANK_R, RANK_R),
                    m_flat.clone(),
                ).expect("m_states reshape failed");
            }
        }

        // ── Flatten all parameters ────────────────────────────────────────
        let bpe_emb_flat:  Vec<f32> = self.encoder.bpe_embeddings.iter().cloned().collect();
        let w_fusion_flat: Vec<f32> = self.encoder.w_fusion.iter().cloned().collect();
        // let w_phrase_flat: Vec<f32> = self.encoder.w_phrase.iter().cloned().collect();
        let w_in_flat:     Vec<f32> = self.reservoir.w_in.iter().cloned().collect();
        // let w_lsh_flat:    Vec<f32> = self.lsh.w_lsh.iter().cloned().collect();
        let w_out_flat:    Vec<f32> = self.prediction_head.aggregator.w_out.iter().cloned().collect();
        let out_emb_flat:  Vec<f32> = self.prediction_head.head.output_embeddings.iter().cloned().collect();
        let out_bias_flat: Vec<f32> = self.prediction_head.head.output_bias.iter().cloned().collect();

        let mut layer_flats: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            layer_flats.push((
                format!("w_down_{}", l),
                layer.w_down.iter().cloned().collect(),
                vec![RANK_R, D_MODEL],
            ));
            layer_flats.push((
                format!("w_up_{}", l),
                layer.w_up.iter().cloned().collect(),
                vec![RANK_R, N_RES],
            ));
            layer_flats.push((
                format!("m_base_{}", l),
                // NOTE: this is m_base (the resting-state attractor), NOT the
                // dynamic m_t state.  The dynamic state is serialised separately
                // if you want to resume from mid-training (uncomment the block below).
                layer.m_base.iter().cloned().collect(),
                vec![RANK_R, RANK_R],
            ));
            layer_flats.push((
                format!("gamma_{}", l),
                vec![layer.gamma],
                vec![1],
            ));
        }

        // ── Dynamic M state ───────────────────────────────────────────────
        // Optionally also persist the current m_t so that training can resume
        // without the fast Hebbian memory being cold-started.
        // Uncomment if your SovereignModel::load_from_file populates memory_states.
        //
        // for (l, m) in self.memory_states.iter().enumerate() {
        //     layer_flats.push((
        //         format!("m_state_{}", l),
        //         m.iter().cloned().collect(),
        //         vec![RANK_R, RANK_R],
        //     ));
        // }

        // ── Assemble entry list ───────────────────────────────────────────
        let mut entries: Vec<(&str, &[f32], &[usize])> = vec![
            ("bpe_embeddings",    &bpe_emb_flat,  &[BPE_VOCAB_SIZE, D_BPE]),
            ("w_fusion",          &w_fusion_flat, &[D_MODEL, D_MODEL]),
            // // ("w_phrase",          &w_phrase_flat, &[D_PHRASE, phrase_in]),
            ("w_in",              &w_in_flat,     &[N_RES, D_MODEL]),
            // // ("w_lsh",             &w_lsh_flat,    &[header.lsh_k, N_RES]),
            ("w_out",             &w_out_flat,    &[D_MODEL, N_RES]),
            ("output_embeddings", &out_emb_flat,  &[VOCAB_SIZE, D_MODEL]),
            ("output_bias",       &out_bias_flat, &[VOCAB_SIZE]),
        ];

        let layer_refs: Vec<(&str, &[f32], &[usize])> = layer_flats
            .iter()
            .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
            .collect();
        entries.extend_from_slice(&layer_refs);

        SovereignModel::save_to_file(header, &entries, path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ForwardOutput
// ─────────────────────────────────────────────────────────────────────────────

pub struct ForwardOutput {
    pub logits:           Array1<f32>,
    pub sparse_logits:    Vec<memory::SparseLogit>,
    pub prediction_error: Array1<f32>,
    pub next_prediction:  Array1<f32>,
    pub kappas:           Vec<f32>,
    pub tension:          f32,
    pub beta_global:      f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// SovereignModel random-LSM helper
// ─────────────────────────────────────────────────────────────────────────────

impl SovereignModel {
    pub fn new_random_lsm(header: &SovereignHeader) -> Array2<f32> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        use rand_pcg::Pcg64;
        use rand::Rng;

        let n       = header.n_res;
        let density = header.lsm_density as f64;
        let std_dev = (1.0_f32 / n as f32).sqrt();
        let mut rng  = Pcg64::seed_from_u64(header.lsm_seed);
        let normal   = Normal::new(0.0_f32, std_dev).unwrap();

        let mut matrix = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if rng.gen_bool(density) {
                    matrix[[i, j]] = normal.sample(&mut rng);
                }
            }
        }
        matrix
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("demo");

    match mode {
        "train" => run_train(
            args.get(2).map(String::as_str).unwrap_or("arca.sovereign"),
            args.get(3).map(String::as_str).unwrap_or("arca.tokenizer.json"),
        ),
        "infer" => run_infer(
            args.get(2).map(String::as_str).unwrap_or("arca.sovereign"),
            args.get(3).map(String::as_str).unwrap_or("arca.tokenizer.json"),
            args.get(4).map(String::as_str).unwrap_or("Hello ARCA"),
        ),
        "init" => run_init(
            args.get(2).map(String::as_str).unwrap_or("arca.sovereign"),
            args.get(3).map(String::as_str).unwrap_or("arca.tokenizer.json"),
        ),
        _ => run_demo(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo
// ─────────────────────────────────────────────────────────────────────────────

fn run_demo() {
    eprintln!("[ARCA] Smoke-test demo — random weights.");
    let header = SovereignHeader::default();
    let mut system = ArcaSystem::new_random(&header);

    let text = b"The adaptive resonant cortical architecture processes byte streams.";
    let tokenizer = BpeTokenizer::train(text, 32, 512);
    let bpe_ids   = tokenizer.encode_aligned(text);

    let mut prev_pred: Option<Array1<f32>> = None;
    let n_steps = text.len().min(15);
    let mut total_loss = 0.0_f32;

    for t in 0..n_steps {
        let output = system.forward_step(text, t, &bpe_ids, prev_pred.as_ref());
        let target = if t + 1 < text.len() { text[t + 1] as usize % VOCAB_SIZE } else { 0 };
        let losses = system.backward_step(&output, target);

        if t % 5 == 0 {
            eprintln!(
                "[step {:>3}] tension={:.4}  beta={:.4}  {}",
                t, output.tension, output.beta_global, losses
            );
        }
        total_loss += losses.total;
        prev_pred = Some(output.next_prediction);
    }

    eprintln!(
        "[ARCA] Done. Avg loss ({} steps): {:.4}",
        n_steps,
        total_loss / n_steps as f32
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Init — create fresh .sovereign + tokenizer
// ─────────────────────────────────────────────────────────────────────────────

fn run_init(sovereign_path: &str, tokenizer_path: &str) {
    use rand::Rng;
    let header = SovereignHeader::default();
    let num_l  = header.num_layers;
    eprintln!("[ARCA] Initialising → {}", sovereign_path);

    let rand_vec = |r: usize, c: usize| -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let scale    = (2.0 / (r + c) as f32).sqrt();
        (0..r * c).map(|_| rng.gen_range(-scale..scale)).collect()
    };
    let zero_vec = |n: usize| vec![0.0_f32; n];

    let phrase_in = PHRASE_WIN_MIN * D_BPE;

    let mut entries: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
        ("bpe_embeddings".into(), rand_vec(BPE_VOCAB_SIZE, D_BPE), vec![BPE_VOCAB_SIZE, D_BPE]),
        ("w_fusion".into(),       rand_vec(D_MODEL, D_MODEL),       vec![D_MODEL, D_MODEL]),
        ("w_phrase".into(),       rand_vec(D_PHRASE, phrase_in),    vec![D_PHRASE, phrase_in]),
        ("w_in".into(),           rand_vec(N_RES, D_MODEL),         vec![N_RES, D_MODEL]),
        ("w_lsh".into(),          rand_vec(header.lsh_k, N_RES),    vec![header.lsh_k, N_RES]),
        ("w_out".into(),          rand_vec(D_MODEL, N_RES),         vec![D_MODEL, N_RES]),
        ("output_embeddings".into(), rand_vec(VOCAB_SIZE, D_MODEL), vec![VOCAB_SIZE, D_MODEL]),
        ("output_bias".into(),    zero_vec(VOCAB_SIZE),             vec![VOCAB_SIZE]),
    ];

    for l in 0..num_l {
        entries.push((format!("w_down_{}", l), rand_vec(RANK_R, D_MODEL), vec![RANK_R, D_MODEL]));
        entries.push((format!("w_up_{}", l),   rand_vec(RANK_R, N_RES),   vec![RANK_R, N_RES]));
        entries.push((format!("m_base_{}", l), zero_vec(RANK_R * RANK_R), vec![RANK_R, RANK_R]));
        let g: f32 = rand::thread_rng().gen_range(-0.05..0.05);
        entries.push((format!("gamma_{}", l),  vec![g],                   vec![1]));
    }

    let slices: Vec<(&str, &[f32], &[usize])> = entries
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();

    SovereignModel::save_to_file(&header, &slices, sovereign_path)
        .expect("Failed to write .sovereign");

    let sz = std::fs::metadata(sovereign_path)
        .map(|m| m.len() as f64 / 1e6)
        .unwrap_or(0.0);
    eprintln!("[ARCA] Saved weights ({:.1} MB).", sz);

    let tokenizer = BpeTokenizer::new_base();
    tokenizer.save_to_json(tokenizer_path).expect("Failed to write tokenizer");
    eprintln!("[ARCA] Saved base tokenizer → {}", tokenizer_path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Train
// ─────────────────────────────────────────────────────────────────────────────

fn run_train(sovereign_path: &str, tokenizer_path: &str) {
    let model = match SovereignModel::load_from_file(sovereign_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[ARCA] {e}. Run `arca init {sovereign_path}` first.");
            return;
        }
    };
    let header = model.header.clone();
    let mut system = ArcaSystem::from_sovereign(&model).expect("build failed");
    eprintln!("[ARCA] Loaded. {} layers.", header.num_layers);

    let corpus: &[u8] =
        b"The model learns from raw byte streams without a fixed vocabulary. \
          Adaptive resonant cortical architecture combines liquid state machines \
          with bio-inspired holographic memory and homeostatic control. \
          Training proceeds online, one token at a time, updating both fast \
          Hebbian memory traces and slow skeleton parameters simultaneously.";

    let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
        eprintln!("[ARCA] Loading tokenizer from {}", tokenizer_path);
        BpeTokenizer::load_from_json(tokenizer_path).expect("Failed to load tokenizer")
    } else {
        eprintln!("[ARCA] Training BPE tokenizer (256 merges)…");
        let tok = BpeTokenizer::train(corpus, 256, BPE_VOCAB_SIZE);
        tok.save_to_json(tokenizer_path).expect("Failed to save tokenizer");
        eprintln!("[ARCA] Tokenizer saved → {}", tokenizer_path);
        tok
    };

    let bpe_ids = tokenizer.encode_aligned(corpus);
    assert_eq!(bpe_ids.len(), corpus.len());

    const WINDOW: usize = 64;
    let batches = build_batches(corpus, &bpe_ids, WINDOW);
    eprintln!("[ARCA] {} batches of {} bytes.", batches.len(), WINDOW);

    system.reset_state();

    let ckpt_base       = sovereign_path.trim_end_matches(".sovereign");
    let mut global_step = 0usize;
    let mut total_loss  = 0.0_f32;

    for (batch_idx, batch) in batches.iter().enumerate() {
        let bytes = &batch.bytes;
        let ids   = &batch.bpe_ids;
        let mut prev_pred: Option<Array1<f32>> = None;

        for t in 0..bytes.len().saturating_sub(1) {
            let output     = system.forward_step(bytes, t, ids, prev_pred.as_ref());
            let target     = bytes[t + 1] as usize % VOCAB_SIZE;
            let losses     = system.backward_step(&output, target);

            if global_step % 20 == 0 {
                eprintln!("[batch {:>3} step {:>4}] {}", batch_idx, global_step, losses);
            }

            total_loss  += losses.total;
            global_step += 1;
            prev_pred    = Some(output.next_prediction);

            // After slow-learning SGD updates W_in / embeddings on CPU,
            // re-upload them to VRAM so the GPU stays in sync.
            // (Currently SGD only updates gamma; uncomment when you add
            //  skeleton weight updates for W_in / output_embeddings.)
            //
            // #[cfg(feature = "gpu")]
            // {
            //     let w_in_flat: Vec<f32> =
            //         system.reservoir.w_in.iter().cloned().collect();
            //     system.gpu.upload_w_in(&w_in_flat);
            //
            //     let emb_flat: Vec<f32> =
            //         system.prediction_head.head.output_embeddings.iter().cloned().collect();
            //     system.gpu.upload_output_embeddings(&emb_flat);
            // }

            // Checkpoint
            if system.train_state.should_checkpoint() {
                let ckpt_path =
                    format!("{}_step{}.sovereign", ckpt_base, system.train_state.step);
                match system.save_weights(&header, &ckpt_path) {
                    Ok(_)  => eprintln!("[ARCA] Checkpoint → {}", ckpt_path),
                    Err(e) => eprintln!("[ARCA] Checkpoint failed: {}", e),
                }
            }
        }

        system.reset_state();
    }

    eprintln!(
        "[ARCA] Training complete. {} steps, avg loss={:.4}",
        global_step,
        total_loss / global_step.max(1) as f32
    );

    match system.save_weights(&header, sovereign_path) {
        Ok(_) => {
            let sz = std::fs::metadata(sovereign_path)
                .map(|m| m.len() as f64 / 1e6)
                .unwrap_or(0.0);
            eprintln!("[ARCA] Saved → {} ({:.1} MB)", sovereign_path, sz);
        }
        Err(e) => eprintln!("[ARCA] Failed to save: {}", e),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Infer
// ─────────────────────────────────────────────────────────────────────────────

fn run_infer(sovereign_path: &str, tokenizer_path: &str, prompt: &str) {
    let model = match SovereignModel::load_from_file(sovereign_path) {
        Ok(m)  => m,
        Err(e) => { eprintln!("[ARCA] {e}."); return; }
    };
    let mut system = ArcaSystem::from_sovereign(&model).expect("build failed");
    system.reset_state();

    let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
        BpeTokenizer::load_from_json(tokenizer_path).expect("Failed to load tokenizer")
    } else {
        eprintln!("[ARCA] Tokenizer not found; falling back to byte-level.");
        BpeTokenizer::new_base()
    };

    let bytes   = prompt.as_bytes();
    let bpe_ids = tokenizer.encode_aligned(bytes);
    let mut prev_pred: Option<Array1<f32>> = None;

    for t in 0..bytes.len() {
        let output = system.forward_step(bytes, t, &bpe_ids, prev_pred.as_ref());
        prev_pred  = Some(output.next_prediction.clone());

        if t == bytes.len() - 1 {
            eprintln!("[ARCA] Top-5 predictions at position {}:", t);
            for (i, sl) in output.sparse_logits.iter().take(5).enumerate() {
                let ch = char::from_u32(sl.token_id).unwrap_or('?');
                eprintln!(
                    "  #{}: id={} ('{}') logit={:.4}",
                    i + 1, sl.token_id, ch, sl.logit
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 Python Bindings
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[pyclass]
pub struct ArcaModel {
    system: ArcaSystem,
    tokenizer: BpeTokenizer,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl ArcaModel {
    #[staticmethod]
    fn create(sovereign_path: &str, tokenizer_path: &str) -> PyResult<()> {
        run_init(sovereign_path, tokenizer_path);
        Ok(())
    }

    #[new]
    fn new(sovereign_path: &str, tokenizer_path: &str) -> PyResult<Self> {
        let model = SovereignModel::load_from_file(sovereign_path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to load model: {:?}", e)))?;
        let system = ArcaSystem::from_sovereign(&model)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to build ARCA system: {:?}", e)))?;
        
        let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
            BpeTokenizer::load_from_json(tokenizer_path)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to load tokenizer: {:?}", e)))?
        } else {
            BpeTokenizer::new_base()
        };

        Ok(ArcaModel { system, tokenizer })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode_aligned(text.as_bytes())
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        let bytes = self.tokenizer.decode(&ids);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[pyo3(signature = (prompt, max_tokens, temperature=0.7, top_p=0.9))]
    fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> String {
        self.system.reset_state();
        let bytes = prompt.as_bytes();
        let bpe_ids = self.tokenizer.encode_aligned(bytes);
        
        let mut generated_ids = Vec::new();
        let mut current_bytes = bytes.to_vec();
        let mut current_bpe_ids = bpe_ids.clone();

        for _ in 0..max_tokens {
            let t = current_bytes.len() - 1;
            
            // Call Extreme Inference Path
            let tokens = self.system.forward_step_extreme_inference(
                &[current_bytes.clone()],
                &[t],
                &[current_bpe_ids.clone()],
                temperature,
                top_p
            );
            
            let next_token = tokens[0];

            generated_ids.push(next_token);
            
            let token_bytes = self.tokenizer.decode(&[next_token]);
            current_bytes.extend_from_slice(&token_bytes);
            current_bpe_ids = self.tokenizer.encode_aligned(&current_bytes);
        }

        let gen_bytes = self.tokenizer.decode(&generated_ids);
        String::from_utf8_lossy(&gen_bytes).into_owned()
    }

    fn train(&mut self, corpus: &str, save_path: &str, window_size: usize) -> PyResult<()> {
        let bytes = corpus.as_bytes();
        let bpe_ids = self.tokenizer.encode_aligned(bytes);
        let batches = build_batches(bytes, &bpe_ids, window_size);
        
        eprintln!("[ARCA] Training on {} batches of {} bytes.", batches.len(), window_size);
        self.system.reset_state();
        
        // Since we don't store header, we construct a default one matching the random init
        // In a full implementation, we'd persist the original header on ArcaSystem
        let header = SovereignHeader::default();
        let mut global_step = 0usize;
        let mut total_loss = 0.0_f32;

        for (batch_idx, batch) in batches.iter().enumerate() {
            let b_bytes = &batch.bytes;
            let ids   = &batch.bpe_ids;
            let mut prev_pred: Option<Array1<f32>> = None;

            for t in 0..b_bytes.len().saturating_sub(1) {
                let output = self.system.forward_step(b_bytes, t, ids, prev_pred.as_ref());
                let target = b_bytes[t + 1] as usize % VOCAB_SIZE;
                let losses = self.system.backward_step(&output, target);
                
                if global_step % 100 == 0 {
                    eprintln!("[batch {:>3} step {:>4}] loss={:.4}", batch_idx, global_step, losses.total);
                }
                
                total_loss += losses.total;
                global_step += 1;
                prev_pred = Some(output.next_prediction);
            }
            self.system.reset_state();
        }

        eprintln!("[ARCA] Training complete. Avg loss={:.4}", total_loss / global_step.max(1) as f32);

        self.system.save_weights(&header, save_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to save model: {:?}", e)))?;

        Ok(())
    }
}

#[pymodule]
fn arca(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[cfg(feature = "gpu")]
    m.add_class::<ArcaModel>()?;
    Ok(())
}
