/// GPU integration additions for `main.rs`
///
/// This file shows the **new and changed sections** that must be merged into
/// `main.rs` when the `gpu` feature is enabled.  Lines that are identical to
/// the CPU-only version have been omitted; look for the `[GPU CHANGE]` markers.
///
/// Build command:
///   cargo build --release --features gpu
///
/// ─────────────────────────────────────────────────────────────────────────────
/// Modules — add `gpu_context` at the top of main.rs
/// ─────────────────────────────────────────────────────────────────────────────
///
/// ```rust
/// mod encoder;
/// mod memory;
/// mod metabolic_core;
/// mod sovereign;
/// mod tokenizer;
/// mod train;
/// #[cfg(feature = "gpu")]
/// mod gpu_context;           // ← [GPU CHANGE] add this line
/// ```
///
/// ─────────────────────────────────────────────────────────────────────────────
/// ArcaSystem struct — add GPU fields
/// ─────────────────────────────────────────────────────────────────────────────
///
/// Add inside `pub struct ArcaSystem { … }`:
///
/// ```rust
///     // [GPU CHANGE] owned GPU context (None when compiled without gpu feature)
///     #[cfg(feature = "gpu")]
///     gpu: gpu_context::GpuContext,
///
///     // [GPU CHANGE] CPU shadow of the reservoir state s_t (read back once per step)
///     #[cfg(feature = "gpu")]
///     s_t_shadow: ndarray::Array1<f32>,
///
///     // [GPU CHANGE] CPU shadows of M_t matrices for read-out (read back once per step)
///     #[cfg(feature = "gpu")]
///     m_shadows: Vec<ndarray::Array2<f32>>,
/// ```
///
/// ─────────────────────────────────────────────────────────────────────────────
/// from_sovereign — add GPU construction
/// ─────────────────────────────────────────────────────────────────────────────
///
/// Append at the end of `ArcaSystem::from_sovereign` before the `Ok(…)`:
///
/// ```rust
///     #[cfg(feature = "gpu")]
///     let gpu = {
///         use gpu_context::GpuContext;
///
///         let r_flat: Vec<f32> = system.reservoir.r_matrix.iter().cloned().collect();
///         let w_in_flat: Vec<f32> = system.reservoir.w_in.iter().cloned().collect();
///         let out_emb_flat: Vec<f32> = system.prediction_head.head
///             .output_embeddings.iter().cloned().collect();
///         let out_bias_flat: Vec<f32> = system.prediction_head.head
///             .output_bias.iter().cloned().collect();
///         let m_base_data: Vec<Vec<f32>> = system.layers.iter()
///             .map(|l| l.m_base.iter().cloned().collect())
///             .collect();
///
///         GpuContext::new(
///             system.layers.len(),
///             &r_flat,
///             &w_in_flat,
///             &out_emb_flat,
///             &out_bias_flat,
///             &m_base_data,
///         )
///     };
///
///     #[cfg(feature = "gpu")]
///     let num_layers = header.num_layers;
///
///     Ok(ArcaSystem {
///         // … existing fields …
///         #[cfg(feature = "gpu")]
///         gpu,
///         #[cfg(feature = "gpu")]
///         s_t_shadow: ndarray::Array1::zeros(metabolic_core::N_RES),
///         #[cfg(feature = "gpu")]
///         m_shadows: (0..num_layers)
///             .map(|_| ndarray::Array2::zeros((metabolic_core::RANK_R, metabolic_core::RANK_R)))
///             .collect(),
///     })
/// ```
///
/// ─────────────────────────────────────────────────────────────────────────────
/// GPU forward_step  ← the most important change
/// ─────────────────────────────────────────────────────────────────────────────

// The complete replacement for `ArcaSystem::forward_step` when compiled with
// the `gpu` feature.  Paste this method into the `impl ArcaSystem` block,
// replacing the existing `forward_step`.

/*
#[cfg(feature = "gpu")]
pub fn forward_step(
    &mut self,
    raw_bytes:       &[u8],
    t:               usize,
    bpe_ids:         &[u32],
    prev_prediction: Option<&ndarray::Array1<f32>>,
) -> ForwardOutput {
    use metabolic_core::{D_MODEL, N_RES, RANK_R};
    use memory::VOCAB_SIZE;

    let x_t = self.encoder.encode_position(raw_bytes, t, bpe_ids);

    // Prediction error (CPU — tiny)
    let e_t: ndarray::Array1<f32> = match prev_prediction {
        Some(pred) => &x_t - pred,
        None       => ndarray::Array1::zeros(D_MODEL),
    };

    // Homeostatic climate (scalar, CPU)
    let (tension_new, beta_g, lambda_g, sigma_g) =
        self.controller.compute_climate(&e_t, self.tension);
    self.tension = tension_new;

    // ── GPU: reservoir step ───────────────────────────────────────────────
    self.reservoir.step_gpu(&mut self.gpu, &x_t);

    // Read back s_t shadow for LSH + Hebbian projections (16 KiB, ~1 µs)
    let s_cpu_vec = self.gpu.readback_s();
    self.s_t_shadow = ndarray::Array1::from_vec(s_cpu_vec);

    // LSH hash on CPU shadow
    let _h_st = self.lsh.hash(&self.s_t_shadow);

    // ── GPU: per-layer Hebbian updates ────────────────────────────────────
    let eta  = self.train_state.config.eta_lr_hebbian;
    let mut kappas = Vec::with_capacity(self.layers.len());

    for (l, layer) in self.layers.iter().enumerate() {
        let depth_scale = 1.0 / (1.0 + l as f32 * 0.1);
        let e_local: ndarray::Array1<f32> = &e_t * depth_scale;

        let kappa = layer.forward_gpu(
            &mut self.gpu,
            l,
            &e_local,
            &self.s_t_shadow,
            beta_g,
            lambda_g,
            sigma_g,
            eta,
            0.05,
            1024.0,
        );
        kappas.push(kappa);
    }

    // Read back per-layer M shadows (RANK_R×RANK_R = 1 KiB each, ~0.1 µs per layer)
    for l in 0..self.layers.len() {
        let m_flat = gpu_context::map_read_f32_layer(&self.gpu, l);
        self.m_shadows[l] = ndarray::Array2::from_shape_vec(
            (RANK_R, RANK_R), m_flat
        ).expect("m_shadow reshape failed");
    }

    // Layer read-outs (CPU, on shadows)
    let layer_readouts: Vec<ndarray::Array1<f32>> = self.layers.iter()
        .enumerate()
        .map(|(l, layer)| layer.read_out_cpu(&self.m_shadows[l], &self.s_t_shadow))
        .collect();

    // ── GPU: logit computation ────────────────────────────────────────────
    let (full_logits, sparse_logits) =
        self.prediction_head.forward_gpu(&mut self.gpu, &self.s_t_shadow, &layer_readouts);

    // Update reservoir state reference for the next step
    // (s_t_shadow already holds the new state from readback_s above)

    let x_hat_next = self.prediction_head.predict_embedding(&self.s_t_shadow);

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
*/

/// ─────────────────────────────────────────────────────────────────────────────
/// reset_state — flush GPU state too
/// ─────────────────────────────────────────────────────────────────────────────
///
/// Replace the existing `reset_state` with:
///
/// ```rust
/// pub fn reset_state(&mut self) {
///     self.reservoir_state.fill(0.0);   // kept for CPU-path compat
///     for m in self.memory_states.iter_mut() { m.fill(0.0); }
///     self.tension = 0.0;
///
///     #[cfg(feature = "gpu")]
///     {
///         self.gpu.reset_reservoir_state();
///         self.gpu.reset_m_states(self.layers.len());
///         self.s_t_shadow.fill(0.0);
///         for m in self.m_shadows.iter_mut() { m.fill(0.0); }
///     }
/// }
/// ```
///
/// ─────────────────────────────────────────────────────────────────────────────
/// save_weights — read back from GPU before serialising
/// ─────────────────────────────────────────────────────────────────────────────
///
/// Add at the top of `save_weights`, before the existing flatten code:
///
/// ```rust
/// #[cfg(feature = "gpu")]
/// {
///     // Sync GPU weights back to CPU structs before flattening for disk.
///     // This is the ONE permitted full GPU→CPU transfer in the training loop,
///     // and it only happens at checkpoint intervals.
///     let readback = self.gpu.readback_all(self.layers.len());
///
///     // Note: R matrix is NOT saved (regenerated from seed).
///     // W_in, output_embeddings, output_bias are updated by slow-learning
///     // SGD on the CPU and re-uploaded; no GPU sync needed for them.
///     // The M matrices (m_states) are GPU-resident; sync them here.
///     for (l, m_flat) in readback.m_states.iter().enumerate() {
///         self.memory_states[l] = ndarray::Array2::from_shape_vec(
///             (metabolic_core::RANK_R, metabolic_core::RANK_R),
///             m_flat.clone(),
///         ).expect("m_states reshape failed");
///     }
/// }
/// ```
///
/// After slow-learning SGD updates W_in / output_embeddings on CPU, call:
///
/// ```rust
/// #[cfg(feature = "gpu")]
/// {
///     let w_in_flat: Vec<f32> = self.reservoir.w_in.iter().cloned().collect();
///     self.gpu.upload_w_in(&w_in_flat);
///
///     let emb_flat: Vec<f32> =
///         self.prediction_head.head.output_embeddings.iter().cloned().collect();
///     self.gpu.upload_output_embeddings(&emb_flat);
///
///     let bias_flat: Vec<f32> =
///         self.prediction_head.head.output_bias.iter().cloned().collect();
///     self.gpu.upload_output_bias(&bias_flat);
/// }
/// ```

// ── Helper: expose per-layer M readback from GpuContext ───────────────────
//
// Add this free function to gpu_context.rs (already present in the full
// gpu_context.rs delivered; shown here for reference):
//
// ```rust
// /// Read back the current M matrix for layer `l` (1 KiB per layer).
// /// Called every forward step to maintain the CPU shadow used by read_out_cpu.
// pub fn readback_m_layer(&self, layer_idx: usize) -> Vec<f32> {
//     use crate::gpu_context::{RANK_R, map_read_f32};
//     let l = layer_idx;
//     let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
//         label: Some(&format!("m-readback-enc-{}", l)),
//     });
//     enc.copy_buffer_to_buffer(
//         &self.buf_m[l][self.m_ping[l]], 0,
//         &self.buf_m_readback[l],        0,
//         (RANK_R * RANK_R * 4) as u64,
//     );
//     self.queue.submit(std::iter::once(enc.finish()));
//     map_read_f32(&self.device, &self.buf_m_readback[l], RANK_R * RANK_R)
// }
// ```

fn _integration_notes() {}
