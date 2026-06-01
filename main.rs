/// ARCA — Adaptive Resonant Cortical Architecture
///
/// Entry point: wires all modules into a complete forward + slow-learning pass.

mod encoder;
mod memory;
mod metabolic_core;
mod sovereign;
mod train;

use ndarray::{Array1, Array2};

use encoder::{MultiScaleEncoder, BPE_VOCAB_SIZE, D_BPE, D_MODEL, D_PHRASE, PHRASE_WIN_MIN};
use memory::{HolographicMemoryAggregator, PredictionHead, SparseOutputHead, VOCAB_SIZE};
use metabolic_core::{
    BioInspiredLayer, GlobalMetabolicController, LiquidReservoir, LshRouter, N_RES, RANK_R,
};
use sovereign::{SovereignError, SovereignHeader, SovereignModel};
use train::{TrainConfig, TrainState};

// ─────────────────────────────────────────────────────────────────────────────
// ARCA system
// ─────────────────────────────────────────────────────────────────────────────

pub struct ArcaSystem {
    encoder: MultiScaleEncoder,
    reservoir: LiquidReservoir,
    lsh: LshRouter,
    controller: GlobalMetabolicController,
    layers: Vec<BioInspiredLayer>,
    prediction_head: PredictionHead,
    train_state: TrainState,
    reservoir_state: Array1<f32>,
    memory_states: Vec<Array2<f32>>,
    tension: f32,
}

fn make_rand_2d(r: usize, c: usize) -> Array2<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let scale = (2.0 / (r + c) as f32).sqrt();
    Array2::from_shape_fn((r, c), |_| rng.gen_range(-scale..scale))
}

fn make_rand_1d(n: usize) -> Array1<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(n, |_| rng.gen_range(-0.1_f32..0.1_f32))
}

impl ArcaSystem {
    pub fn from_sovereign(model: &SovereignModel) -> Result<Self, SovereignError> {
        let h = &model.header;
        let num_l = h.num_layers;

        let bpe_emb = model.tensor_as_array2("bpe_embeddings")?;
        let w_fusion = model.tensor_as_array2("w_fusion")?;
        let w_phrase = model.tensor_as_array2("w_phrase")?;
        let encoder = MultiScaleEncoder::new(bpe_emb, w_fusion, w_phrase);

        let r_matrix = model.generate_sparse_lsm();
        let w_in = model.tensor_as_array2("w_in")?;
        let reservoir = LiquidReservoir::new(r_matrix, w_in);

        let w_lsh = model.tensor_as_array2("w_lsh")?;
        let lsh = LshRouter::new(w_lsh);

        let controller = GlobalMetabolicController::new(num_l, 0.01, 1.0, 0.8, 0.999);

        let mut layers = Vec::with_capacity(num_l);
        for l in 0..num_l {
            let w_down = model.tensor_as_array2(&format!("w_down_{}", l))?;
            let w_up = model.tensor_as_array2(&format!("w_up_{}", l))?;
            let m_base = model.tensor_as_array2(&format!("m_base_{}", l))?;
            let gamma_1d = model.tensor_as_array1(&format!("gamma_{}", l))?;
            layers.push(BioInspiredLayer::new(w_down, w_up, m_base, gamma_1d[0]));
        }

        let w_out = model.tensor_as_array2("w_out")?;
        let aggregator = HolographicMemoryAggregator::new(w_out);
        let out_emb = model.tensor_as_array2("output_embeddings")?;
        let out_bias = model.tensor_as_array1("output_bias")?;
        let head = SparseOutputHead::new(out_emb, out_bias);
        let prediction_head = PredictionHead::new(aggregator, head);
        let train_state = TrainState::new(TrainConfig::default());
        let memory_states = (0..num_l).map(|_| Array2::zeros((RANK_R, RANK_R))).collect();

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
        })
    }

    pub fn new_random(header: &SovereignHeader) -> Self {
        use rand::Rng;
        let num_l = header.num_layers;
        let phrase_input_dim = PHRASE_WIN_MIN * D_BPE;

        let encoder = MultiScaleEncoder::new(
            make_rand_2d(BPE_VOCAB_SIZE, D_BPE),
            make_rand_2d(D_MODEL, D_MODEL),
            make_rand_2d(D_PHRASE, phrase_input_dim),
        );

        let r_matrix = SovereignModel::new_random_lsm(header);
        let reservoir = LiquidReservoir::new(r_matrix, make_rand_2d(N_RES, D_MODEL));

        let lsh = LshRouter::new(make_rand_2d(header.lsh_k, N_RES));

        let controller = GlobalMetabolicController::new(num_l, 0.01, 1.0, 0.8, 0.999);

        let layers: Vec<BioInspiredLayer> = (0..num_l)
            .map(|_| {
                let gamma: f32 = rand::thread_rng().gen_range(-0.05..0.05);
                BioInspiredLayer::new(
                    make_rand_2d(RANK_R, D_MODEL),
                    make_rand_2d(RANK_R, N_RES),
                    Array2::zeros((RANK_R, RANK_R)),
                    gamma,
                )
            })
            .collect();

        let aggregator = HolographicMemoryAggregator::new(make_rand_2d(D_MODEL, N_RES));
        let head = SparseOutputHead::new(make_rand_2d(VOCAB_SIZE, D_MODEL), make_rand_1d(VOCAB_SIZE));
        let prediction_head = PredictionHead::new(aggregator, head);
        let train_state = TrainState::new(TrainConfig::default());
        let memory_states = (0..num_l).map(|_| Array2::zeros((RANK_R, RANK_R))).collect();

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
        }
    }

    pub fn forward_step(
        &mut self,
        raw_bytes: &[u8],
        t: usize,
        bpe_ids: &[u32],
        prev_prediction: Option<&Array1<f32>>,
    ) -> ForwardOutput {
        let x_t = self.encoder.encode_position(raw_bytes, t, bpe_ids);

        let e_t: Array1<f32> = match prev_prediction {
            Some(pred) => &x_t - pred,
            None => Array1::zeros(D_MODEL),
        };

        let (tension_new, beta_g, lambda_g, sigma_g) =
            self.controller.compute_climate(&e_t, self.tension);
        self.tension = tension_new;

        let s_t = self.reservoir.step(&self.reservoir_state, &x_t);
        let _h_st = self.lsh.hash(&s_t);

        let mut kappas = Vec::with_capacity(self.layers.len());
        let mut layer_readouts = Vec::with_capacity(self.layers.len());
        let eta = self.train_state.config.eta_lr_hebbian;

        for (l, layer) in self.layers.iter().enumerate() {
            let depth_scale = 1.0 / (1.0 + l as f32 * 0.1);
            let e_local: Array1<f32> = &e_t * depth_scale;

            let (m_new, kappa_l) = layer.forward_and_adapt(
                &self.memory_states[l],
                &e_local,
                &s_t,
                beta_g,
                lambda_g,
                sigma_g,
                eta,
                0.05,
                1024.0,
            );

            let readout = layer.read_out(&m_new, &s_t);
            layer_readouts.push(readout);
            kappas.push(kappa_l);
            self.memory_states[l] = m_new;
        }

        self.reservoir_state = s_t.clone();

        let (full_logits, sparse_logits) =
            self.prediction_head.forward(&s_t, &layer_readouts);

        let x_hat_next = self.prediction_head.predict_embedding(&s_t);

        ForwardOutput {
            logits: full_logits,
            sparse_logits,
            prediction_error: e_t,
            next_prediction: x_hat_next,
            kappas,
            tension: self.tension,
            beta_global: beta_g,
        }
    }

    pub fn backward_step(
        &mut self,
        output: &ForwardOutput,
        target_token: usize,
    ) -> train::LossComponents {
        let layer_refs: Vec<&BioInspiredLayer> = self.layers.iter().collect();
        let losses = self.train_state.compute_losses(
            &output.logits,
            target_token,
            &layer_refs,
            &output.prediction_error,
            &self.reservoir_state,
            &output.kappas,
        );

        for (l, layer) in self.layers.iter_mut().enumerate() {
            let kappa = output.kappas[l];
            self.train_state.update_gamma(&mut layer.gamma, kappa);
        }

        losses
    }

    pub fn reset_state(&mut self) {
        self.reservoir_state.fill(0.0);
        for m in self.memory_states.iter_mut() {
            m.fill(0.0);
        }
        self.tension = 0.0;
    }
}

pub struct ForwardOutput {
    pub logits: Array1<f32>,
    pub sparse_logits: Vec<memory::SparseLogit>,
    pub prediction_error: Array1<f32>,
    pub next_prediction: Array1<f32>,
    pub kappas: Vec<f32>,
    pub tension: f32,
    pub beta_global: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// SovereignModel helper (avoids file load for random init)
// ─────────────────────────────────────────────────────────────────────────────
impl SovereignModel {
    pub fn new_random_lsm(header: &SovereignHeader) -> Array2<f32> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        use rand_pcg::Pcg64;
        use rand::Rng;

        let n = header.n_res;
        let density = header.lsm_density as f64;
        let std_dev = (1.0_f32 / n as f32).sqrt();
        let mut rng = Pcg64::seed_from_u64(header.lsm_seed);
        let normal = Normal::new(0.0_f32, std_dev).unwrap();
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
        "train" => run_train(args.get(2).map(String::as_str).unwrap_or("arca.sovereign")),
        "infer" => run_infer(
            args.get(2).map(String::as_str).unwrap_or("arca.sovereign"),
            args.get(3).map(String::as_str).unwrap_or("Hello ARCA"),
        ),
        "init" => run_init(args.get(2).map(String::as_str).unwrap_or("arca.sovereign")),
        _ => run_demo(),
    }
}

fn run_demo() {
    eprintln!("[ARCA] Smoke-test demo — random weights, no file I/O.");
    let header = SovereignHeader::default();
    let mut system = ArcaSystem::new_random(&header);

    let text = b"The adaptive resonant cortical architecture processes byte streams.";
    let bpe_ids: Vec<u32> = text.iter().map(|&b| b as u32).collect();

    let mut prev_pred: Option<Array1<f32>> = None;
    let mut total_loss_sum = 0.0_f32;
    let n_steps = text.len().min(15);

    for t in 0..n_steps {
        let output = system.forward_step(text, t, &bpe_ids, prev_pred.as_ref());
        let target = if t + 1 < text.len() { text[t + 1] as usize } else { 0 };
        let losses = system.backward_step(&output, target);

        if t % 5 == 0 {
            eprintln!(
                "[step {:>3}] tension={:.4}  beta={:.4}  {}",
                t, output.tension, output.beta_global, losses
            );
        }
        total_loss_sum += losses.total;
        prev_pred = Some(output.next_prediction);
    }

    eprintln!(
        "[ARCA] Done. Avg loss ({} steps): {:.4}",
        n_steps, total_loss_sum / n_steps as f32
    );
}

fn run_init(path: &str) {
    use rand::Rng;
    let header = SovereignHeader::default();
    let num_l = header.num_layers;
    eprintln!("[ARCA] Initialising → {}", path);

    let rand_vec = |r: usize, c: usize| -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (r + c) as f32).sqrt();
        (0..r * c).map(|_| rng.gen_range(-scale..scale)).collect()
    };
    let zero_vec = |n: usize| vec![0.0_f32; n];

    let phrase_in = PHRASE_WIN_MIN * D_BPE;

    let mut entries: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
        ("bpe_embeddings".into(), rand_vec(BPE_VOCAB_SIZE, D_BPE), vec![BPE_VOCAB_SIZE, D_BPE]),
        ("w_fusion".into(), rand_vec(D_MODEL, D_MODEL), vec![D_MODEL, D_MODEL]),
        ("w_phrase".into(), rand_vec(D_PHRASE, phrase_in), vec![D_PHRASE, phrase_in]),
        ("w_in".into(), rand_vec(N_RES, D_MODEL), vec![N_RES, D_MODEL]),
        ("w_lsh".into(), rand_vec(header.lsh_k, N_RES), vec![header.lsh_k, N_RES]),
        ("w_out".into(), rand_vec(D_MODEL, N_RES), vec![D_MODEL, N_RES]),
        ("output_embeddings".into(), rand_vec(VOCAB_SIZE, D_MODEL), vec![VOCAB_SIZE, D_MODEL]),
        ("output_bias".into(), zero_vec(VOCAB_SIZE), vec![VOCAB_SIZE]),
    ];

    for l in 0..num_l {
        entries.push((format!("w_down_{}", l), rand_vec(RANK_R, D_MODEL), vec![RANK_R, D_MODEL]));
        entries.push((format!("w_up_{}", l), rand_vec(RANK_R, N_RES), vec![RANK_R, N_RES]));
        entries.push((format!("m_base_{}", l), zero_vec(RANK_R * RANK_R), vec![RANK_R, RANK_R]));
        let g: f32 = rand::thread_rng().gen_range(-0.05..0.05);
        entries.push((format!("gamma_{}", l), vec![g], vec![1]));
    }

    let slices: Vec<(&str, &[f32], &[usize])> = entries
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();

    SovereignModel::save_to_file(&header, &slices, path)
        .expect("Failed to write .sovereign");
    let sz = std::fs::metadata(path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
    eprintln!("[ARCA] Saved ({:.1} MB).", sz);
}

fn run_train(path: &str) {
    let model = match SovereignModel::load_from_file(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("[ARCA] {e}. Run `arca init {path}` first."); return; }
    };
    let mut system = ArcaSystem::from_sovereign(&model).expect("build failed");
    eprintln!("[ARCA] Loaded. {} layers.", model.header.num_layers);

    let corpus = b"The model learns from raw byte streams without a fixed vocabulary.";
    let bpe_ids: Vec<u32> = corpus.iter().map(|&b| b as u32).collect();
    system.reset_state();
    let mut prev_pred: Option<Array1<f32>> = None;

    for t in 0..corpus.len() - 1 {
        let output = system.forward_step(corpus, t, &bpe_ids, prev_pred.as_ref());
        let losses = system.backward_step(&output, corpus[t + 1] as usize);
        if t % 10 == 0 { eprintln!("[step {:>3}] {}", t, losses); }
        prev_pred = Some(output.next_prediction);
    }
    eprintln!("[ARCA] Training pass complete.");
}

fn run_infer(path: &str, prompt: &str) {
    let model = match SovereignModel::load_from_file(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("[ARCA] {e}."); return; }
    };
    let mut system = ArcaSystem::from_sovereign(&model).expect("build failed");
    system.reset_state();

    let bytes = prompt.as_bytes();
    let bpe_ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    let mut prev_pred: Option<Array1<f32>> = None;

    for t in 0..bytes.len() {
        let output = system.forward_step(bytes, t, &bpe_ids, prev_pred.as_ref());
        prev_pred = Some(output.next_prediction.clone());
        if t == bytes.len() - 1 {
            eprintln!("[ARCA] Top-5 predictions:");
            for (i, sl) in output.sparse_logits.iter().take(5).enumerate() {
                let ch = char::from_u32(sl.token_id).unwrap_or('?');
                eprintln!("  #{}: id={} ('{}') logit={:.4}", i+1, sl.token_id, ch, sl.logit);
            }
        }
    }
}
