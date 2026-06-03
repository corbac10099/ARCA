use arca::system::*;
use arca::encoder::*;
use arca::sovereign::*;
use arca::tokenizer::*;
use arca::memory::*;
use arca::metabolic_core::*;

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
