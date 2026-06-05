/// Extreme-Performance GPU Context — Zero-Sync Inference Pipeline
///
/// Refactored for ARCA Extreme GPU-First Inference.
/// 
/// All forward-pass computations are pushed to the GPU.
/// CPU orchestration only submits command buffers and reads back the final sampled token.

use wgpu::util::DeviceExt;

pub const N_RES:      usize = 4096;
pub const D_MODEL:    usize = 512;
pub const RANK_R:     usize = 32;
pub const VOCAB_SIZE: usize = 50_000;

#[inline(always)]
pub const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader sources
// ─────────────────────────────────────────────────────────────────────────────


pub const ATTENTION_SHADER: &str = r#"
fn get_f16_val(packed: u32, i: u32) -> f32 {
    let vec = unpack2x16float(packed);
    if (i % 2u) == 1u { return vec.y; }
    return vec.x;
}

const D_MODEL_C: u32 = 512u;
const MAX_SEQ_LEN_C: u32 = 1024u;
const HEADS_C: u32 = 8u;
const HEAD_DIM_C: u32 = 64u; // 512 / 8

@group(0) @binding(0) var<storage, read> x_t: array<f32>;
@group(0) @binding(1) var<storage, read> w_q: array<u32>;
@group(0) @binding(2) var<storage, read> w_k: array<u32>;
@group(0) @binding(3) var<storage, read> w_v: array<u32>;
@group(0) @binding(4) var<storage, read> w_o: array<u32>;
@group(0) @binding(5) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(7) var<storage, read_write> x_attn: array<f32>;

struct AttnParams {
    t: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(8) var<uniform> params: AttnParams;

var<workgroup> q_shared: array<f32, 512>;
var<workgroup> attn_scores: array<f32, 1024>; // MAX_SEQ_LEN
var<workgroup> attn_out: array<f32, 512>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let b = gid.y;
    
    // 1. Compute Q, K, V for this token
    for (var i = tid; i < D_MODEL_C; i += 64u) {
        var q_val = 0.0;
        var k_val = 0.0;
        var v_val = 0.0;
        for (var j = 0u; j < D_MODEL_C; j++) {
            let x_val = x_t[b * D_MODEL_C + j];
            q_val += x_val * get_f16_val(w_q[(i * D_MODEL_C + j) / 2u], i * D_MODEL_C + j);
            k_val += x_val * get_f16_val(w_k[(i * D_MODEL_C + j) / 2u], i * D_MODEL_C + j);
            v_val += x_val * get_f16_val(w_v[(i * D_MODEL_C + j) / 2u], i * D_MODEL_C + j);
        }
        q_shared[i] = q_val;
        
        let t_idx = params.t % MAX_SEQ_LEN_C;
        let cache_idx = b * (MAX_SEQ_LEN_C * D_MODEL_C) + t_idx * D_MODEL_C + i;
        k_cache[cache_idx] = k_val;
        v_cache[cache_idx] = v_val;
    }
    workgroupBarrier();
    
    // 2. Compute Attention Scores for each head
    let valid_history = select(params.t + 1u, MAX_SEQ_LEN_C, params.t >= MAX_SEQ_LEN_C);
    
    for (var h = 0u; h < HEADS_C; h++) {
        // Compute dots (q * K^T)
        for (var t_hist = tid; t_hist < valid_history; t_hist += 64u) {
            var score = 0.0;
            let cache_base = b * (MAX_SEQ_LEN_C * D_MODEL_C) + t_hist * D_MODEL_C + h * HEAD_DIM_C;
            for (var d = 0u; d < HEAD_DIM_C; d++) {
                score += q_shared[h * HEAD_DIM_C + d] * k_cache[cache_base + d];
            }
            // Scale
            score = score * 0.125; // 1 / sqrt(64)
            attn_scores[t_hist] = score;
        }
        workgroupBarrier();
        
        // Softmax per head (simplified max subtraction for stability)
        var max_score = -99999.0;
        for (var t_hist = 0u; t_hist < valid_history; t_hist++) {
            if attn_scores[t_hist] > max_score {
                max_score = attn_scores[t_hist];
            }
        }
        var sum_exp = 0.0;
        for (var t_hist = tid; t_hist < valid_history; t_hist += 64u) {
            let e = exp(attn_scores[t_hist] - max_score);
            attn_scores[t_hist] = e;
        }
        workgroupBarrier();
        
        // Tree reduce sum_exp ? For simplicity, single thread sums it up
        if tid == 0u {
            var s = 0.0;
            for (var t_hist = 0u; t_hist < valid_history; t_hist++) {
                s += attn_scores[t_hist];
            }
            for (var t_hist = 0u; t_hist < valid_history; t_hist++) {
                attn_scores[t_hist] /= s;
            }
        }
        workgroupBarrier();
        
        // Compute Out = Softmax * V
        for (var d = tid; d < HEAD_DIM_C; d += 64u) {
            var out_val = 0.0;
            for (var t_hist = 0u; t_hist < valid_history; t_hist++) {
                let cache_base = b * (MAX_SEQ_LEN_C * D_MODEL_C) + t_hist * D_MODEL_C + h * HEAD_DIM_C;
                out_val += attn_scores[t_hist] * v_cache[cache_base + d];
            }
            attn_out[h * HEAD_DIM_C + d] = out_val;
        }
        workgroupBarrier();
    }
    
    // 3. Final projection W_o
    for (var i = tid; i < D_MODEL_C; i += 64u) {
        var o_val = 0.0;
        for (var j = 0u; j < D_MODEL_C; j++) {
            o_val += attn_out[j] * get_f16_val(w_o[(i * D_MODEL_C + j) / 2u], i * D_MODEL_C + j);
        }
        // Residual connection: Attention + Original x_t
        x_attn[b * D_MODEL_C + i] = x_t[b * D_MODEL_C + i] + o_val;
    }
}
"#;

pub const ENCODER_SHADER: &str = r#"


@group(0) @binding(0) var<storage, read> bpe_embeddings: array<u32>;
@group(0) @binding(1) var<storage, read> w_fusion: array<u32>;
@group(0) @binding(2) var<storage, read> w_phrase: array<u32>;
@group(0) @binding(3) var<storage, read_write> x_t_out: array<f32>;
struct BatchInput {
    bpe_id_t: u32,
    t: u32,
    window_size: u32,
    byte_0: u32,
    byte_1: u32,
    byte_2: u32,
    bpe_0: u32,
    bpe_1: u32,
    bpe_2: u32,
    bpe_3: u32,
    bpe_4: u32,
    bpe_5: u32,
    bpe_6: u32,
    bpe_7: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<storage, read> batch_inputs: array<BatchInput>;


fn get_f16_val(packed: u32, i: u32) -> f32 {
    let vec = unpack2x16float(packed);
    if (i % 2u) == 1u { return vec.y; }
    return vec.x;
}

const BPE_VOCAB_SIZE_C: u32 = 4096u;

var<workgroup> concat: array<f32, 512>;

fn ngram_hash(bytes: array<u32, 3>, len: u32) -> u32 {
    var h: u32 = 2166136261u;
    if len > 0u {
        h = h ^ bytes[0];
        h = h * 16777619u;
    }
    if len > 1u {
        h = h ^ bytes[1];
        h = h * 16777619u;
    }
    if len > 2u {
        h = h ^ bytes[2];
        h = h * 16777619u;
    }
    return h;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let b = gid.y;


    for (var i = tid; i < 512u; i += 64u) {
        concat[i] = 0.0;
    }
    workgroupBarrier();

    if tid == 0u {
        let b_arr = array<u32, 3>(batch_inputs[b].byte_0, batch_inputs[b].byte_1, batch_inputs[b].byte_2);
        var e_bytes_local = array<f32, 128>();
        for (var i=0u; i<128u; i++) { e_bytes_local[i] = 0.0; }

        for (var n = 1u; n <= 3u; n++) {
            if batch_inputs[b].t + 1u >= n {
                var gram = array<u32, 3>(0u, 0u, 0u);
                if n == 1u { gram[0] = b_arr[0]; }
                else if n == 2u { gram[0] = b_arr[1]; gram[1] = b_arr[0]; }
                else if n == 3u { gram[0] = b_arr[2]; gram[1] = b_arr[1]; gram[2] = b_arr[0]; }
                
                let h = ngram_hash(gram, n);
                let bucket = h % 128u;
                e_bytes_local[bucket] += 1.0 / f32(n);
            }
        }
        
        var norm = 0.0;
        for (var i=0u; i<128u; i++) { norm += e_bytes_local[i] * e_bytes_local[i]; }
        norm = sqrt(norm);
        if norm > 1e-8 {
            for (var i=0u; i<128u; i++) { concat[i] = e_bytes_local[i] / norm; }
        } else {
            for (var i=0u; i<128u; i++) { concat[i] = e_bytes_local[i]; }
        }
    }
    
    if tid < 64u {
        let id = batch_inputs[b].bpe_id_t % BPE_VOCAB_SIZE_C;
        let base_concat = 128u;
        for (var k = 0u; k < 4u; k++) {
            let offset = tid * 4u + k;
            concat[base_concat + offset] = get_f16_val(bpe_embeddings[(id * 256u + offset) / 2u], id * 256u + offset);
        }
    }

    if tid < 64u {
        var bpes = array<u32, 8>(batch_inputs[b].bpe_0, batch_inputs[b].bpe_1, batch_inputs[b].bpe_2, batch_inputs[b].bpe_3, batch_inputs[b].bpe_4, batch_inputs[b].bpe_5, batch_inputs[b].bpe_6, batch_inputs[b].bpe_7);
        let win = batch_inputs[b].window_size;
        
        for (var p = 0u; p < 2u; p++) {
            let row = tid * 2u + p;
            var dot = 0.0;
            for (var k = 0u; k < win; k++) {
                let pos_idx = win - 1u - k;
                if batch_inputs[b].t >= pos_idx {
                    let bpe_id = bpes[pos_idx] % BPE_VOCAB_SIZE_C;
                    let emb_base = bpe_id * 256u;
                    let w_base = row * win * 256u + k * 256u;
                    for (var e = 0u; e < 256u; e++) {
                        dot += get_f16_val(w_phrase[(w_base + e) / 2u], w_base + e) * get_f16_val(bpe_embeddings[(emb_base + e) / 2u], emb_base + e);
                    }
                }
            }
            concat[384u + row] = dot;
        }
    }
    
    workgroupBarrier();

    for (var k = 0u; k < 8u; k++) {
        let row = tid * 8u + k;
        var dot = 0.0;
        let w_base = row * 512u;
        for (var c = 0u; c < 512u; c++) {
            dot += get_f16_val(w_fusion[(w_base + c) / 2u], w_base + c) * concat[c];
        }
        x_t_out[b * 512u + row] = dot;
    }
}
"#;

pub const RESERVOIR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       r_matrix : array<u32>;
@group(0) @binding(1) var<storage, read>       w_in     : array<u32>;
@group(0) @binding(2) var<storage, read>       s_prev   : array<f32>;
@group(0) @binding(3) var<storage, read>       x_t      : array<f32>;
@group(0) @binding(4) var<storage, read_write> s_out    : array<f32>;


fn get_f16_val(packed: u32, i: u32) -> f32 {
    let vec = unpack2x16float(packed);
    if (i % 2u) == 1u { return vec.y; }
    return vec.x;
}

const N_RES_C:   u32 = 4096u;
const D_MODEL_C: u32 = 512u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let b: u32 = gid.y;
    if i >= N_RES_C { return; }

    var acc_r: f32 = 0.0;
    let row_r: u32 = i * N_RES_C;
    for (var k = 0u; k < N_RES_C; k++) {
        acc_r += get_f16_val(r_matrix[(row_r + k) / 2u], row_r + k) * s_prev[b * N_RES_C + k];
    }

    var acc_w: f32 = 0.0;
    let row_w: u32 = i * D_MODEL_C;
    for (var m = 0u; m < D_MODEL_C; m++) {
        acc_w += get_f16_val(w_in[(row_w + m) / 2u], row_w + m) * x_t[b * D_MODEL_C + m];
    }
    s_out[b * N_RES_C + i] = tanh(acc_r + acc_w);
}
"#;

pub const PROJECTIONS_SHADER: &str = r#"
// Computes local_s_l = W_up_l * s_t and ro_l = M_l * local_s_l
@group(0) @binding(0) var<storage, read> s_t: array<f32>;
@group(0) @binding(1) var<storage, read> w_up_all: array<u32>;
@group(0) @binding(2) var<storage, read> m_all: array<u32>;
@group(0) @binding(3) var<storage, read_write> local_s_all: array<f32>;
@group(0) @binding(4) var<storage, read_write> ro_all: array<f32>;
@group(0) @binding(5) var<uniform> num_layers: u32;


fn get_f16_val(packed: u32, i: u32) -> f32 {
    let vec = unpack2x16float(packed);
    if (i % 2u) == 1u { return vec.y; }
    return vec.x;
}

const N_RES_C: u32 = 4096u;
const RANK_R_C: u32 = 32u;

var<workgroup> shared_local_s: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let l = wid.x;
    let b = gid.y;
    if l >= num_layers { return; }
    let i = lid;

    var dot_s = 0.0;
    let w_up_offset = (l * RANK_R_C + i) * N_RES_C;
    for (var k = 0u; k < N_RES_C; k++) {
        dot_s += get_f16_val(w_up_all[(w_up_offset + k) / 2u], w_up_offset + k) * s_t[b * N_RES_C + k];
    }
    shared_local_s[i] = dot_s;
    local_s_all[b * (num_layers * RANK_R_C) + l * RANK_R_C + i] = dot_s;

    workgroupBarrier();

    var dot_ro = 0.0;
    let m_offset = (l * RANK_R_C + i) * RANK_R_C;
    for (var j = 0u; j < RANK_R_C; j++) {
        dot_ro += get_f16_val(m_all[(m_offset + j) / 2u], m_offset + j) * shared_local_s[j];
    }
    ro_all[b * (num_layers * RANK_R_C) + l * RANK_R_C + i] = dot_ro;
}
"#;

pub const AGGREGATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> s_t: array<f32>;
@group(0) @binding(1) var<storage, read> ro_all: array<f32>;
@group(0) @binding(2) var<storage, read> w_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> y_hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev_prediction: array<f32>;
@group(0) @binding(5) var<uniform> num_layers: u32;


fn get_f16_val(packed: u32, i: u32) -> f32 {
    let vec = unpack2x16float(packed);
    if (i % 2u) == 1u { return vec.y; }
    return vec.x;
}

const N_RES_C: u32 = 4096u;
const RANK_R_C: u32 = 32u;
const D_MODEL_C: u32 = 512u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let b = gid.y;
    if j >= D_MODEL_C { return; }

    var dot = 0.0;
    let w_out_offset = j * N_RES_C;
    
    // Read modulation
    var mod_sums: array<f32, 32>;
    for (var i=0u; i<RANK_R_C; i++) { mod_sums[i] = 0.0; }
    for (var l=0u; l<num_layers; l++) {
        for (var i=0u; i<RANK_R_C; i++) {
            mod_sums[i] += ro_all[b * (num_layers * RANK_R_C) + l * RANK_R_C + i];
        }
    }

    for (var k = 0u; k < N_RES_C; k++) {
        var s_val = s_t[b * N_RES_C + k];
        if k < RANK_R_C {
            s_val += mod_sums[k];
        }
        dot += get_f16_val(w_out[(w_out_offset + k) / 2u], w_out_offset + k) * s_val;
    }
    y_hidden[b * D_MODEL_C + j] = dot;
    prev_prediction[b * D_MODEL_C + j] = dot;
}
"#;

pub const LOGIT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       output_embeddings : array<u32>;
@group(0) @binding(1) var<storage, read>       output_bias       : array<u32>;
@group(0) @binding(2) var<storage, read>       y_hidden          : array<f32>;
@group(0) @binding(3) var<storage, read_write> logits_out        : array<f32>;


fn get_f16_val(packed: u32, i: u32) -> f32 {
    let vec = unpack2x16float(packed);
    if (i % 2u) == 1u { return vec.y; }
    return vec.x;
}

const VOCAB_SIZE_C: u32 = 50000u;
const D_MODEL_C:    u32 = 512u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v: u32 = gid.x;
    let b: u32 = gid.y;
    if v >= VOCAB_SIZE_C { return; }

    var dot: f32 = 0.0;
    let row_base: u32 = v * D_MODEL_C;
    for (var k = 0u; k < D_MODEL_C; k++) {
        dot += get_f16_val(output_embeddings[(row_base + k) / 2u], row_base + k) * y_hidden[b * D_MODEL_C + k];
    }
    logits_out[b * VOCAB_SIZE_C + v] = dot + get_f16_val(output_bias[(v) / 2u], v);
}
"#;

pub const SAMPLING_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> logits_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> top_k_tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> top_k_probs: array<f32>;

const VOCAB_SIZE_C: u32 = 50000u;
const K_VAL_C: u32 = 50u; 

var<workgroup> shared_max_val: array<f32, 256>;
var<workgroup> shared_max_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let stride = 256u;
    let b = gid.y;
    
    for (var k_idx = 0u; k_idx < K_VAL_C; k_idx++) {
        var max_val = -999999.0;
        var max_idx = 0u;

        for (var i = lid; i < VOCAB_SIZE_C; i += stride) {
            let v = logits_out[b * VOCAB_SIZE_C + i];
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }

        shared_max_val[lid] = max_val;
        shared_max_idx[lid] = max_idx;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s >>= 1u) {
            if lid < s {
                if shared_max_val[lid + s] > shared_max_val[lid] {
                    shared_max_val[lid] = shared_max_val[lid + s];
                    shared_max_idx[lid] = shared_max_idx[lid + s];
                }
            }
            workgroupBarrier();
        }

        if lid == 0u {
            let best_idx = shared_max_idx[0];
            let best_val = shared_max_val[0];
            top_k_tokens[b * K_VAL_C + k_idx] = best_idx;
            top_k_probs[b * K_VAL_C + k_idx] = best_val;
            
            logits_out[b * VOCAB_SIZE_C + best_idx] = -999999.0;
        }
        
        storageBarrier();
        workgroupBarrier();
    }
}
"#;


// ─────────────────────────────────────────────────────────────────────────────
// GpuInferenceContext
// ─────────────────────────────────────────────────────────────────────────────

pub struct GpuInferenceContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,

    // Pipelines
    enc_pl: wgpu::ComputePipeline,
    pub attn_pl: wgpu::ComputePipeline,
    res_pl: wgpu::ComputePipeline,
    proj_pl: wgpu::ComputePipeline,
    agg_pl: wgpu::ComputePipeline,
    log_pl: wgpu::ComputePipeline,
    samp_pl: wgpu::ComputePipeline,

    // BGLs
    enc_bgl: wgpu::BindGroupLayout,
    pub attn_bgl: wgpu::BindGroupLayout,
    res_bgl: wgpu::BindGroupLayout,
    proj_bgl: wgpu::BindGroupLayout,
    agg_bgl: wgpu::BindGroupLayout,
    log_bgl: wgpu::BindGroupLayout,
    samp_bgl: wgpu::BindGroupLayout,

    // Buffers
    buf_bpe_embeddings: wgpu::Buffer,
    buf_w_fusion: wgpu::Buffer,
    buf_w_phrase: wgpu::Buffer,
    buf_encoder_params: wgpu::Buffer,
    pub phrase_window: usize,
    buf_r: wgpu::Buffer,
    buf_w_in: wgpu::Buffer,
    buf_w_q: wgpu::Buffer,
    buf_w_k: wgpu::Buffer,
    buf_w_v: wgpu::Buffer,
    buf_w_o: wgpu::Buffer,
    buf_k_cache: wgpu::Buffer,
    buf_v_cache: wgpu::Buffer,
    buf_x_attn: wgpu::Buffer,
    buf_attn_params: wgpu::Buffer,
    buf_s: [wgpu::Buffer; 2],
    pub s_ping: usize,

    buf_x_t: wgpu::Buffer,
    buf_y_hidden: wgpu::Buffer,
    buf_prev_pred: wgpu::Buffer,
    
    // Hebbian & Aggregation
    buf_w_up_all: wgpu::Buffer,
    buf_w_out: wgpu::Buffer,
    buf_m_all: [wgpu::Buffer; 2],
    pub m_ping: usize,
    buf_local_s_all: wgpu::Buffer,
    buf_ro_all: wgpu::Buffer,
    buf_num_layers: wgpu::Buffer,

    // Logits & Sampling
    buf_out_emb: wgpu::Buffer,
    buf_out_bias: wgpu::Buffer,
    buf_logits: wgpu::Buffer,
    buf_top_k_tokens: wgpu::Buffer,
    buf_top_k_probs: wgpu::Buffer,
    buf_top_k_tokens_readback: wgpu::Buffer,
    buf_top_k_probs_readback: wgpu::Buffer,
    pub top_k_size: usize,
    pub max_batch_size: usize,
    pub buf_encoder_batch_inputs: Vec<wgpu::Buffer>,
    
    num_layers: usize,
}


fn pack_f32_to_f16(data: &[f32]) -> Vec<u32> {
    let mut packed = Vec::with_capacity((data.len() + 1) / 2);
    for chunk in data.chunks(2) {
        let h0 = half::f16::from_f32(chunk[0]).to_bits() as u32;
        let h1 = if chunk.len() > 1 { half::f16::from_f32(chunk[1]).to_bits() as u32 } else { 0 };
        packed.push(h0 | (h1 << 16));
    }
    packed
}

impl GpuInferenceContext {
    pub fn new(
        num_layers: usize,
        r_matrix_data: &[f32],
        w_in_data: &[f32],
        w_q_data: &[f32],
        w_k_data: &[f32],
        w_v_data: &[f32],
        w_o_data: &[f32],
        out_emb_data: &[f32],
        out_bias_data: &[f32],
        w_up_all_data: &[f32],
        w_out_data: &[f32],
        m_base_all_data: &[f32],
        bpe_embeddings_data: &[f32],
        w_fusion_data: &[f32],
        w_phrase_data: &[f32],
        phrase_window: usize,
    ) -> Self {
        let max_batch_size = 32;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("No adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_buffer_size: 256 * 1024 * 1024,
                    max_storage_buffer_binding_size: 256 * 1024 * 1024,
                    ..Default::default()
                },
                ..Default::default()
            }, None
        )).unwrap();

        let create_shader = |src: &str, label: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        
        let attn_mod = create_shader(ATTENTION_SHADER, "attn_mod");
        let enc_mod = create_shader(ENCODER_SHADER, "enc_mod");
        let res_mod = create_shader(RESERVOIR_SHADER, "res_mod");
        let proj_mod = create_shader(PROJECTIONS_SHADER, "proj_mod");
        let agg_mod = create_shader(AGGREGATE_SHADER, "agg_mod");
        let log_mod = create_shader(LOGIT_SHADER, "log_mod");
        let samp_mod = create_shader(SAMPLING_SHADER, "samp_mod");

        let ro = |b| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let rw = |b| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let uni = |b| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };

        let attn_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), ro(3), ro(4), rw(5), rw(6), rw(7), uni(8)] });
        let enc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3), ro(4)] });
        let res_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), ro(3), rw(4)] });
        let proj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3), rw(4), uni(5)] });
        let agg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3), rw(4), uni(5)] });
        let log_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3)] });
        let samp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[rw(0), rw(1), rw(2)] });

        let make_pl = |m, bgl| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&layout), module: m, entry_point: "main", compilation_options: Default::default()  })
        };

        let attn_pl = make_pl(&attn_mod, &attn_bgl);
        let enc_pl = make_pl(&enc_mod, &enc_bgl);
        let res_pl = make_pl(&res_mod, &res_bgl);
        let proj_pl = make_pl(&proj_mod, &proj_bgl);
        let agg_pl = make_pl(&agg_mod, &agg_bgl);
        let log_pl = make_pl(&log_mod, &log_bgl);
        let samp_pl = make_pl(&samp_mod, &samp_bgl);

        let upload = |data: &[f32], lbl: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(lbl), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC });
        let upload_u32 = |data: &[u32], lbl: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(lbl), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC });
        let empty = |n: usize, lbl: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(lbl), size: (n*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let readback = |n: usize, lbl: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(lbl), size: (n*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let buf_bpe_embeddings = upload_u32(&pack_f32_to_f16(bpe_embeddings_data), "bpe_emb");
        let buf_w_fusion = upload_u32(&pack_f32_to_f16(w_fusion_data), "w_fus");
        let buf_w_phrase = upload_u32(&pack_f32_to_f16(w_phrase_data), "w_phr");
        let buf_encoder_params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("enc_p"), size: (max_batch_size * 64) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let buf_r = upload_u32(&pack_f32_to_f16(r_matrix_data), "r");
        let buf_w_in = upload_u32(&pack_f32_to_f16(w_in_data), "w_in");
        let buf_w_q = upload_u32(&pack_f32_to_f16(w_q_data), "w_q");
        let buf_w_k = upload_u32(&pack_f32_to_f16(w_k_data), "w_k");
        let buf_w_v = upload_u32(&pack_f32_to_f16(w_v_data), "w_v");
        let buf_w_o = upload_u32(&pack_f32_to_f16(w_o_data), "w_o");
        let buf_k_cache = empty(max_batch_size * 1024 * D_MODEL, "k_cache");
        let buf_v_cache = empty(max_batch_size * 1024 * D_MODEL, "v_cache");
        let buf_x_attn = empty(max_batch_size * D_MODEL, "x_attn");
        let buf_attn_params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("attn_params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    
        let buf_s = [empty(max_batch_size * N_RES, "s0"), empty(max_batch_size * N_RES, "s1")];
        let buf_x_t = empty(max_batch_size * D_MODEL, "x_t");
        let buf_y_hidden = empty(max_batch_size * D_MODEL, "y");
        let buf_prev_pred = empty(max_batch_size * D_MODEL, "prev");
        
        let buf_w_up_all = upload_u32(&pack_f32_to_f16(w_up_all_data), "w_up_all");
        let buf_w_out = upload_u32(&pack_f32_to_f16(w_out_data), "w_out");
        
        let packed_m = pack_f32_to_f16(m_base_all_data);
        let buf_m_all = [upload_u32(&packed_m, "m0"), upload_u32(&packed_m, "m1")];
        
        let buf_local_s_all = empty(max_batch_size * num_layers * RANK_R, "loc_s");
        let buf_ro_all = empty(max_batch_size * num_layers * RANK_R, "ro");
        
        let buf_num_layers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("num_layers"),
            contents: bytemuck::cast_slice(&[num_layers as u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let buf_out_emb = upload_u32(&pack_f32_to_f16(out_emb_data), "emb");
        let buf_out_bias = upload_u32(&pack_f32_to_f16(out_bias_data), "bias");
        let buf_logits = empty(max_batch_size * VOCAB_SIZE, "log");
        let buf_top_k_tokens = empty(max_batch_size * 50, "token_k");
        let buf_top_k_probs = empty(max_batch_size * 50, "prob_k");
        let buf_top_k_tokens_readback = readback(max_batch_size * 50, "token_k_rb");
        let buf_top_k_probs_readback = readback(max_batch_size * 50, "prob_k_rb");
        let top_k_size = 50;
        let buf_encoder_batch_inputs = vec![empty(max_batch_size * 1024, "enc_b")];

        GpuInferenceContext {
            device, queue, enc_pl, attn_pl, res_pl, proj_pl, agg_pl, log_pl, samp_pl,
            enc_bgl, attn_bgl, res_bgl, proj_bgl, agg_bgl, log_bgl, samp_bgl,
            buf_bpe_embeddings, buf_w_fusion, buf_w_phrase, buf_encoder_params, phrase_window, buf_encoder_batch_inputs, 
            buf_r, buf_w_in, buf_w_q, buf_w_k, buf_w_v, buf_w_o, buf_k_cache, buf_v_cache, buf_x_attn, buf_attn_params, buf_s, s_ping: 0, buf_x_t, buf_y_hidden, buf_prev_pred,
            buf_w_up_all, buf_w_out, buf_m_all, m_ping: 0, buf_local_s_all, buf_ro_all, buf_num_layers,
            max_batch_size,
            buf_out_emb, buf_out_bias, buf_logits, buf_top_k_tokens, buf_top_k_probs, buf_top_k_tokens_readback, buf_top_k_probs_readback, top_k_size,
            num_layers,

        }
    }

    /// Single call to perform the entire forward pass and return 1 token ID.
    pub fn forward_inference(
        &mut self,
        bytes_batch: &[Vec<u8>],
        t_batch: &[usize],
        bpe_ids_batch: &[Vec<u32>],
    ) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
        let b = bytes_batch.len();
        assert!(b <= self.max_batch_size);

        let mut params = Vec::with_capacity(b * 16);
        for i in 0..b {
            let bytes = &bytes_batch[i];
            let t = t_batch[i];
            let bpe_ids = &bpe_ids_batch[i];
            
            let bpe_id_t = bpe_ids[t];
            let mut byte_0 = 0u32; let mut byte_1 = 0u32; let mut byte_2 = 0u32;
            if t < bytes.len() { byte_0 = bytes[t] as u32; }
            if t >= 1 { byte_1 = bytes[t-1] as u32; }
            if t >= 2 { byte_2 = bytes[t-2] as u32; }

            let mut bpes = [0u32; 8];
            for k in 0..8 {
                if t >= k { bpes[k] = bpe_ids[t-k]; }
            }

            params.extend_from_slice(&[
                bpe_id_t, t as u32, self.phrase_window as u32,
                byte_0, byte_1, byte_2,
                bpes[0], bpes[1], bpes[2], bpes[3],
                bpes[4], bpes[5], bpes[6], bpes[7],
                0, 0
            ]);
        }
        self.queue.write_buffer(&self.buf_encoder_params, 0, bytemuck::cast_slice(&params));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 0. Encoder
        let bg_enc = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.attn_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_bpe_embeddings.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_w_fusion.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_w_phrase.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_x_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_encoder_batch_inputs[b].as_entire_binding() },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.enc_pl); cp.set_bind_group(0, &bg_enc, &[]);
            cp.dispatch_workgroups(1, b as u32, 1); // are enough to process x_t
        }

        // 1. Reservoir
        let prev_s = self.s_ping;
        let next_s = 1 - prev_s;
        
        let params_attn = [t_batch[0] as u32, 0, 0, 0];
        self.queue.write_buffer(&self.buf_attn_params, 0, bytemuck::cast_slice(&params_attn));

        let bg_attn = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.attn_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_x_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_w_q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_w_k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_w_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_w_o.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.buf_x_attn.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.buf_attn_params.as_entire_binding() },
            ],
        });

        let bg_res = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.res_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_w_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_s[prev_s].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_x_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_s[next_s].as_entire_binding() },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            
        cp.set_pipeline(&self.attn_pl);
        cp.set_bind_group(0, &bg_attn, &[]);
        cp.dispatch_workgroups(1, b as u32, 1);

        cp.set_pipeline(&self.res_pl); cp.set_bind_group(0, &bg_res, &[]);
            cp.dispatch_workgroups((N_RES as u32 + 63) / 64, b as u32, 1);
        }
        self.s_ping = next_s;

        // 2. Projections
        let bg_proj = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.proj_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_s[self.s_ping].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_w_up_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_m_all[self.m_ping].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_local_s_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_ro_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_num_layers.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.proj_pl); cp.set_bind_group(0, &bg_proj, &[]);
            cp.dispatch_workgroups(self.num_layers as u32, b as u32, 1);
        }

        // 3. Aggregate
        let bg_agg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.agg_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_s[self.s_ping].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_ro_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_w_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_y_hidden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_prev_pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_num_layers.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.agg_pl); cp.set_bind_group(0, &bg_agg, &[]);
            cp.dispatch_workgroups((D_MODEL as u32 + 63) / 64, b as u32, 1);
        }

        // 4. Logits
        let bg_log = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.log_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_out_emb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_out_bias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_y_hidden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_logits.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.log_pl); cp.set_bind_group(0, &bg_log, &[]);
            cp.dispatch_workgroups((VOCAB_SIZE as u32 + 255) / 256, b as u32, 1);
        }

        // 5. Sampling
        let bg_samp = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.samp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_top_k_tokens.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_top_k_probs.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.samp_pl); cp.set_bind_group(0, &bg_samp, &[]);
            cp.dispatch_workgroups(1, b as u32, 1);
        }

        enc.copy_buffer_to_buffer(&self.buf_top_k_tokens, 0, &self.buf_top_k_tokens_readback, 0, (self.top_k_size * 4) as u64);
        enc.copy_buffer_to_buffer(&self.buf_top_k_probs, 0, &self.buf_top_k_probs_readback, 0, (self.top_k_size * 4) as u64);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice_tok = self.buf_top_k_tokens_readback.slice(..);
        let slice_prob = self.buf_top_k_probs_readback.slice(..);
        
        let (tx, rx) = std::sync::mpsc::channel();
        slice_tok.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        
        // We map prob async as well, but we can poll for both
        let (tx2, rx2) = std::sync::mpsc::channel();
        slice_prob.map_async(wgpu::MapMode::Read, move |r| { tx2.send(r).unwrap(); });
        
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        rx2.recv().unwrap().unwrap();
        
        let mapped_tok = slice_tok.get_mapped_range();
        let flat_tokens = bytemuck::cast_slice::<u8, u32>(&mapped_tok).to_vec();
        drop(mapped_tok);
        self.buf_top_k_tokens_readback.unmap();
        
        let mapped_prob = slice_prob.get_mapped_range();
        let flat_probs = bytemuck::cast_slice::<u8, f32>(&mapped_prob).to_vec();
        drop(mapped_prob);
        self.buf_top_k_probs_readback.unmap();
        
        let mut out_tokens = Vec::with_capacity(b);
        let mut out_probs = Vec::with_capacity(b);
        for i in 0..b {
            let start = i * self.top_k_size;
            let end = start + self.top_k_size;
            out_tokens.push(flat_tokens[start..end].to_vec());
            out_probs.push(flat_probs[start..end].to_vec());
        }
        
        (out_tokens, out_probs)
    
    }
}
