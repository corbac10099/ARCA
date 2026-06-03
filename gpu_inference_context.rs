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

pub const RESERVOIR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       r_matrix : array<f32>;
@group(0) @binding(1) var<storage, read>       w_in     : array<f32>;
@group(0) @binding(2) var<storage, read>       s_prev   : array<f32>;
@group(0) @binding(3) var<storage, read>       x_t      : array<f32>;
@group(0) @binding(4) var<storage, read_write> s_out    : array<f32>;

const N_RES_C:   u32 = 4096u;
const D_MODEL_C: u32 = 512u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if i >= N_RES_C { return; }

    var acc_r: f32 = 0.0;
    let row_r: u32 = i * N_RES_C;
    for (var k = 0u; k < N_RES_C; k++) {
        acc_r += r_matrix[row_r + k] * s_prev[k];
    }

    var acc_w: f32 = 0.0;
    let row_w: u32 = i * D_MODEL_C;
    for (var m = 0u; m < D_MODEL_C; m++) {
        acc_w += w_in[row_w + m] * x_t[m];
    }
    s_out[i] = tanh(acc_r + acc_w);
}
"#;

pub const PROJECTIONS_SHADER: &str = r#"
// Computes local_s_l = W_up_l * s_t and ro_l = M_l * local_s_l
@group(0) @binding(0) var<storage, read> s_t: array<f32>;
@group(0) @binding(1) var<storage, read> w_up_all: array<f32>;
@group(0) @binding(2) var<storage, read> m_all: array<f32>;
@group(0) @binding(3) var<storage, read_write> local_s_all: array<f32>;
@group(0) @binding(4) var<storage, read_write> ro_all: array<f32>;
@group(0) @binding(5) var<uniform> num_layers: u32;

const N_RES_C: u32 = 4096u;
const RANK_R_C: u32 = 32u;

var<workgroup> shared_local_s: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let l = wid.x;
    if l >= num_layers { return; }
    let i = lid;

    var dot_s = 0.0;
    let w_up_offset = (l * RANK_R_C + i) * N_RES_C;
    for (var k = 0u; k < N_RES_C; k++) {
        dot_s += w_up_all[w_up_offset + k] * s_t[k];
    }
    shared_local_s[i] = dot_s;
    local_s_all[l * RANK_R_C + i] = dot_s;

    workgroupBarrier();

    var dot_ro = 0.0;
    let m_offset = (l * RANK_R_C + i) * RANK_R_C;
    for (var j = 0u; j < RANK_R_C; j++) {
        dot_ro += m_all[m_offset + j] * shared_local_s[j];
    }
    ro_all[l * RANK_R_C + i] = dot_ro;
}
"#;

pub const AGGREGATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> s_t: array<f32>;
@group(0) @binding(1) var<storage, read> ro_all: array<f32>;
@group(0) @binding(2) var<storage, read> w_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev_prediction: array<f32>;
@group(0) @binding(5) var<uniform> num_layers: u32;

const N_RES_C: u32 = 4096u;
const RANK_R_C: u32 = 32u;
const D_MODEL_C: u32 = 512u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= D_MODEL_C { return; }

    var dot = 0.0;
    let w_out_offset = j * N_RES_C;
    
    // Read modulation
    var mod_sums: array<f32, 32>;
    for (var i=0u; i<RANK_R_C; i++) { mod_sums[i] = 0.0; }
    for (var l=0u; l<num_layers; l++) {
        for (var i=0u; i<RANK_R_C; i++) {
            mod_sums[i] += ro_all[l * RANK_R_C + i];
        }
    }

    for (var k = 0u; k < N_RES_C; k++) {
        var s_val = s_t[k];
        if k < RANK_R_C {
            s_val += mod_sums[k];
        }
        dot += w_out[w_out_offset + k] * s_val;
    }
    y_hidden[j] = dot;
    prev_prediction[j] = dot;
}
"#;

pub const LOGIT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       output_embeddings : array<f32>;
@group(0) @binding(1) var<storage, read>       output_bias       : array<f32>;
@group(0) @binding(2) var<storage, read>       y_hidden          : array<f32>;
@group(0) @binding(3) var<storage, read_write> logits_out        : array<f32>;

const VOCAB_SIZE_C: u32 = 50000u;
const D_MODEL_C:    u32 = 512u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v: u32 = gid.x;
    if v >= VOCAB_SIZE_C { return; }

    var dot: f32 = 0.0;
    let row_base: u32 = v * D_MODEL_C;
    for (var k = 0u; k < D_MODEL_C; k++) {
        dot += output_embeddings[row_base + k] * y_hidden[k];
    }
    logits_out[v] = dot + output_bias[v];
}
"#;

pub const SAMPLING_SHADER: &str = r#"
// Argmax reduction shader to find the max logit (greedy sampling)
// Returns just the token ID to minimize PCIe traffic
@group(0) @binding(0) var<storage, read> logits_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> chosen_token: array<u32>;

const VOCAB_SIZE_C: u32 = 50000u;

var<workgroup> shared_max_val: array<f32, 256>;
var<workgroup> shared_max_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let stride = 256u;
    var max_val = -999999.0;
    var max_idx = 0u;

    // Each thread finds max over its strided elements
    for (var i = lid; i < VOCAB_SIZE_C; i += stride) {
        if logits_out[i] > max_val {
            max_val = logits_out[i];
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
        chosen_token[0] = shared_max_idx[0];
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
    res_pl: wgpu::ComputePipeline,
    proj_pl: wgpu::ComputePipeline,
    agg_pl: wgpu::ComputePipeline,
    log_pl: wgpu::ComputePipeline,
    samp_pl: wgpu::ComputePipeline,

    // BGLs
    res_bgl: wgpu::BindGroupLayout,
    proj_bgl: wgpu::BindGroupLayout,
    agg_bgl: wgpu::BindGroupLayout,
    log_bgl: wgpu::BindGroupLayout,
    samp_bgl: wgpu::BindGroupLayout,

    // Buffers
    buf_r: wgpu::Buffer,
    buf_w_in: wgpu::Buffer,
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
    buf_chosen_token: wgpu::Buffer,
    buf_chosen_readback: wgpu::Buffer,
    
    num_layers: usize,
}

impl GpuInferenceContext {
    pub fn new(
        num_layers: usize,
        r_matrix_data: &[f32],
        w_in_data: &[f32],
        out_emb_data: &[f32],
        out_bias_data: &[f32],
        w_up_all_data: &[f32],
        w_out_data: &[f32],
        m_base_all_data: &[f32], // using m_base as initial M
    ) -> Self {
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

        let res_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), ro(3), rw(4)] });
        let proj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3), rw(4), uni(5)] });
        let agg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3), rw(4), uni(5)] });
        let log_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3)] });
        let samp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), rw(1)] });

        let make_pl = |m, bgl| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&layout), module: m, entry_point: "main", compilation_options: Default::default(), cache: None })
        };

        let res_pl = make_pl(&res_mod, &res_bgl);
        let proj_pl = make_pl(&proj_mod, &proj_bgl);
        let agg_pl = make_pl(&agg_mod, &agg_bgl);
        let log_pl = make_pl(&log_mod, &log_bgl);
        let samp_pl = make_pl(&samp_mod, &samp_bgl);

        let upload = |data: &[f32], lbl: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(lbl), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC });
        let empty = |n: usize, lbl: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(lbl), size: (n*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let readback = |n: usize, lbl: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(lbl), size: (n*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let buf_r = upload(r_matrix_data, "r");
        let buf_w_in = upload(w_in_data, "w_in");
        let buf_s = [empty(N_RES, "s0"), empty(N_RES, "s1")];
        let buf_x_t = empty(D_MODEL, "x_t");
        let buf_y_hidden = empty(D_MODEL, "y");
        let buf_prev_pred = empty(D_MODEL, "prev");
        
        let buf_w_up_all = upload(w_up_all_data, "w_up_all");
        let buf_w_out = upload(w_out_data, "w_out");
        
        let buf_m_all = [upload(m_base_all_data, "m0"), upload(m_base_all_data, "m1")];
        
        let buf_local_s_all = empty(num_layers * RANK_R, "loc_s");
        let buf_ro_all = empty(num_layers * RANK_R, "ro");
        
        let buf_num_layers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("num_layers"),
            contents: bytemuck::cast_slice(&[num_layers as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_out_emb = upload(out_emb_data, "emb");
        let buf_out_bias = upload(out_bias_data, "bias");
        let buf_logits = empty(VOCAB_SIZE, "log");
        let buf_chosen_token = empty(1, "token");
        let buf_chosen_readback = readback(1, "token_rb");

        GpuInferenceContext {
            device, queue, res_pl, proj_pl, agg_pl, log_pl, samp_pl,
            res_bgl, proj_bgl, agg_bgl, log_bgl, samp_bgl,
            buf_r, buf_w_in, buf_s, s_ping: 0, buf_x_t, buf_y_hidden, buf_prev_pred,
            buf_w_up_all, buf_w_out, buf_m_all, m_ping: 0, buf_local_s_all, buf_ro_all, buf_num_layers,
            buf_out_emb, buf_out_bias, buf_logits, buf_chosen_token, buf_chosen_readback,
            num_layers,
        }
    }

    /// Single call to perform the entire forward pass and return 1 token ID.
    pub fn forward_inference(&mut self, x_t_data: &[f32]) -> u32 {
        self.queue.write_buffer(&self.buf_x_t, 0, bytemuck::cast_slice(x_t_data));
        
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 1. Reservoir
        let prev_s = self.s_ping;
        let next_s = 1 - prev_s;
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
            cp.set_pipeline(&self.res_pl); cp.set_bind_group(0, &bg_res, &[]);
            cp.dispatch_workgroups((N_RES as u32 + 63) / 64, 1, 1);
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
            cp.dispatch_workgroups(self.num_layers as u32, 1, 1);
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
            cp.dispatch_workgroups((D_MODEL as u32 + 63) / 64, 1, 1);
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
            cp.dispatch_workgroups((VOCAB_SIZE as u32 + 255) / 256, 1, 1);
        }

        // 5. Sampling
        let bg_samp = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.samp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_chosen_token.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.samp_pl); cp.set_bind_group(0, &bg_samp, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }

        enc.copy_buffer_to_buffer(&self.buf_chosen_token, 0, &self.buf_chosen_readback, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        // Readback 4 bytes
        let slice = self.buf_chosen_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
        drop(mapped);
        self.buf_chosen_readback.unmap();
        
        result
    }
}
