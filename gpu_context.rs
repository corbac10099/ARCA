/// GPU Context — wgpu device/queue/pipeline initialisation.
///
/// Refactored to eliminate CPU-GPU syncs during the training hot loop while
/// keeping explicit host orchestration, clean swap buffers, and a single
/// stable readback point.

use wgpu::util::DeviceExt;

pub const N_RES:      usize = 4096;
pub const D_MODEL:    usize = 512;
pub const RANK_R:     usize = 32;
pub const VOCAB_SIZE: usize = 50_000;
pub const TOP_K:      usize = 200;

pub const ALIGN: usize = 64;

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
@group(0) @binding(0) var<storage, read> s_t: array<f32>;
@group(0) @binding(1) var<storage, read> e_t: array<f32>;
@group(0) @binding(2) var<storage, read> w_up_all: array<f32>;
@group(0) @binding(3) var<storage, read> w_down_all: array<f32>;
@group(0) @binding(4) var<storage, read_write> local_s_all: array<f32>;
@group(0) @binding(5) var<storage, read_write> local_e_all: array<f32>;
@group(0) @binding(6) var<uniform> num_layers: u32;

const N_RES_C: u32 = 4096u;
const D_MODEL_C: u32 = 512u;
const RANK_R_C: u32 = 32u;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let l = wid.x;
    if l >= num_layers { return; }
    let i = lid;

    // local_s
    var dot_s = 0.0;
    let w_up_offset = (l * RANK_R_C + i) * N_RES_C;
    for (var k = 0u; k < N_RES_C; k++) {
        dot_s += w_up_all[w_up_offset + k] * s_t[k];
    }
    local_s_all[l * RANK_R_C + i] = dot_s;

    // local_e
    var dot_e = 0.0;
    let w_down_offset = (l * RANK_R_C + i) * D_MODEL_C;
    for (var k = 0u; k < D_MODEL_C; k++) {
        dot_e += w_down_all[w_down_offset + k] * e_t[k];
    }
    let depth_scale = 1.0 / (1.0 + f32(l) * 0.1);
    local_e_all[l * RANK_R_C + i] = dot_e * depth_scale;
}
"#;

pub const HEBBIAN_SHADER: &str = r#"
struct HebbianParams {
    lambda       : f32,
    kappa        : f32,
    beta3_eta    : f32,
    sigma_global : f32,
    alpha_fatigue: f32,
    tau_sat      : f32,
    _pad0        : f32,
    _pad1        : f32,
}

@group(0) @binding(0) var<storage, read>       m_prev  : array<f32>;
@group(0) @binding(1) var<storage, read>       local_e_all : array<f32>;
@group(0) @binding(2) var<storage, read>       local_s_all : array<f32>;
@group(0) @binding(3) var<storage, read>       m_base  : array<f32>;
@group(0) @binding(4) var<storage, read_write> m_out   : array<f32>;
@group(0) @binding(5) var<storage, read_write> ro_all  : array<f32>;
@group(0) @binding(6) var<uniform>             params  : HebbianParams;
@group(0) @binding(7) var<uniform>             layer_idx: u32;

const RANK_R_C : u32 = 32u;
const WG_SIZE_C: u32 = 64u;

var<workgroup> shared_frob: array<f32, 64>;

@compute @workgroup_size(32, 2, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
    let i: u32 = gid.x;
    let j: u32 = gid.y;
    let l_offset = layer_idx * RANK_R_C;

    var m_next_val: f32 = 0.0;
    if i < RANK_R_C && j < RANK_R_C {
        let idx: u32   = j * RANK_R_C + i;
        let delta: f32 = local_e_all[l_offset + j] * local_s_all[l_offset + i];
        m_next_val = params.lambda * m_prev[idx] + params.kappa * params.beta3_eta * delta;
    }

    shared_frob[lid] = m_next_val * m_next_val;
    workgroupBarrier();

    for (var stride: u32 = WG_SIZE_C >> 1u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_frob[lid] += shared_frob[lid + stride];
        }
        workgroupBarrier();
    }
    let frob_sq: f32 = shared_frob[0];

    var final_val = 0.0;
    if i < RANK_R_C && j < RANK_R_C {
        let idx: u32 = j * RANK_R_C + i;
        let fatigue: f32     = params.alpha_fatigue * tanh(frob_sq / params.tau_sat);
        let sigma_local: f32 = clamp(params.sigma_global + fatigue, 0.0, 0.95);
        let blended: f32 = (1.0 - sigma_local) * m_next_val + sigma_local * m_base[idx];
        final_val = tanh(blended) * 1.5;
        m_out[idx] = final_val;
    }

    workgroupBarrier();

    // Compute Readout: ro_all[l_offset + j] = sum_i m_out[j, i] * local_s[i]
    if lid < RANK_R_C {
        var ro_dot = 0.0;
        let row_offset = lid * RANK_R_C;
        for (var k=0u; k<RANK_R_C; k++) {
            ro_dot += m_out[row_offset + k] * local_s_all[l_offset + k];
        }
        ro_all[l_offset + lid] = ro_dot;
    }
}
"#;

pub const AGGREGATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> s_t: array<f32>;
@group(0) @binding(1) var<storage, read> ro_all: array<f32>;
@group(0) @binding(2) var<storage, read> w_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_hidden: array<f32>;
@group(0) @binding(4) var<uniform> num_layers: u32;

const N_RES_C: u32 = 4096u;
const RANK_R_C: u32 = 32u;
const D_MODEL_C: u32 = 512u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= D_MODEL_C { return; }

    var dot = 0.0;
    let w_out_offset = j * N_RES_C;
    
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

// ─────────────────────────────────────────────────────────────────────────────
// GpuContext
// ─────────────────────────────────────────────────────────────────────────────

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,

    // Pipelines
    res_pl: wgpu::ComputePipeline,
    proj_pl: wgpu::ComputePipeline,
    heb_pl: wgpu::ComputePipeline,
    agg_pl: wgpu::ComputePipeline,
    log_pl: wgpu::ComputePipeline,

    // BGLs
    res_bgl: wgpu::BindGroupLayout,
    proj_bgl: wgpu::BindGroupLayout,
    heb_bgl: wgpu::BindGroupLayout,
    agg_bgl: wgpu::BindGroupLayout,
    log_bgl: wgpu::BindGroupLayout,

    // Buffers
    buf_r_matrix: wgpu::Buffer,
    buf_w_in:     wgpu::Buffer,
    buf_s:        [wgpu::Buffer; 2],
    pub s_ping:   usize,
    
    buf_x_t:      wgpu::Buffer,
    buf_e_t:      wgpu::Buffer,
    buf_y_hidden: wgpu::Buffer,

    buf_w_up_all: wgpu::Buffer,
    buf_w_down_all: wgpu::Buffer,
    buf_w_out: wgpu::Buffer,
    
    buf_local_s_all: wgpu::Buffer,
    buf_local_e_all: wgpu::Buffer,
    buf_ro_all: wgpu::Buffer,

    pub buf_m:             Vec<[wgpu::Buffer; 2]>,
    pub m_ping:            Vec<usize>,
    buf_m_base:            Vec<wgpu::Buffer>,
    buf_hebbian_params:    Vec<wgpu::Buffer>,
    buf_layer_idx:         Vec<wgpu::Buffer>,
    buf_num_layers:        wgpu::Buffer,

    buf_output_embeddings: wgpu::Buffer,
    buf_output_bias:       wgpu::Buffer,
    buf_logits:            wgpu::Buffer,

    // Readback
    buf_s_readback:        wgpu::Buffer,
    buf_logits_readback:   wgpu::Buffer,
    buf_m_readback:        Vec<wgpu::Buffer>,
    
    pub num_layers: usize,
}

impl GpuContext {
    pub fn new(
        num_layers:     usize,
        r_matrix_data:  &[f32],
        w_in_data:      &[f32],
        out_emb_data:   &[f32],
        out_bias_data:  &[f32],
        m_base_data:    &[Vec<f32>],
        w_up_all_data:  &[f32],
        w_down_all_data: &[f32],
        w_out_data:     &[f32],
    ) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_buffer_size: 256 * 1024 * 1024,
                max_storage_buffer_binding_size: 256 * 1024 * 1024,
                ..Default::default()
            }, ..Default::default()
        }, None)).unwrap();

        let create_shader = |src: &str, label: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        
        let res_mod = create_shader(RESERVOIR_SHADER, "res_mod");
        let proj_mod = create_shader(PROJECTIONS_SHADER, "proj_mod");
        let heb_mod = create_shader(HEBBIAN_SHADER, "heb_mod");
        let agg_mod = create_shader(AGGREGATE_SHADER, "agg_mod");
        let log_mod = create_shader(LOGIT_SHADER, "log_mod");

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
        let proj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), ro(3), rw(4), rw(5), uni(6)] });
        let heb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), ro(3), rw(4), rw(5), uni(6), uni(7)] });
        let agg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3), uni(4)] });
        let log_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ro(0), ro(1), ro(2), rw(3)] });

        let make_pl = |m, bgl| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&layout), module: m, entry_point: "main", compilation_options: Default::default(), cache: None })
        };

        let res_pl = make_pl(&res_mod, &res_bgl);
        let proj_pl = make_pl(&proj_mod, &proj_bgl);
        let heb_pl = make_pl(&heb_mod, &heb_bgl);
        let agg_pl = make_pl(&agg_mod, &agg_bgl);
        let log_pl = make_pl(&log_mod, &log_bgl);

        let upload = |data: &[f32], lbl: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(lbl), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC });
        let empty = |n: usize, lbl: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(lbl), size: (n*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let readback = |n: usize, lbl: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(lbl), size: (n*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let buf_r_matrix = upload(r_matrix_data, "r");
        let buf_w_in = upload(w_in_data, "w_in");
        let buf_s = [empty(N_RES, "s0"), empty(N_RES, "s1")];
        
        let buf_x_t = empty(D_MODEL, "x_t");
        let buf_e_t = empty(D_MODEL, "e_t");
        let buf_y_hidden = empty(D_MODEL, "y");

        let buf_w_up_all = upload(w_up_all_data, "w_up_all");
        let buf_w_down_all = upload(w_down_all_data, "w_down_all");
        let buf_w_out = upload(w_out_data, "w_out");
        
        let buf_local_s_all = empty(num_layers * RANK_R, "loc_s");
        let buf_local_e_all = empty(num_layers * RANK_R, "loc_e");
        let buf_ro_all = empty(num_layers * RANK_R, "ro");

        let buf_num_layers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("num_layers"),
            contents: bytemuck::cast_slice(&[num_layers as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut buf_m = vec![];
        let mut m_ping = vec![];
        let mut buf_m_base = vec![];
        let mut buf_hebbian_params = vec![];
        let mut buf_layer_idx = vec![];
        let mut buf_m_readback = vec![];

        for l in 0..num_layers {
            buf_m.push([upload(&m_base_data[l], "m0"), upload(&m_base_data[l], "m1")]);
            m_ping.push(0);
            buf_m_base.push(upload(&m_base_data[l], "mbase"));
            
            let p_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hebbian_params"), size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            buf_hebbian_params.push(p_buf);
            
            let l_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layer_idx"), contents: bytemuck::cast_slice(&[l as u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            buf_layer_idx.push(l_buf);
            
            buf_m_readback.push(readback(RANK_R*RANK_R, "m_rb"));
        }

        let buf_output_embeddings = upload(out_emb_data, "emb");
        let buf_output_bias = upload(out_bias_data, "bias");
        let buf_logits = empty(VOCAB_SIZE, "log");

        let buf_s_readback = readback(N_RES, "s_rb");
        let buf_logits_readback = readback(VOCAB_SIZE, "log_rb");

        GpuContext {
            device, queue, res_pl, proj_pl, heb_pl, agg_pl, log_pl,
            res_bgl, proj_bgl, heb_bgl, agg_bgl, log_bgl,
            buf_r_matrix, buf_w_in, buf_s, s_ping: 0,
            buf_x_t, buf_e_t, buf_y_hidden,
            buf_w_up_all, buf_w_down_all, buf_w_out,
            buf_local_s_all, buf_local_e_all, buf_ro_all,
            buf_m, m_ping, buf_m_base, buf_hebbian_params, buf_layer_idx, buf_num_layers,
            buf_output_embeddings, buf_output_bias, buf_logits,
            buf_s_readback, buf_logits_readback, buf_m_readback, num_layers,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Orchestration methods
    // ─────────────────────────────────────────────────────────────────────────────
    
    pub fn upload_x_and_e(&mut self, x_t: &[f32], e_t: &[f32]) {
        self.queue.write_buffer(&self.buf_x_t, 0, bytemuck::cast_slice(x_t));
        self.queue.write_buffer(&self.buf_e_t, 0, bytemuck::cast_slice(e_t));
    }
    
    pub fn dispatch_reservoir(&mut self, enc: &mut wgpu::CommandEncoder) {
        let prev_s = self.s_ping;
        let next_s = 1 - prev_s;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.res_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_r_matrix.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_w_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_s[prev_s].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_x_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_s[next_s].as_entire_binding() },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.res_pl); cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((N_RES as u32 + 63) / 64, 1, 1);
        }
        self.s_ping = next_s;
    }
    
    pub fn dispatch_projections(&mut self, enc: &mut wgpu::CommandEncoder) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.proj_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_s[self.s_ping].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_e_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_w_up_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_w_down_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_local_s_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_local_e_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_num_layers.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.proj_pl); cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(self.num_layers as u32, 1, 1);
        }
    }
    
    pub fn dispatch_hebbian(
        &mut self,
        layer_idx: usize,
        lambda: f32,
        kappa: f32,
        beta3_eta: f32,
        sigma_global: f32,
        alpha_fatigue: f32,
        tau_sat: f32,
        enc: &mut wgpu::CommandEncoder,
    ) {
        let params = [lambda, kappa, beta3_eta, sigma_global, alpha_fatigue, tau_sat, 0.0, 0.0];
        self.queue.write_buffer(&self.buf_hebbian_params[layer_idx], 0, bytemuck::cast_slice(&params));
        
        let prev_m = self.m_ping[layer_idx];
        let next_m = 1 - prev_m;
        
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.heb_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_m[layer_idx][prev_m].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_local_e_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_local_s_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_m_base[layer_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_m[layer_idx][next_m].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_ro_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_hebbian_params[layer_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.buf_layer_idx[layer_idx].as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.heb_pl); cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(1, 16, 1);
        }
        self.m_ping[layer_idx] = next_m;
    }
    
    pub fn dispatch_aggregate(&mut self, enc: &mut wgpu::CommandEncoder) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.agg_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_s[self.s_ping].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_ro_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_w_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_y_hidden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_num_layers.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.agg_pl); cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((D_MODEL as u32 + 63) / 64, 1, 1);
        }
    }
    
    pub fn dispatch_logits(&mut self, enc: &mut wgpu::CommandEncoder) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.log_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_output_embeddings.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_output_bias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_y_hidden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_logits.as_entire_binding() },
            ]
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cp.set_pipeline(&self.log_pl); cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((VOCAB_SIZE as u32 + 255) / 256, 1, 1);
        }
    }
    
    /// Trigger single readback sync for s_t and logits
    pub fn readback_stable_point(&mut self, enc: &mut wgpu::CommandEncoder) -> (Vec<f32>, Vec<f32>) {
        enc.copy_buffer_to_buffer(&self.buf_s[self.s_ping], 0, &self.buf_s_readback, 0, (N_RES * 4) as u64);
        enc.copy_buffer_to_buffer(&self.buf_logits, 0, &self.buf_logits_readback, 0, (VOCAB_SIZE * 4) as u64);
        
        self.queue.submit(std::iter::once(enc.finish()));
        
        let slice_s = self.buf_s_readback.slice(..);
        let slice_log = self.buf_logits_readback.slice(..);
        
        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        
        slice_s.map_async(wgpu::MapMode::Read, move |_| { tx.send(()).unwrap(); });
        slice_log.map_async(wgpu::MapMode::Read, move |_| { tx2.send(()).unwrap(); });
        
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap();
        rx.recv().unwrap();
        
        let map_s = slice_s.get_mapped_range();
        let s_out = bytemuck::cast_slice::<u8, f32>(&map_s).to_vec();
        drop(map_s);
        self.buf_s_readback.unmap();
        
        let map_log = slice_log.get_mapped_range();
        let log_out = bytemuck::cast_slice::<u8, f32>(&map_log).to_vec();
        drop(map_log);
        self.buf_logits_readback.unmap();
        
        (s_out, log_out)
    }

    pub fn reset_reservoir_state(&mut self) {
        self.s_ping = 0;
        let zero = vec![0.0_f32; N_RES];
        self.queue.write_buffer(&self.buf_s[0], 0, bytemuck::cast_slice(&zero));
        self.queue.write_buffer(&self.buf_s[1], 0, bytemuck::cast_slice(&zero));
    }

    pub fn reset_m_states(&mut self, num_layers: usize) {
        for l in 0..num_layers {
            self.m_ping[l] = 0;
            let zero = vec![0.0_f32; RANK_R * RANK_R];
            self.queue.write_buffer(&self.buf_m[l][0], 0, bytemuck::cast_slice(&zero));
            self.queue.write_buffer(&self.buf_m[l][1], 0, bytemuck::cast_slice(&zero));
        }
    }
    
    pub fn readback_all(&self, num_layers: usize) -> crate::gpu_context::GpuReadback {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.buf_s[self.s_ping], 0, &self.buf_s_readback, 0, (N_RES * 4) as u64);
        enc.copy_buffer_to_buffer(&self.buf_logits, 0, &self.buf_logits_readback, 0, (VOCAB_SIZE * 4) as u64);
        for l in 0..num_layers {
            enc.copy_buffer_to_buffer(&self.buf_m[l][self.m_ping[l]], 0, &self.buf_m_readback[l], 0, (RANK_R * RANK_R * 4) as u64);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        
        let slice_s = self.buf_s_readback.slice(..);
        let slice_log = self.buf_logits_readback.slice(..);
        let mut map_txs = vec![];
        let mut map_rxs = vec![];
        for l in 0..num_layers {
            let (t, r) = std::sync::mpsc::channel();
            map_txs.push(t);
            map_rxs.push(r);
        }
        
        let (tx_s, rx_s) = std::sync::mpsc::channel();
        let (tx_l, rx_l) = std::sync::mpsc::channel();
        
        slice_s.map_async(wgpu::MapMode::Read, move |_| tx_s.send(()).unwrap());
        slice_log.map_async(wgpu::MapMode::Read, move |_| tx_l.send(()).unwrap());
        for l in 0..num_layers {
            let tx = map_txs[l].clone();
            self.buf_m_readback[l].slice(..).map_async(wgpu::MapMode::Read, move |_| tx.send(()).unwrap());
        }
        
        self.device.poll(wgpu::Maintain::Wait);
        rx_s.recv().unwrap();
        rx_l.recv().unwrap();
        for r in map_rxs { r.recv().unwrap(); }
        
        let s_out = bytemuck::cast_slice::<u8, f32>(&slice_s.get_mapped_range()).to_vec();
        let log_out = bytemuck::cast_slice::<u8, f32>(&slice_log.get_mapped_range()).to_vec();
        
        let mut m_outs = vec![];
        for l in 0..num_layers {
            let map = self.buf_m_readback[l].slice(..).get_mapped_range();
            m_outs.push(bytemuck::cast_slice::<u8, f32>(&map).to_vec());
        }
        
        self.buf_s_readback.unmap();
        self.buf_logits_readback.unmap();
        for l in 0..num_layers { self.buf_m_readback[l].unmap(); }
        
        crate::gpu_context::GpuReadback {
            s_state: s_out,
            logits: log_out,
            m_states: m_outs,
        }
    }
}

pub struct GpuReadback {
    pub s_state: Vec<f32>,
    pub logits: Vec<f32>,
    pub m_states: Vec<Vec<f32>>,
}
