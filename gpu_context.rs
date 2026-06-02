/// GPU Context — wgpu device/queue/pipeline initialisation.
///
/// # Architecture overview
///
/// `GpuContext` owns:
///   1. The wgpu `Device` + `Queue`.
///   2. Three compiled `ComputePipeline`s: reservoir, logit, hebbian.
///   3. All persistent GPU-resident `wgpu::Buffer`s that must never
///      round-trip over PCIe during the hot inference/training loop.
///
/// # Adapter selection
///
/// We enumerate all Vulkan/Metal/DX12 adapters and pick the first
/// `DiscreteGpu`.  On AMD hardware this selects the RDNA/GCN device via the
/// Vulkan backend.  If no discrete GPU is found we fall back to the
/// high-performance adapter reported by wgpu.
///
/// # Buffer lifetime policy
///
/// | Buffer                  | Size        | Upload policy              |
/// |-------------------------|-------------|----------------------------|
/// | `buf_r_matrix`          | 64 MiB      | Once at init, never again  |
/// | `buf_w_in`              | 8 MiB       | Once + after SGD update    |
/// | `buf_output_embeddings` | 97 MiB      | Once + after SGD update    |
/// | `buf_output_bias`       | 195 KB      | Once + after SGD update    |
/// | `buf_s[ping/pong]`      | 16 KB       | Zero-copy ping-pong        |
/// | `buf_m[l][ping/pong]`   | 4 KB each   | Zero-copy ping-pong        |
/// | `buf_x_t`               | 2 KB        | Every step (tiny)          |
/// | `buf_y_hidden`          | 2 KB        | Every step (tiny)          |
/// | `buf_local_e/s[l]`      | 128 B each  | Every step per layer       |
///
/// # Feature flag
///
/// This file is only compiled when `--features gpu` is set.
/// All types are re-exported from `crate::gpu_context` so that
/// `metabolic_core.rs` and `memory.rs` can import with `#[cfg(feature="gpu")]`.

use wgpu::util::DeviceExt;

// ─────────────────────────────────────────────────────────────────────────────
// Public constants (mirror the CPU-side values; must stay in sync)
// ─────────────────────────────────────────────────────────────────────────────
pub const N_RES:      usize = 4096;
pub const D_MODEL:    usize = 512;
pub const RANK_R:     usize = 32;
pub const VOCAB_SIZE: usize = 50_000;
pub const TOP_K:      usize = 200;

/// 64-byte alignment matching the Sovereign binary format and AMD
/// cache-line size.
pub const ALIGN: usize = 64;

/// Round `x` up to the next multiple of `align`.
#[inline(always)]
pub const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader sources (embedded inline — no external .wgsl files required)
// ─────────────────────────────────────────────────────────────────────────────

/// `reservoir_update.wgsl`
///
/// Parallelised LSM transition:  s_t = tanh( R·s_{t-1} + W_in·x_t )
///
/// Layout
/// ------
/// • One thread per output element i ∈ [0, N_RES).
/// • Workgroup size 64 = one full RDNA/GCN wavefront.
/// • Grid: ceil(4096 / 64) = 64 workgroups.
///
/// AMD optimisation notes
/// ----------------------
/// • R and W_in are declared `read` (the compiler may promote them to the
///   texture cache on GCN/RDNA, reducing L2 pressure).
/// • s_prev (4096 × 4 B = 16 KiB) fits in the L0 scalar data cache.
/// • x_t (512 × 4 B = 2 KiB) fits entirely in the L0 data cache.
/// • The inner loop over N_RES columns reads R row-by-row; each thread reads
///   a unique row → perfect coalescing with no bank conflicts.
pub const RESERVOIR_SHADER: &str = r#"
// reservoir_update.wgsl ── tanh( R·s_prev + W_in·x_t )

@group(0) @binding(0) var<storage, read>       r_matrix : array<f32>;  // [N_RES * N_RES]
@group(0) @binding(1) var<storage, read>       w_in     : array<f32>;  // [N_RES * D_MODEL]
@group(0) @binding(2) var<storage, read>       s_prev   : array<f32>;  // [N_RES]
@group(0) @binding(3) var<storage, read>       x_t      : array<f32>;  // [D_MODEL]
@group(0) @binding(4) var<storage, read_write> s_out    : array<f32>;  // [N_RES]

const N_RES_C:   u32 = 4096u;
const D_MODEL_C: u32 = 512u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if i >= N_RES_C { return; }

    // ── R[i,:] · s_prev ───────────────────────────────────────────────────
    var acc_r: f32 = 0.0;
    let row_r: u32 = i * N_RES_C;

    // Unroll by 8 to maximise VALU occupancy on RDNA.
    // N_RES_C = 4096 = 512 * 8, so the loop divides evenly.
    var k: u32 = 0u;
    loop {
        if k + 7u >= N_RES_C { break; }
        acc_r += r_matrix[row_r + k]      * s_prev[k]
               + r_matrix[row_r + k + 1u] * s_prev[k + 1u]
               + r_matrix[row_r + k + 2u] * s_prev[k + 2u]
               + r_matrix[row_r + k + 3u] * s_prev[k + 3u]
               + r_matrix[row_r + k + 4u] * s_prev[k + 4u]
               + r_matrix[row_r + k + 5u] * s_prev[k + 5u]
               + r_matrix[row_r + k + 6u] * s_prev[k + 6u]
               + r_matrix[row_r + k + 7u] * s_prev[k + 7u];
        k = k + 8u;
    }
    for (; k < N_RES_C; k = k + 1u) {
        acc_r += r_matrix[row_r + k] * s_prev[k];
    }

    // ── W_in[i,:] · x_t ──────────────────────────────────────────────────
    var acc_w: f32 = 0.0;
    let row_w: u32 = i * D_MODEL_C;

    // Unroll by 8 (D_MODEL = 512 = 64 * 8).
    var m: u32 = 0u;
    loop {
        if m + 7u >= D_MODEL_C { break; }
        acc_w += w_in[row_w + m]      * x_t[m]
               + w_in[row_w + m + 1u] * x_t[m + 1u]
               + w_in[row_w + m + 2u] * x_t[m + 2u]
               + w_in[row_w + m + 3u] * x_t[m + 3u]
               + w_in[row_w + m + 4u] * x_t[m + 4u]
               + w_in[row_w + m + 5u] * x_t[m + 5u]
               + w_in[row_w + m + 6u] * x_t[m + 6u]
               + w_in[row_w + m + 7u] * x_t[m + 7u];
        m = m + 8u;
    }
    for (; m < D_MODEL_C; m = m + 1u) {
        acc_w += w_in[row_w + m] * x_t[m];
    }

    s_out[i] = tanh(acc_r + acc_w);
}
"#;

/// `logit_compute.wgsl`
///
/// Parallelised GEMV over the 50 000 × 512 output embedding table.
///   logits[v] = output_embeddings[v,:] · y_hidden + output_bias[v]
///
/// Layout
/// ------
/// • One thread per vocabulary entry v ∈ [0, VOCAB_SIZE).
/// • Workgroup size 256 → 4 wavefronts per CU on RDNA3, high occupancy.
/// • Grid: ceil(50 000 / 256) = 196 workgroups (4 threads idle in last WG).
///
/// AMD optimisation notes
/// ----------------------
/// • `y_hidden` (512 × 4 B = 2 KB) is broadcast across all threads; it fits
///   in the L0 scalar data cache per CU.
/// • Each thread reads a unique row of `output_embeddings` → fully coalesced.
/// • The inner loop is unrolled by 4 to increase instruction-level parallelism
///   on RDNA SIMD pipelines.
pub const LOGIT_SHADER: &str = r#"
// logit_compute.wgsl ── logits[v] = output_embeddings[v,:] · y_hidden + bias[v]

@group(0) @binding(0) var<storage, read>       output_embeddings : array<f32>; // [VOCAB * D_MODEL]
@group(0) @binding(1) var<storage, read>       output_bias       : array<f32>; // [VOCAB]
@group(0) @binding(2) var<storage, read>       y_hidden          : array<f32>; // [D_MODEL]
@group(0) @binding(3) var<storage, read_write> logits_out        : array<f32>; // [VOCAB]

const VOCAB_SIZE_C: u32 = 50000u;
const D_MODEL_C:    u32 = 512u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v: u32 = gid.x;
    if v >= VOCAB_SIZE_C { return; }

    var dot: f32 = 0.0;
    let row_base: u32 = v * D_MODEL_C;

    // Unroll by 4 (D_MODEL = 512 = 128 * 4).
    var k: u32 = 0u;
    loop {
        if k + 3u >= D_MODEL_C { break; }
        dot += output_embeddings[row_base + k]      * y_hidden[k]
             + output_embeddings[row_base + k + 1u] * y_hidden[k + 1u]
             + output_embeddings[row_base + k + 2u] * y_hidden[k + 2u]
             + output_embeddings[row_base + k + 3u] * y_hidden[k + 3u];
        k = k + 4u;
    }
    for (; k < D_MODEL_C; k = k + 1u) {
        dot += output_embeddings[row_base + k] * y_hidden[k];
    }

    logits_out[v] = dot + output_bias[v];
}
"#;

/// `hebbian_plasticity.wgsl`
///
/// Full Hebbian outer-product update with homeostatic clamping and soft
/// saturation, executed entirely on the GPU for one layer.
///
/// Equations
/// ---------
///   delta_M[i,j]  = local_e[i] * local_s[j]
///   M_next[i,j]   = λ * M_prev[i,j] + κ * (β³η * delta_M[i,j])
///   E             = ‖M_next‖_F²          (Frobenius-squared, workgroup reduce)
///   σ_local       = clamp(σ_g + α_f * tanh(E / τ), 0, 0.95)
///   M_out[i,j]    = tanh( (1-σ)*M_next[i,j] + σ*M_base[i,j] ) * 1.5
///
/// Layout
/// ------
/// • Workgroup (32, 2, 1) = 64 threads = 1 RDNA wavefront.
/// • Grid (1, 1, 1): since RANK_R=32, the 32×32=1024 cells are covered by
///   ceil(32/32) × ceil(32/2) = 1 × 16 workgroups.
///   Wait — to cover 32 columns we need one workgroup in X (32 threads = RANK_R),
///   and to cover 32 rows we need ceil(32/2)=16 workgroups in Y.
///   Dispatch: (1, 16, 1).  See `dispatch_hebbian` in GpuContext.
///
/// Frobenius² reduction
/// --------------------
/// Phase 2 uses a 64-element shared-memory accumulator.  Each thread deposits
/// its squared element; a binary tree reduces to shared_frob[0].  All threads
/// in the workgroup then use the same σ_local for the homeostatic blend,
/// guaranteeing consistent convergence across the full matrix slice owned by
/// the workgroup.
///
/// Note: this workgroup covers a 32×2 tile.  The full 32×32 matrix requires
/// 16 such workgroups (Y dimension).  The Frobenius reduction is therefore
/// *per-tile*, not global.  The resulting σ_local is tile-local, which is an
/// acceptable approximation for RANK_R=32 (tile = 64 cells, global ≈ 1024).
/// For a fully global Frobenius a two-pass kernel would be needed; the
/// per-tile version matches the CPU reference implementation's precision.
pub const HEBBIAN_SHADER: &str = r#"
// hebbian_plasticity.wgsl ── Hebbian outer-product + homeostatic saturation

struct HebbianParams {
    lambda       : f32,   // memory decay factor λ
    kappa        : f32,   // conductance gate κ = sigmoid(γ)
    beta3_eta    : f32,   // β³ * η_lr  (pre-multiplied on CPU)
    sigma_global : f32,   // homeostatic reset strength σ_global
    alpha_fatigue: f32,   // fatigue coefficient α_f
    tau_sat      : f32,   // saturation time-constant τ
    _pad0        : f32,   // padding → 32-byte uniform alignment
    _pad1        : f32,
}

@group(0) @binding(0) var<storage, read>       m_prev  : array<f32>;         // [RANK_R * RANK_R]
@group(0) @binding(1) var<storage, read>       local_e : array<f32>;         // [RANK_R]
@group(0) @binding(2) var<storage, read>       local_s : array<f32>;         // [RANK_R]
@group(0) @binding(3) var<storage, read>       m_base  : array<f32>;         // [RANK_R * RANK_R]
@group(0) @binding(4) var<storage, read_write> m_out   : array<f32>;         // [RANK_R * RANK_R]
@group(0) @binding(5) var<uniform>             params  : HebbianParams;

const RANK_R_C : u32 = 32u;
const WG_SIZE_C: u32 = 64u;   // workgroup_size(32, 2, 1) = 64 threads

var<workgroup> shared_frob: array<f32, 64>;

@compute @workgroup_size(32, 2, 1)
fn main(
    @builtin(global_invocation_id)   gid : vec3<u32>,
    @builtin(local_invocation_index) lid : u32,
) {
    let i: u32 = gid.x;   // column index in M (0..RANK_R)
    let j: u32 = gid.y;   // row index    in M (0..RANK_R)

    // ── Phase 1: Compute M_next for cell (j, i) ────────────────────────────
    // Note: row = j, col = i.  Row-major index = j * RANK_R + i.
    var m_next_val: f32 = 0.0;
    if i < RANK_R_C && j < RANK_R_C {
        let idx: u32   = j * RANK_R_C + i;
        let delta: f32 = local_e[j] * local_s[i];
        m_next_val = params.lambda * m_prev[idx]
                   + params.kappa * params.beta3_eta * delta;
    }

    // ── Phase 2: Frobenius² tile reduction ─────────────────────────────────
    shared_frob[lid] = m_next_val * m_next_val;
    workgroupBarrier();

    // Binary tree reduction over 64 threads
    for (var stride: u32 = WG_SIZE_C >> 1u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_frob[lid] += shared_frob[lid + stride];
        }
        workgroupBarrier();
    }
    let frob_sq: f32 = shared_frob[0];  // sum of squared elements in this tile

    // ── Phase 3: Homeostatic clamping + soft saturation ────────────────────
    if i < RANK_R_C && j < RANK_R_C {
        let idx: u32 = j * RANK_R_C + i;

        let fatigue: f32     = params.alpha_fatigue * tanh(frob_sq / params.tau_sat);
        let sigma_local: f32 = clamp(params.sigma_global + fatigue, 0.0, 0.95);

        let blended: f32 = (1.0 - sigma_local) * m_next_val
                         +        sigma_local   * m_base[idx];

        m_out[idx] = tanh(blended) * 1.5;
    }
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// GpuContext
// ─────────────────────────────────────────────────────────────────────────────

/// Central GPU handle: device + queue + pipelines + all persistent VRAM buffers.
///
/// Owned by `ArcaSystem` in `main.rs`.  Passed by `&mut` reference into the
/// hot-loop dispatch helpers.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,

    // ── Compiled compute pipelines ────────────────────────────────────────
    reservoir_pipeline: wgpu::ComputePipeline,
    logit_pipeline:     wgpu::ComputePipeline,
    hebbian_pipeline:   wgpu::ComputePipeline,

    // ── Bind-group layouts (reused every dispatch) ────────────────────────
    reservoir_bgl: wgpu::BindGroupLayout,
    logit_bgl:     wgpu::BindGroupLayout,
    hebbian_bgl:   wgpu::BindGroupLayout,

    // ── Persistent GPU-resident tensors ───────────────────────────────────
    //
    // R matrix — 4096×4096 f32 ≈ 64 MiB.  Read-only after init.
    buf_r_matrix: wgpu::Buffer,
    // W_in — 4096×512 f32 ≈ 8 MiB.  Re-uploaded after slow-learning SGD.
    buf_w_in:     wgpu::Buffer,
    // Reservoir state ping-pong: buf_s[s_ping] = s_{t-1}, buf_s[1-s_ping] = s_t.
    buf_s:        [wgpu::Buffer; 2],
    /// Current "input" ping index (0 or 1).
    pub s_ping:   usize,

    // Output embedding table — 50000×512 f32 ≈ 97 MiB.
    buf_output_embeddings: wgpu::Buffer,
    // Output bias — 50000 f32 ≈ 195 KB.
    buf_output_bias:       wgpu::Buffer,
    // GPU-computed full logit vector — 50000 f32.  Staging (GPU only).
    buf_logits:            wgpu::Buffer,
    // CPU readback for top-K selection (only mapped at checkpoint or per step for top-K).
    buf_logits_readback:   wgpu::Buffer,

    // x_t upload scratch — D_MODEL f32 (2 KB).  Written once per step.
    buf_x_t:      wgpu::Buffer,
    // y_hidden (aggregated memory output) — D_MODEL f32 (2 KB).
    buf_y_hidden: wgpu::Buffer,

    // Per-layer Hebbian buffers
    /// M matrices ping-pong: buf_m[l][m_ping[l]] = M_{t-1}.
    pub buf_m:             Vec<[wgpu::Buffer; 2]>,
    /// Current "input" ping index per layer.
    pub m_ping:            Vec<usize>,
    buf_local_e:           Vec<wgpu::Buffer>,   // RANK_R f32 each
    buf_local_s:           Vec<wgpu::Buffer>,   // RANK_R f32 each
    buf_m_base:            Vec<wgpu::Buffer>,   // RANK_R*RANK_R f32 each
    buf_hebbian_params:    Vec<wgpu::Buffer>,   // 32-byte uniform each

    // Readback buffers (only mapped at checkpoint / end-of-training)
    buf_s_readback:        wgpu::Buffer,
    buf_m_readback:        Vec<wgpu::Buffer>,
}

impl GpuContext {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Initialise the GPU context.
    ///
    /// Parameters
    /// ----------
    /// - `num_layers`      — number of `BioInspiredLayer`s.
    /// - `r_matrix_data`   — row-major f32 slice, length `N_RES * N_RES`.
    /// - `w_in_data`       — row-major f32 slice, length `N_RES * D_MODEL`.
    /// - `out_emb_data`    — row-major f32 slice, length `VOCAB_SIZE * D_MODEL`.
    /// - `out_bias_data`   — f32 slice, length `VOCAB_SIZE`.
    /// - `m_base_data`     — per-layer flat f32 slice, each `RANK_R * RANK_R`.
    ///
    /// This function **blocks** the current thread until the GPU adapter and
    /// device are ready (uses `pollster::block_on`).
    pub fn new(
        num_layers:     usize,
        r_matrix_data:  &[f32],
        w_in_data:      &[f32],
        out_emb_data:   &[f32],
        out_bias_data:  &[f32],
        m_base_data:    &[Vec<f32>],
    ) -> Self {
        // ── 1. Adapter selection ──────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN
                    | wgpu::Backends::METAL
                    | wgpu::Backends::DX12,
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = pollster::block_on(async {
            // Prefer AMD/NVIDIA discrete GPU; fall back to any high-perf adapter.
            let all: Vec<wgpu::Adapter> =
                instance.enumerate_adapters(wgpu::Backends::all()).collect();
            let mut chosen: Option<wgpu::Adapter> = None;
            for a in &all {
                let info = a.get_info();
                eprintln!(
                    "[ARCA GPU] Found adapter: {} ({:?}) backend={:?}",
                    info.name, info.device_type, info.backend
                );
                if info.device_type == wgpu::DeviceType::DiscreteGpu && chosen.is_none() {
                    chosen = Some(a.clone());
                }
            }
            if let Some(a) = chosen {
                return a;
            }
            // Fallback
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("[ARCA GPU] No GPU adapter found")
        });

        let info = adapter.get_info();
        eprintln!(
            "[ARCA GPU] Selected: {} ({:?}) backend={:?}",
            info.name, info.device_type, info.backend
        );

        // ── 2. Device + queue ─────────────────────────────────────────────
        let (device, queue) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ARCA-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        // R matrix  ≈  64 MiB
                        // Embedding ≈  97 MiB
                        // Safety headroom: 256 MiB total
                        max_buffer_size:               256 * 1024 * 1024,
                        max_storage_buffer_binding_size: 256 * 1024 * 1024,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            ),
        )
        .expect("[ARCA GPU] Failed to create device");

        // ── 3. Compile shaders ────────────────────────────────────────────
        let res_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("reservoir_update"),
            source: wgpu::ShaderSource::Wgsl(RESERVOIR_SHADER.into()),
        });
        let log_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("logit_compute"),
            source: wgpu::ShaderSource::Wgsl(LOGIT_SHADER.into()),
        });
        let heb_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("hebbian_plasticity"),
            source: wgpu::ShaderSource::Wgsl(HEBBIAN_SHADER.into()),
        });

        // ── 4. Bind-group layouts ─────────────────────────────────────────
        // Helper closures — returns a BindGroupLayoutEntry for each binding type.
        let bgl_storage_ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding:    b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };
        let bgl_storage_rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding:    b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };
        let bgl_uniform = |b: u32| wgpu::BindGroupLayoutEntry {
            binding:    b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };

        let reservoir_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reservoir-bgl"),
            entries: &[
                bgl_storage_ro(0), // r_matrix
                bgl_storage_ro(1), // w_in
                bgl_storage_ro(2), // s_prev
                bgl_storage_ro(3), // x_t
                bgl_storage_rw(4), // s_out
            ],
        });
        let logit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("logit-bgl"),
            entries: &[
                bgl_storage_ro(0), // output_embeddings
                bgl_storage_ro(1), // output_bias
                bgl_storage_ro(2), // y_hidden
                bgl_storage_rw(3), // logits_out
            ],
        });
        let hebbian_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hebbian-bgl"),
            entries: &[
                bgl_storage_ro(0), // m_prev
                bgl_storage_ro(1), // local_e
                bgl_storage_ro(2), // local_s
                bgl_storage_ro(3), // m_base
                bgl_storage_rw(4), // m_out
                bgl_uniform(5),    // params (HebbianParams)
            ],
        });

        // ── 5. Compute pipelines ──────────────────────────────────────────
        let make_pipeline = |module:  &wgpu::ShaderModule,
                              bgl:    &wgpu::BindGroupLayout,
                              label:  &str|
         -> wgpu::ComputePipeline {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some(label),
                layout:              Some(&layout),
                module,
                entry_point:         "main",
                compilation_options: Default::default(),
                cache:               None,
            })
        };

        let reservoir_pipeline =
            make_pipeline(&res_module, &reservoir_bgl, "reservoir-pl");
        let logit_pipeline =
            make_pipeline(&log_module, &logit_bgl,     "logit-pl");
        let hebbian_pipeline =
            make_pipeline(&heb_module, &hebbian_bgl,   "hebbian-pl");

        // ── 6. Allocate / upload persistent GPU buffers ───────────────────
        let f32_bytes = |n: usize| (n * 4) as u64;

        // Upload helper: creates a STORAGE|COPY_SRC|COPY_DST buffer with data.
        let upload_buf = |data: &[f32], label: &str| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytemuck::cast_slice(data),
                usage:    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
            })
        };

        // Empty helper: creates a zero-init STORAGE buffer (no initial upload).
        let empty_storage = |n: usize, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(label),
                size:               f32_bytes(n),
                usage:              wgpu::BufferUsages::STORAGE
                                  | wgpu::BufferUsages::COPY_DST
                                  | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        // Readback helper: MAP_READ | COPY_DST — only used at checkpoint.
        let readback_buf = |n: usize, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(label),
                size:               f32_bytes(n),
                usage:              wgpu::BufferUsages::MAP_READ
                                  | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Large persistent tensors
        let buf_r_matrix          = upload_buf(r_matrix_data, "r_matrix");
        let buf_w_in              = upload_buf(w_in_data,     "w_in");
        let buf_output_embeddings = upload_buf(out_emb_data,  "output_embeddings");
        let buf_output_bias       = upload_buf(out_bias_data, "output_bias");

        // Ping-pong reservoir state (zero-initialised)
        let buf_s = [
            empty_storage(N_RES, "s_buf_0"),
            empty_storage(N_RES, "s_buf_1"),
        ];
        let buf_s_readback = readback_buf(N_RES, "s_readback");

        // Logit output + readback
        let buf_logits          = empty_storage(VOCAB_SIZE, "logits");
        let buf_logits_readback = readback_buf(VOCAB_SIZE, "logits_readback");

        // Per-step upload scratches
        let buf_x_t      = empty_storage(D_MODEL, "x_t");
        let buf_y_hidden = empty_storage(D_MODEL, "y_hidden");

        // Per-layer Hebbian buffers
        let mut buf_m              = Vec::with_capacity(num_layers);
        let mut m_ping             = Vec::with_capacity(num_layers);
        let mut buf_local_e        = Vec::with_capacity(num_layers);
        let mut buf_local_s        = Vec::with_capacity(num_layers);
        let mut buf_m_base         = Vec::with_capacity(num_layers);
        let mut buf_hebbian_params = Vec::with_capacity(num_layers);
        let mut buf_m_readback     = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            buf_m.push([
                empty_storage(RANK_R * RANK_R, &format!("m_buf_{}_0", l)),
                empty_storage(RANK_R * RANK_R, &format!("m_buf_{}_1", l)),
            ]);
            m_ping.push(0usize);

            buf_local_e.push(empty_storage(RANK_R, &format!("local_e_{}", l)));
            buf_local_s.push(empty_storage(RANK_R, &format!("local_s_{}", l)));
            buf_m_base.push(upload_buf(&m_base_data[l], &format!("m_base_{}", l)));

            // HebbianParams uniform: 8 × f32 = 32 bytes (6 meaningful + 2 padding).
            buf_hebbian_params.push(device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&format!("hebbian_params_{}", l)),
                size:               32,
                usage:              wgpu::BufferUsages::UNIFORM
                                  | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            buf_m_readback.push(readback_buf(RANK_R * RANK_R, &format!("m_readback_{}", l)));
        }

        GpuContext {
            device,
            queue,
            reservoir_pipeline,
            logit_pipeline,
            hebbian_pipeline,
            reservoir_bgl,
            logit_bgl,
            hebbian_bgl,
            buf_r_matrix,
            buf_w_in,
            buf_s,
            s_ping: 0,
            buf_output_embeddings,
            buf_output_bias,
            buf_logits,
            buf_logits_readback,
            buf_x_t,
            buf_y_hidden,
            buf_m,
            m_ping,
            buf_local_e,
            buf_local_s,
            buf_m_base,
            buf_hebbian_params,
            buf_s_readback,
            buf_m_readback,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Hot-loop dispatch helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Upload `x_t` (D_MODEL f32 = 2 KiB) and execute the reservoir shader.
    ///
    /// The updated state is written to `buf_s[1 - s_ping]`; `s_ping` is
    /// flipped so subsequent dispatches read the new state.
    ///
    /// **Non-blocking** — command submission returns immediately; the GPU
    /// executes asynchronously.
    pub fn dispatch_reservoir(&mut self, x_t_data: &[f32]) {
        debug_assert_eq!(x_t_data.len(), D_MODEL);

        self.queue.write_buffer(&self.buf_x_t, 0, bytemuck::cast_slice(x_t_data));

        let prev = self.s_ping;
        let next = 1 - prev;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("reservoir-bg"),
            layout:  &self.reservoir_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_r_matrix.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_w_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_s[prev].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_x_t.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_s[next].as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reservoir-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("reservoir-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.reservoir_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // ceil(4096 / 64) = 64 workgroups
            cpass.dispatch_workgroups((N_RES as u32 + 63) / 64, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        self.s_ping = next;
    }

    /// Run the logit shader.
    ///
    /// **Pre-condition**: `buf_y_hidden` must be populated by a prior call to
    /// `upload_y_hidden`.  The result is written to `buf_logits`.
    ///
    /// **Non-blocking**.
    pub fn dispatch_logits(&self) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("logit-bg"),
            layout:  &self.logit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_output_embeddings.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_output_bias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_y_hidden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_logits.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logit-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("logit-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.logit_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // ceil(50000 / 256) = 196 workgroups (4 threads idle in last WG)
            cpass.dispatch_workgroups((VOCAB_SIZE as u32 + 255) / 256, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// Upload `local_e` and `local_s` for layer `layer_idx`, then run the
    /// Hebbian shader.
    ///
    /// The updated M matrix is written to `buf_m[layer_idx][1 - m_ping[layer_idx]]`;
    /// the ping index is flipped on return.
    ///
    /// **Non-blocking**.
    pub fn dispatch_hebbian(
        &mut self,
        layer_idx:     usize,
        local_e_data:  &[f32],
        local_s_data:  &[f32],
        lambda:        f32,
        kappa:         f32,
        beta3_eta:     f32,
        sigma_global:  f32,
        alpha_fatigue: f32,
        tau_sat:       f32,
    ) {
        debug_assert_eq!(local_e_data.len(), RANK_R);
        debug_assert_eq!(local_s_data.len(), RANK_R);
        let l = layer_idx;

        // Upload small (128-byte) input vectors
        self.queue.write_buffer(&self.buf_local_e[l], 0, bytemuck::cast_slice(local_e_data));
        self.queue.write_buffer(&self.buf_local_s[l], 0, bytemuck::cast_slice(local_s_data));

        // Pack HebbianParams uniform (8 × f32 = 32 bytes; last 2 are padding)
        let params_raw: [f32; 8] = [
            lambda, kappa, beta3_eta, sigma_global, alpha_fatigue, tau_sat, 0.0, 0.0,
        ];
        self.queue.write_buffer(
            &self.buf_hebbian_params[l],
            0,
            bytemuck::cast_slice(&params_raw),
        );

        let prev = self.m_ping[l];
        let next = 1 - prev;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some(&format!("hebbian-bg-{}", l)),
            layout:  &self.hebbian_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_m[l][prev].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_local_e[l].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_local_s[l].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_m_base[l].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_m[l][next].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_hebbian_params[l].as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("hebbian-enc-{}", l)),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some(&format!("hebbian-pass-{}", l)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.hebbian_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Workgroup (32, 2, 1) covers a 32×2 tile.
            // Full grid for 32×32 = 1 × 16 workgroups:
            //   X: RANK_R / 32 = 1
            //   Y: RANK_R /  2 = 16
            cpass.dispatch_workgroups(
                RANK_R as u32 / 32,
                RANK_R as u32 / 2,
                1,
            );
        }
        self.queue.submit(std::iter::once(enc.finish()));
        self.m_ping[l] = next;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Per-step readback helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Read back the current reservoir state (N_RES f32 = 16 KiB).
    ///
    /// Used to maintain the CPU shadow of s_t needed for LSH hashing and
    /// the Hebbian layer projections.  Called once per forward step.
    pub fn readback_s(&self) -> Vec<f32> {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("s-readback-enc"),
        });
        enc.copy_buffer_to_buffer(
            &self.buf_s[self.s_ping], 0,
            &self.buf_s_readback,    0,
            (N_RES * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc.finish()));
        map_read_f32(&self.device, &self.buf_s_readback, N_RES)
    }

    /// Read back the full logit vector (VOCAB_SIZE f32 = 195 KiB).
    ///
    /// Called once per forward step for CPU-side top-K selection.
    pub fn readback_logits(&self) -> Vec<f32> {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logits-readback-enc"),
        });
        enc.copy_buffer_to_buffer(
            &self.buf_logits,         0,
            &self.buf_logits_readback, 0,
            (VOCAB_SIZE * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc.finish()));
        map_read_f32(&self.device, &self.buf_logits_readback, VOCAB_SIZE)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Checkpoint readback (only at save points — blocks until GPU is idle)
    // ─────────────────────────────────────────────────────────────────────

    /// Read back **all** persistent GPU state into CPU-owned `Vec<f32>` buffers.
    ///
    /// This function **blocks** until the GPU is idle and all transfers complete.
    /// It should be called only at checkpoint intervals or end-of-training, not
    /// in the hot loop.
    ///
    /// Returns `GpuReadback` containing:
    /// - `s_state`:  current reservoir state (N_RES)
    /// - `logits`:   most-recently computed logit vector (VOCAB_SIZE)
    /// - `m_states`: per-layer memory matrices [num_layers][RANK_R * RANK_R]
    pub fn readback_all(&self, num_layers: usize) -> GpuReadback {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("checkpoint-readback-enc"),
        });

        enc.copy_buffer_to_buffer(
            &self.buf_s[self.s_ping], 0,
            &self.buf_s_readback,    0,
            (N_RES * 4) as u64,
        );
        enc.copy_buffer_to_buffer(
            &self.buf_logits,         0,
            &self.buf_logits_readback, 0,
            (VOCAB_SIZE * 4) as u64,
        );
        for l in 0..num_layers {
            enc.copy_buffer_to_buffer(
                &self.buf_m[l][self.m_ping[l]], 0,
                &self.buf_m_readback[l],        0,
                (RANK_R * RANK_R * 4) as u64,
            );
        }
        self.queue.submit(std::iter::once(enc.finish()));

        let s_state  = map_read_f32(&self.device, &self.buf_s_readback,    N_RES);
        let logits   = map_read_f32(&self.device, &self.buf_logits_readback, VOCAB_SIZE);
        let m_states = (0..num_layers)
            .map(|l| map_read_f32(&self.device, &self.buf_m_readback[l], RANK_R * RANK_R))
            .collect();

        GpuReadback { s_state, logits, m_states }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Upload helpers (called after slow-learning SGD weight updates on CPU)
    // ─────────────────────────────────────────────────────────────────────

    /// Write `y_hidden` (output of the CPU aggregator) to the GPU before
    /// calling `dispatch_logits`.
    pub fn upload_y_hidden(&self, data: &[f32]) {
        debug_assert_eq!(data.len(), D_MODEL);
        self.queue.write_buffer(&self.buf_y_hidden, 0, bytemuck::cast_slice(data));
    }

    /// Re-upload `W_in` after a slow-learning SGD step updated it on CPU.
    pub fn upload_w_in(&self, data: &[f32]) {
        debug_assert_eq!(data.len(), N_RES * D_MODEL);
        self.queue.write_buffer(&self.buf_w_in, 0, bytemuck::cast_slice(data));
    }

    /// Re-upload the output embedding table after a slow-learning SGD step.
    pub fn upload_output_embeddings(&self, data: &[f32]) {
        debug_assert_eq!(data.len(), VOCAB_SIZE * D_MODEL);
        self.queue.write_buffer(&self.buf_output_embeddings, 0, bytemuck::cast_slice(data));
    }

    /// Re-upload the output bias after a slow-learning SGD step.
    pub fn upload_output_bias(&self, data: &[f32]) {
        debug_assert_eq!(data.len(), VOCAB_SIZE);
        self.queue.write_buffer(&self.buf_output_bias, 0, bytemuck::cast_slice(data));
    }

    /// Zero-fill the reservoir state (called between batches by `ArcaSystem::reset_state`).
    pub fn reset_reservoir_state(&self) {
        let zeros = vec![0u8; N_RES * 4];
        self.queue.write_buffer(&self.buf_s[self.s_ping], 0, &zeros);
    }

    /// Zero-fill all layer M matrices (called between batches).
    pub fn reset_m_states(&self, num_layers: usize) {
        let zeros = vec![0u8; RANK_R * RANK_R * 4];
        for l in 0..num_layers {
            self.queue.write_buffer(&self.buf_m[l][self.m_ping[l]], 0, &zeros);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Per-step per-layer M readback (1 KiB per layer)
    // ─────────────────────────────────────────────────────────────────────

    /// Read back the current M matrix for layer `layer_idx`.
    ///
    /// Called once per forward step per layer in `ArcaSystem::forward_step_gpu`
    /// to maintain the CPU shadow needed by `BioInspiredLayer::read_out_cpu`.
    ///
    /// Transfer size: RANK_R × RANK_R × 4 = 4 096 bytes ≈ **4 KiB per layer**.
    /// For L=4 layers this is 16 KiB/step — negligible PCIe overhead.
    pub fn readback_m_layer(&self, layer_idx: usize) -> Vec<f32> {
        let l = layer_idx;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("m-readback-enc-{}", l)),
        });
        enc.copy_buffer_to_buffer(
            &self.buf_m[l][self.m_ping[l]], 0,
            &self.buf_m_readback[l],        0,
            (RANK_R * RANK_R * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc.finish()));
        map_read_f32(&self.device, &self.buf_m_readback[l], RANK_R * RANK_R)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuReadback — data returned from a full checkpoint readback
// ─────────────────────────────────────────────────────────────────────────────

/// All GPU-resident state copied back to CPU memory for checkpointing.
pub struct GpuReadback {
    /// Current reservoir state s_t — length N_RES.
    pub s_state:  Vec<f32>,
    /// Most recently computed full logit vector — length VOCAB_SIZE.
    pub logits:   Vec<f32>,
    /// Per-layer memory matrices — `m_states[l]` has length RANK_R * RANK_R.
    pub m_states: Vec<Vec<f32>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: blocking buffer-map helper
// ─────────────────────────────────────────────────────────────────────────────

/// Map `buf` for reading, copy its contents into a `Vec<f32>`, then unmap.
///
/// Blocks the current thread until the map is ready by polling the device.
fn map_read_f32(device: &wgpu::Device, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("map_read_f32: channel closed unexpectedly")
        .expect("map_read_f32: GPU buffer map failed");

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped)
        .iter()
        .cloned()
        .collect();
    drop(mapped);
    buf.unmap();
    debug_assert_eq!(result.len(), len, "map_read_f32: length mismatch");
    result
}
