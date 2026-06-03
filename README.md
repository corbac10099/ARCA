# ARCA: Extreme-Performance GPU Inference Edition

This branch (`extreme-gpu-inference`) implements a zero-sync, pure-VRAM GPU inference pipeline for the ARCA model.

## Architecture "GPU-First"

In the standard implementation, ARCA relies on the GPU for heavy matrix multiplications but uses the CPU as an orchestrator, leading to multiple PCIe syncs per token (Reservoir `s_t` readback, Hebbian `M` readbacks, Logits readback).

In this version:
1. **Fused WGSL Compute Graph**: `Reservoir -> Layer Projections -> Holographic Aggregation -> Logits -> Sampling`.
2. **Zero PCIe Sync in Hot Loop**: All intermediate states (`s_t`, `M`, `y_hidden`, `prev_prediction`) remain resident in VRAM.
3. **GPU-side Sampling**: A reduction shader computes the Top-1 token directly on the GPU. The only data transferred back to the CPU per token is a single 4-byte `u32` token ID.
4. **VRAM Memory Optimization**: `W_up`, `W_down`, and `M_base` for all layers are flattened and uploaded continuously to reduce bind group pressure.

## Usage

```rust
// Extreme zero-sync inference
let token_id = system.forward_step_extreme_inference(text, t, &bpe_ids);
```

---

