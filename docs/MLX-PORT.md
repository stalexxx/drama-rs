# MLX port (Apple Silicon)

Goal: build and run drama-rs on macOS with inference on MLX/Metal instead of CUDA.

## Stage 1 (this commit): the crate builds on machines without CUDA

- `cuda` cargo feature (off by default). With it: `build.rs` compiles
  `kernels/*.cu` with `nvcc` and links `cudart`/`cublas`, exactly as before.
  Without it: no native code, no CUDA link.
- `src/ffi.rs` without `cuda` provides **stub functions with identical
  signatures**. `engine.rs`, `codec.rs`, `model.rs` and `gpu.rs` compile
  **unchanged**; any inference attempt fails at runtime via the existing
  `check`/`launched` paths with:
  `CUDA backend not compiled into this binary (rebuild with --features cuda)`.
- Release builds/CI and docs pass `--features cuda`, so Linux artifacts are
  unaffected.

Build matrix:

| command | result |
|---|---|
| `cargo build` (macOS, no CUDA) | builds, runs, inference errors cleanly |
| `cargo build --features cuda` (Linux + nvcc) | full CUDA binary, as before |
| `cargo test` | host-only tests pass (no GPU needed) |

## Why the port is kernel-by-kernel

The decode loop (`Engine::record_step`) replays one captured CUDA graph per
frame; every heavy op is a hand-written kernel for a custom quantization:

- routed experts: **int4 asymmetric, group 64** (`Q4`)
- everything else linear: **int8 symmetric, group 128** (`Q8`)

Neither dtype exists in MLX, so the hot GEMV paths need **custom Metal
kernels** regardless — MLX only covers the "light" ops (norms, RoPE, softmax,
sampling, permutes). Each `dr_*` function in `src/ffi.rs` is one porting unit;
engine logic above it stays untouched.

## Checklist

- [x] `cuda` feature; stub `ffi.rs`; macOS build + tests green
- [ ] backend runtime (device memory, streams, graph capture)
- [ ] decode GEMV + sampler (first audible output, tiny model)
- [ ] attention + MoE + DeltaNet (correct frames)
- [ ] prefill path
- [ ] codec encode/decode
- [ ] parity run vs CUDA on `check_engine.py` / `check_codec.py`
