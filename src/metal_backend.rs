//! Apple-Silicon backend for `src/ffi.rs` (feature `metal`).
//!
//! How it maps to CUDA concepts:
//! - Unified memory: every "device" allocation is an `MTLBuffer` with shared storage,
//!   so its address is stable and host-visible. CPU reference code reads/writes it
//!   directly; MSL kernels take it via `set_buffer`.
//! - Execution is synchronous in v1: each launch commits and waits, so
//!   `dr_stream_sync` is a no-op. Graphs are recorded as closure lists (`dr_graph_*`)
//!   and replayed eagerly, which preserves `engine.rs` semantics exactly.
//! - Ops already verified on the GPU are marked MSL below; the rest are CPU reference
//!   implementations over the same host-visible buffers (bit-exact with
//!   `kernels/*.cu` by construction, except the sampler RNG, see below).

use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::ffi::{GraphExec, Half, Stream};

// ---------------------------------------------------------------------------
// thread-safe Metal handle wrappers (Metal objects are internally synchronized)

#[derive(Clone)]
struct MDevice(metal::Device);
unsafe impl Send for MDevice {}
unsafe impl Sync for MDevice {}

#[derive(Clone)]
struct MQueue(metal::CommandQueue);
unsafe impl Send for MQueue {}
unsafe impl Sync for MQueue {}

#[derive(Clone)]
struct MLib(metal::Library);
unsafe impl Send for MLib {}
unsafe impl Sync for MLib {}

#[derive(Clone)]
struct MPipe(metal::ComputePipelineState);
unsafe impl Send for MPipe {}
unsafe impl Sync for MPipe {}

#[derive(Clone)]
struct MBuf(metal::Buffer);
unsafe impl Send for MBuf {}
unsafe impl Sync for MBuf {}

// ---------------------------------------------------------------------------
// device context

struct Ctx {
    device: MDevice,
    queue: MQueue,
    lib: MLib,
    pipes: Mutex<HashMap<&'static str, MPipe>>,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> &'static Ctx {
    CTX.get_or_init(|| {
        let device = metal::Device::system_default().expect("metal: no system default device");
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(MSL_SRC, &metal::CompileOptions::new())
            .unwrap_or_else(|e| panic!("metal: MSL compile failed: {e}"));
        Ctx { device: MDevice(device), queue: MQueue(queue), lib: MLib(lib), pipes: Mutex::new(HashMap::new()) }
    })
}

fn pipe(name: &'static str) -> MPipe {
    let c = ctx();
    if let Some(p) = c.pipes.lock().unwrap().get(name) {
        return p.clone();
    }
    let f = c.lib.0.get_function(name, None).unwrap_or_else(|e| panic!("metal: function {name}: {e}"));
    let p = MPipe(c.device.0.new_compute_pipeline_state_with_function(&f).unwrap_or_else(|e| {
        panic!("metal: pipeline {name}: {e}")
    }));
    c.pipes.lock().unwrap().insert(name, p.clone());
    p
}

// ---------------------------------------------------------------------------
// buffer registry: raw pointer -> (buffer, base). `DevBuf::at` creates interior
// pointers, so lookup finds the containing buffer and its offset.

static BUFS: Mutex<BTreeMap<usize, MBuf>> = Mutex::new(BTreeMap::new());

fn track(buf: &metal::Buffer) -> *mut c_void {
    let p = buf.contents() as *mut c_void;
    BUFS.lock().unwrap().insert(p as usize, MBuf(buf.clone()));
    p
}

fn lookup(ptr: *const c_void) -> Option<(MBuf, u64)> {
    if ptr.is_null() {
        return None;
    }
    let m = BUFS.lock().unwrap();
    let p = ptr as usize;
    let (&base, buf) = m.range(..=p).next_back()?;
    let off = p - base;
    if off > buf.0.length() as usize {
        return None;
    }
    Some((buf.clone(), off as u64))
}

fn require(ptr: *const c_void) -> (MBuf, u64) {
    lookup(ptr).expect("metal: pointer is not a tracked device buffer")
}

// ---------------------------------------------------------------------------
// pinned host memory

static HOST: Mutex<BTreeMap<usize, std::alloc::Layout>> = Mutex::new(BTreeMap::new());

// ---------------------------------------------------------------------------
// streams (synchronous execution; the id only routes graph capture)

static NEXT_STREAM: AtomicUsize = AtomicUsize::new(1);
static QUEUES: Mutex<BTreeMap<usize, MQueue>> = Mutex::new(BTreeMap::new());

// ---------------------------------------------------------------------------
// graph capture: record closures, replay eagerly (thread-local).
//
// Execution is synchronous, so capture only needs to live until the matching
// `dr_graph_end` on the SAME thread (this matches engine.rs: Engine::new records
// the graph and step() replays it on one thread; the codec never uses graphs).
// Cross-thread record/replay is rejected in dr_graph_end/dr_graph_launch.

type Op = Box<dyn Fn()>;

struct Capture {
    stream: usize,
    thread: std::thread::ThreadId,
    ops: Vec<Op>,
}

std::thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Capture>> = const { std::cell::RefCell::new(None) };
}

struct GraphData {
    ops: Vec<Op>,
}

/// Run `f` now, or record it when `st` is being captured on this thread.
fn launch(st: Stream, f: impl Fn() + 'static) -> i32 {
    let key = st as usize;
    let mut f = Some(f);
    let captured = CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        match c.as_mut() {
            Some(cap) if cap.stream == key => {
                cap.ops.push(Box::new(f.take().expect("launch: closure taken twice")));
                true
            }
            _ => false,
        }
    });
    if captured {
        return 0;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f.expect("launch: closure missing"))) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

// ---------------------------------------------------------------------------
// small helpers shared by the CPU reference ops

#[inline]
fn f32_at(p: *const f32, i: usize) -> f32 {
    unsafe { *p.add(i) }
}

#[inline]
fn f32s(p: *const f32, n: usize) -> &'static [f32] {
    unsafe { std::slice::from_raw_parts(p, n) }
}

#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn f16_to_f32(h: Half) -> f32 {
    half_to_f32(h)
}

fn half_to_f32(h: u16) -> f32 {
    // IEEE 754 binary16 -> f32.
    let s = (h & 0x8000) as u32;
    let e = (h & 0x7c00) as u32;
    let m = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 16
        } else {
            // subnormal: renormalize
            let mut m = m;
            let mut e = 127 - 15 + 1;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            (s << 16) | (e << 23) | (m << 13)
        }
    } else if e == 0x7c00 {
        (s << 16) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 16) | (((e >> 10) + (127 - 15)) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even f32 -> bf16 bits (matches `__float2bfloat16`; NaN -> 0x7fc0).
fn f32_to_bf16_rn(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rms_inv(x: &[f32], eps: f32) -> f32 {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    (ss / x.len() as f32 + eps).sqrt().recip()
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MSL kernels (MSL = Metal Shading Language)

const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline ushort f32_to_bf16_rn(float v) {
    uint b = as_type<uint>(v);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) return (ushort)0x7fc0;
    return (ushort)(((b + 0x7fffu + ((b >> 16) & 1u)) >> 16));
}
inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }

struct NormP { uint rows; uint d; float eps; };
kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    device ushort* yb [[buffer(3)]],
                    constant NormP& p [[buffer(4)]],
                    uint3 tpg [[threads_per_threadgroup]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.x;
    if (r >= p.rows) return;
    uint d = p.d;
    float acc = 0.0f;
    for (uint i = tid.x; i < d; i += tpg.x) { float v = x[r*d+i]; acc += v*v; }
    threadgroup float part[256];
    part[tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(part[0] / (float)d + p.eps);
    bool has_w = (w != nullptr);
    for (uint i = tid.x; i < d; i += tpg.x) {
        float v = x[r*d+i] * inv * (has_w ? w[i] : 1.0f);
        if (y != nullptr) y[r*d+i] = v;
        if (yb != nullptr) yb[r*d+i] = f32_to_bf16_rn(v);
    }
}

struct GemvP { uint n; uint k; };
kernel void gemv_bf16(device const ushort* W [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemvP& p [[buffer(3)]],
                      uint3 tpg [[threads_per_threadgroup]],
                      uint3 tid [[thread_position_in_threadgroup]],
                      uint3 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.x;
    if (r >= p.n) return;
    float acc = 0.0f;
    for (uint j = tid.x; j < p.k; j += tpg.x) acc += bf16_to_f32(W[r*p.k+j]) * x[j];
    threadgroup float part[256];
    part[tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid.x == 0) y[r] = part[0];
}

struct DqP { uint n; uint k; };
kernel void dequant_q8(device const char* W [[buffer(0)]],
                       device const ushort* S [[buffer(1)]],
                       device ushort* o [[buffer(2)]],
                       constant DqP& p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    uint total = p.n * p.k;
    if (tid >= total) return;
    uint k = tid % p.k;
    float s = (float)as_type<half>(S[(tid / p.k) * (p.k / 128) + k / 128]);
    o[tid] = f32_to_bf16_rn((float)W[tid] * s);
}

kernel void f32_to_bf16(device const float* x [[buffer(0)]],
                        device ushort* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid < n) y[tid] = f32_to_bf16_rn(x[tid]);
}

kernel void swiglu_rows(device const float* h [[buffer(0)]],
                        device ushort* y [[buffer(1)]],
                        constant uint2& dims [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    uint I = dims.y;
    uint total = dims.x * I;
    if (tid >= total) return;
    uint j = tid % I;
    uint r = tid / I;
    float g = h[r * 2 * I + j];
    float u = h[r * 2 * I + I + j];
    y[tid] = f32_to_bf16_rn(g / (1.0f + exp(-g)) * u);
}
"#;

#[repr(C)]
struct NormP {
    rows: u32,
    d: u32,
    eps: f32,
}

#[repr(C)]
struct GemvP {
    n: u32,
    k: u32,
}

#[repr(C)]
struct DqP {
    n: u32,
    k: u32,
}

fn pb<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Dispatch an MSL kernel. `bufs` are (pointer, extra_offset); null pointers become
/// nil buffers. `params` go to the next buffer index as inline constants.
fn dispatch(name: &'static str, bufs: &[Option<(MBuf, u64)>], params: &[u8], grid: (u64, u64, u64), tg: (u64, u64, u64), threadgroups: bool) {
    let p = pipe(name);
    let c = ctx();
    let cb = c.queue.0.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&p.0);
    for (i, b) in bufs.iter().enumerate() {
        match b {
            Some((buf, off)) => enc.set_buffer(i as u64, Some(&buf.0), *off),
            None => enc.set_buffer(i as u64, None, 0),
        }
    }
    enc.set_bytes(bufs.len() as u64, params.len() as u64, params.as_ptr() as *const c_void);
    let g = metal::MTLSize { width: grid.0, height: grid.1, depth: grid.2 };
    let t = metal::MTLSize { width: tg.0, height: tg.1, depth: tg.2 };
    if threadgroups {
        enc.dispatch_thread_groups(g, t);
    } else {
        enc.dispatch_threads(g, t);
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn opt_buf(ptr: *const c_void, extra: u64) -> Option<(MBuf, u64)> {
    if ptr.is_null() {
        return None;
    }
    let (b, off) = require(ptr);
    Some((b, off + extra))
}

fn ceil256(n: u64) -> u64 {
    (n + 255) / 256 * 256
}

// Scratch f32 vector for fused-norm GEMVs (grown monotonically, reused; execution
// is synchronous so there is no aliasing hazard).
static SCRATCH: Mutex<Option<MBuf>> = Mutex::new(None);

fn scratch_f32(n_elems: usize) -> (MBuf, u64) {
    let mut s = SCRATCH.lock().unwrap();
    let need = (n_elems.max(1) as u64) * 4;
    let ok = s.as_ref().map(|b| b.0.length() >= need).unwrap_or(false);
    if !ok {
        let b = ctx().device.0.new_buffer(need, metal::MTLResourceOptions::StorageModeShared);
        track(&b);
        *s = Some(MBuf(b));
    }
    (s.as_ref().unwrap().clone(), 0)
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// runtime: memory, streams, graphs, misc (all CPU-side; synchronous)

static METAL_UNAVAILABLE: &[u8] = b"metal backend error (op not yet ported or dispatch failed; see docs/MLX-PORT.md)\0";

pub unsafe fn dr_error_string(_e: i32) -> *const c_char {
    METAL_UNAVAILABLE.as_ptr() as *const c_char
}

pub unsafe fn dr_last_error() -> i32 {
    0
}

pub unsafe fn dr_malloc(p: *mut *mut c_void, n: usize) -> i32 {
    let n = n.max(16);
    let buf = ctx().device.0.new_buffer(n as u64, metal::MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() {
        return 1;
    }
    *p = track(&buf);
    0
}

pub unsafe fn dr_free(p: *mut c_void) -> i32 {
    if !p.is_null() {
        BUFS.lock().unwrap().remove(&(p as usize));
    }
    0
}

pub unsafe fn dr_host_alloc(p: *mut *mut c_void, n: usize) -> i32 {
    let layout = match std::alloc::Layout::from_size_align(n.max(1), 16) {
        Ok(l) => l,
        Err(_) => return 1,
    };
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        return 1;
    }
    HOST.lock().unwrap().insert(ptr as usize, layout);
    *p = ptr as *mut c_void;
    0
}

pub unsafe fn dr_host_free(p: *mut c_void) -> i32 {
    if !p.is_null() {
        if let Some(layout) = HOST.lock().unwrap().remove(&(p as usize)) {
            std::alloc::dealloc(p as *mut u8, layout);
        }
    }
    0
}

unsafe fn raw_copy(dst: *mut c_void, src: *const c_void, n: usize) {
    if n == 0 {
        return;
    }
    std::ptr::copy(src as *const u8, dst as *mut u8, n);
}

pub unsafe fn dr_h2d(d: *mut c_void, h: *const c_void, n: usize, st: Stream) -> i32 {
    launch(st, move || raw_copy(d, h, n))
}

pub unsafe fn dr_d2h(h: *mut c_void, d: *const c_void, n: usize, st: Stream) -> i32 {
    launch(st, move || raw_copy(h, d, n))
}

pub unsafe fn dr_d2d(d: *mut c_void, s: *const c_void, n: usize, st: Stream) -> i32 {
    launch(st, move || raw_copy(d, s, n))
}

pub unsafe fn dr_memset(d: *mut c_void, v: i32, n: usize, st: Stream) -> i32 {
    launch(st, move || {
        if n > 0 {
            std::slice::from_raw_parts_mut(d as *mut u8, n).fill(v as u8);
        }
    })
}

pub unsafe fn dr_stream_create(s: *mut Stream) -> i32 {
    let id = NEXT_STREAM.fetch_add(1, Ordering::Relaxed);
    QUEUES.lock().unwrap().insert(id, MQueue(ctx().device.0.new_command_queue()));
    *s = id as Stream;
    0
}

pub unsafe fn dr_stream_sync(_s: Stream) -> i32 {
    // Synchronous execution: every launch already waited.
    0
}

pub unsafe fn dr_mem_info(free_b: *mut usize, total_b: *mut usize) -> i32 {
    let total = ctx().device.0.recommended_max_working_set_size();
    let used = ctx().device.0.current_allocated_size();
    *total_b = total as usize;
    *free_b = total.saturating_sub(used) as usize;
    0
}

pub unsafe fn dr_graph_begin(st: Stream) -> i32 {
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(Capture { stream: st as usize, thread: std::thread::current().id(), ops: Vec::new() });
    });
    0
}

pub unsafe fn dr_graph_end(st: Stream, exec: *mut GraphExec) -> i32 {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        match c.take() {
            Some(cap) if cap.stream == st as usize && cap.thread == std::thread::current().id() => {
                let g = Box::new(GraphData { ops: cap.ops });
                *exec = Box::into_raw(g) as GraphExec;
                0
            }
            _ => 1,
        }
    })
}

pub unsafe fn dr_graph_launch(exec: GraphExec, _st: Stream) -> i32 {
    let g = &*(exec as *const GraphData);
    for op in &g.ops {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op())).is_err() {
            return 1;
        }
    }
    0
}

pub unsafe fn dr_graph_destroy(exec: GraphExec) -> i32 {
    if !exec.is_null() {
        drop(Box::from_raw(exec as *mut GraphData));
    }
    0
}

pub unsafe fn dr_cublas_create(h: *mut *mut c_void, _st: Stream) -> i32 {
    // No handle needed: GEMMs run on the CPU in v1 (see dr_gemm).
    *h = std::ptr::null_mut();
    0
}

/// CPU: C[m, n] = A[m, k] . W[n, k]^T + beta * C (bf16 inputs, f32 output).
///
/// Prefill only; decode uses the GEMV kernels. Slow on big shapes (a blocked or
/// MPS matmul is future work) but exact and fine for validation.
pub unsafe fn dr_gemm(_h: *mut c_void, a: *const Half, w: *const Half, c: *mut f32, m: i32, n: i32, k: i32, beta: f32) -> i32 {
    if m <= 0 || n <= 0 || k <= 0 {
        return 0;
    }
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let av = std::slice::from_raw_parts(a, m * k);
    let wv = std::slice::from_raw_parts(w, n * k);
    let cv = std::slice::from_raw_parts_mut(c, m * n);
    for r in 0..m {
        for cc in 0..n {
            let mut acc = 0.0f32;
            let arow = &av[r * k..(r + 1) * k];
            let wrow = &wv[cc * k..(cc + 1) * k];
            for j in 0..k {
                acc += bf16_to_f32(arow[j]) * bf16_to_f32(wrow[j]);
            }
            let o = &mut cv[r * n + cc];
            *o = acc + beta * *o;
        }
    }
    0
}

pub unsafe fn dr_gemm_f16(_h: *mut c_void, _a: *const Half, _w: *const Half, _c: *mut f32, _m: i32, _n: i32, _k: i32, _beta: f32) -> i32 {
    // Codec-only in the current engine; the codec is not ported yet.
    unported()
}

pub unsafe fn dr_attn_part_floats() -> i32 {
    // Must match the CUDA value: callers size the `part` buffer with it.
    16 * 258
}

pub unsafe fn dr_sample_init() -> i32 {
    0
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// not yet ported: same error convention as the stage-1 stubs

fn unported() -> i32 {
    1
}

pub unsafe fn dr_attn_decode(q: *const f32, kc: *const Half, vc: *const Half, gate: *const f32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32, _part: *mut f32, o: *mut f32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_attn_prefill(q: *const f32, kc: *const Half, vc: *const Half, gate: *const f32, gate_stride: i32, rows: i32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32, o: *mut f32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_codec_attn(_q: *const f32, _kc: *const f32, _vc: *const f32, _n: i32, _h: i32, _win: i32, _t0: i32, _ring: i32, _out: *mut Half, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_codec_rope(_qkv: *const f32, _n: i32, _h: i32, _inv: *const f32, _t0: i32, _ring: i32, _q: *mut f32, _kc: *mut f32, _vc: *mut f32, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_conv(input: *const f32, in_stride: i32, w: *const f32, state: *mut f32, out: *mut f32, n: i32, c: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_conv_out(_x: *const f32, _n: i32, _c: i32, _k: i32, _alpha: *const f32, _w: *const f32, _b: *const f32, _out: *mut f32, _st: Stream) -> i32 {
    unported()
}


pub unsafe fn dr_dwconv(_x: *const f32, _t: i32, _c: i32, _k: i32, _w: *const f32, _b: *const f32, _out: *mut f32, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_embed(rows_emb: *const Half, cand_table: *const Half, frames: *const i32, fstride: i32, cb_emb: *const Half, x: *mut f32, rows: i32, d: i32, sem0: i32, sem1: i32, eos: i32, cb: i32, nb: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_epilogue(_y: *const f32, _n: usize, _c: i32, _bias: *const f32, _scale: *const f32, _res: *const f32, _act: i32, _out: *mut f32, _o16: *mut Half, _st: Stream) -> i32 {
    unported()
}


pub unsafe fn dr_frame_end(next: *const i32, in_frame: *mut i32, ras: *mut i32, ras_len: i32, pos: *mut i32, step: *mut u64, hist: *mut i32, hist_len: *mut i32, hist_cap: i32, width: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_gdn(conv: *const f32, proj: *const f32, ps: i32, z_off: i32, b_off: i32, a_off: i32, a_log: *const f32, dt_bias: *const f32, norm_w: *const f32, state: *mut f32, o: *mut f32, n: i32, nk: i32, nv: i32, dk: i32, dv: i32, eps: f32, st: Stream) -> i32 {
    unported()
}


pub unsafe fn dr_gemv_q8(w: *const i8, s: *const Half, x: *const f32, res: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_gemv_q8_swiglu(w: *const i8, s: *const Half, x: *const f32, y: *mut f32, i: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_im2col(_x: *const f32, _n: i32, _c: i32, _k: i32, _d: i32, _s: i32, _alpha: *const f32, _out: *mut Half, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_layernorm(_x: *const f32, _rows: i32, _c: i32, _w: *const f32, _b: *const f32, _eps: f32, _out: *mut Half, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_moe_combine(x: *mut f32, y: *const f32, slot_of: *const i32, wts: *const f32, sgate: *const f32, shared: *const f32, rows: i32, topk: i32, h: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_moe_count(idx: *const i32, n: i32, counts: *mut i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_moe_decode(gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, sgu_q: *const i8, sgu_s: *const Half, sd_q: *const i8, sd_s: *const Half, x: *const f32, idx: *const i32, wts: *const f32, sgate: *const f32, res: *const f32, act: *mut f32, y: *mut f32, h: i32, i: i32, topk: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_moe_permute(idx: *const i32, rows: i32, topk: i32, offsets: *const i32, fill: *mut i32, slot_of: *mut i32, x: *const Half, xp: *mut Half, h: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_moe_prefill(gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, xp: *const Half, offsets: *const i32, counts: *const i32, act: *mut Half, y: *mut f32, e: i32, h: i32, i: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_moe_topk(logits: *const f32, rows: i32, e: i32, topk: i32, stride: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_qk_norm_rope(qkv: *const f32, rs: i32, rows: i32, h: i32, hkv: i32, hd: i32, qn: *const f32, kn: *const f32, inv_freq: *const f32, rot: i32, interleaved: i32, pos_base: *const i32, pos_off: i32, q_out: *mut f32, kc: *mut Half, vc: *mut Half, eps: f32, st: Stream) -> i32 {
    unported()
}


pub unsafe fn dr_rmsnorm_plain(_x: *const f32, _rows: i32, _c: i32, _w: *const f32, _eps: f32, _out: *mut f32, _o16: *mut Half, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_router_topk(w: *const Half, x: *const f32, nw: *const f32, eps: f32, logits: *mut f32, e: i32, topk: i32, k: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, counter: *mut u32, _gq: *const c_char, _gs: *const c_char, _dq: *const c_char, _ds: *const c_char, _gqb: usize, _gsb: usize, _dqb: usize, _dsb: usize, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_rvq_sum(_codes: *const i32, _frames: i32, _ncb: i32, _table: *const f32, _out: *mut f32, _d: i32, _sem: i32, _res: i32, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_sample_head(logits: *const f32, n: i32, prm: *const c_void, stream: i32, step: *const u64, frame: *mut i32, col: i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_sample_main(logits: *const f32, n: i32, prm: *const c_void, ras: *const i32, ras_len: i32, step: *const u64, eos: i32, sem0: i32, cb: i32, frame: *mut i32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_swiglu16(_h: *const f32, _y: *mut Half, _rows: i32, _i: i32, _st: Stream) -> i32 {
    unported()
}


pub unsafe fn dr_tconv_gather(_x: *const f32, _n: i32, _c: i32, _two: i32, _alpha: *const f32, _out: *mut Half, _st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_tower_in(h: *const f32, frames: *const i32, fstride: i32, code_col: i32, in_frames: *const i32, in_stride: i32, tok_emb: *const Half, cb_emb: *const Half, enorm: *const f32, hnorm: *const f32, pre_norm: *const f32, y: *mut f32, yb: *mut Half, rows: i32, d: i32, sem0: i32, sem1: i32, cb: i32, ncb: i32, nb: i32, eps: f32, st: Stream) -> i32 {
    unported()
}

pub unsafe fn dr_vq_encode(_r: *mut f32, _rows: i32, _d: i32, _in_w: *const f32, _in_b: *const f32, _cbn: *const f32, _n: i32, _table: *const f32, _codes: *mut i32, _cs: i32, _st: Stream) -> i32 {
    unported()
}
// ---------------------------------------------------------------------------
// MSL-backed ops

/// MSL: rmsnorm.
///
/// y[r] = x[r] * inv * w (f32 and/or bf16). `w` is required by every caller.
pub unsafe fn dr_rmsnorm(x: *const f32, w: *const f32, y: *mut f32, yb: *mut Half, rows: i32, d: i32, eps: f32, st: Stream) -> i32 {
    launch(st, move || {
        let xb = require(x as *const c_void);
        let wb = require(w as *const c_void);
        let prm = NormP { rows: rows as u32, d: d as u32, eps };
        dispatch(
            "rmsnorm",
            &[Some(xb), Some(wb), opt_buf(y as *const c_void, 0), opt_buf(yb as *const c_void, 0)],
            pb(&prm),
            (rows as u64, 1, 1),
            (256, 1, 1),
            true,
        );
    })
}

/// MSL: y[n] = W[n] . norm(x) (+ fused RMSNorm when `nw` is set).
pub unsafe fn dr_gemv_bf16(w: *const Half, x: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    launch(st, move || {
        let wb = require(w as *const c_void);
        let yb = require(y as *const c_void);
        let xv: (MBuf, u64) = if nw.is_null() {
            require(x as *const c_void)
        } else {
            let (sb, soff) = scratch_f32(k as usize);
            let nwb = require(nw as *const c_void);
            let xb = require(x as *const c_void);
            let prm = NormP { rows: 1, d: k as u32, eps };
            dispatch("rmsnorm", &[Some(xb), Some(nwb), Some((sb.clone(), soff)), None], pb(&prm), (1, 1, 1), (256, 1, 1), true);
            (sb, soff)
        };
        let prm = GemvP { n: n as u32, k: k as u32 };
        dispatch("gemv_bf16", &[Some(wb), Some(xv), Some(yb)], pb(&prm), (n as u64, 1, 1), (256, 1, 1), true);
    })
}

/// MSL: dequantise int8 group-128 weights to bf16.
pub unsafe fn dr_dequant_q8(w: *const i8, s: *const Half, out: *mut Half, n: i32, k: i32, st: Stream) -> i32 {
    launch(st, move || {
        let prm = DqP { n: n as u32, k: k as u32 };
        let total = n as u64 * k as u64;
        dispatch(
            "dequant_q8",
            &[opt_buf(w as *const c_void, 0), opt_buf(s as *const c_void, 0), opt_buf(out as *const c_void, 0)],
            pb(&prm),
            (ceil256(total), 1, 1),
            (256, 1, 1),
            false,
        );
    })
}

/// MSL: f32 -> bf16 (round-to-nearest-even).
pub unsafe fn dr_f32_to_bf16(x: *const f32, y: *mut Half, n: usize, st: Stream) -> i32 {
    launch(st, move || {
        let nu = n as u32;
        dispatch(
            "f32_to_bf16",
            &[opt_buf(x as *const c_void, 0), opt_buf(y as *const c_void, 0)],
            pb(&nu),
            (ceil256(n as u64), 1, 1),
            (256, 1, 1),
            false,
        );
    })
}

/// MSL: y[r, j] = silu(h[r, j]) * h[r, I + j] in bf16.
pub unsafe fn dr_swiglu_rows(h: *const f32, y: *mut Half, rows: i32, i: i32, st: Stream) -> i32 {
    launch(st, move || {
        let dims = [rows as u32, i as u32];
        let total = rows as u64 * i as u64;
        dispatch(
            "swiglu_rows",
            &[opt_buf(h as *const c_void, 0), opt_buf(y as *const c_void, 0)],
            pb(&dims),
            (ceil256(total), 1, 1),
            (256, 1, 1),
            false,
        );
    })
}

// ---------------------------------------------------------------------------
