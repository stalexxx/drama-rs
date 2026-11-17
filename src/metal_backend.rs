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

/// CPU: C[m, n] = A[m, k] . W[n, k]^T + beta * C (f16 inputs, f32 output).
pub unsafe fn dr_gemm_f16(_h: *mut c_void, a: *const Half, w: *const Half, c: *mut f32, m: i32, n: i32, k: i32, beta: f32) -> i32 {
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
                acc += f16_to_f32(arow[j]) * f16_to_f32(wrow[j]);
            }
            let o = &mut cv[r * n + cc];
            *o = acc + beta * *o;
        }
    }
    0
}

pub unsafe fn dr_attn_part_floats() -> i32 {
    // Must match the CUDA value: callers size the `part` buffer with it.
    16 * 258
}

pub unsafe fn dr_sample_init() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// CPU reference: MoE routing helpers

/// CPU: softmax over E logits, top-k (descending, lower-index tie-break),
/// renormalised weights; shared-expert gate logit at column E.
pub unsafe fn dr_moe_topk(logits: *const f32, rows: i32, e: i32, topk: i32, stride: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, e, topk, stride) = (rows as usize, e as usize, topk as usize, stride as usize);
        for r in 0..rows {
            let l = &f32s(logits, (r + 1) * stride)[r * stride..];
            let m = l[..e].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
            let ex: Vec<f32> = order.iter().map(|&i| (l[i] - m).exp()).collect();
            let z: f32 = ex.iter().sum();
            let zt: f32 = ex[..topk].iter().sum();
            for k in 0..topk {
                *idx.add(r * topk + k) = order[k] as i32;
                *wts.add(r * topk + k) = ex[k] / zt;
            }
            let _ = z;
            *sgate.add(r) = l[e];
        }
    })
}

/// CPU: counts[e] += #{i : idx[i] == e}.
pub unsafe fn dr_moe_count(idx: *const i32, n: i32, counts: *mut i32, st: Stream) -> i32 {
    launch(st, move || {
        for i in 0..n as usize {
            let e = *idx.add(i) as usize;
            *counts.add(e) += 1;
        }
    })
}

/// CPU: scatter bf16 rows of x into expert-sorted order.
pub unsafe fn dr_moe_permute(idx: *const i32, rows: i32, topk: i32, offsets: *const i32, fill: *mut i32, slot_of: *mut i32, x: *const Half, xp: *mut Half, h: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, topk, h) = (rows as usize, topk as usize, h as usize);
        for i in 0..rows * topk {
            let e = *idx.add(i) as usize;
            let slot = (*offsets.add(e) + *fill.add(e)) as usize;
            *fill.add(e) += 1;
            *slot_of.add(i) = slot as i32;
            let r = i / topk;
            std::ptr::copy_nonoverlapping(x.add(r * h), xp.add(slot * h), h);
        }
    })
}

/// CPU: x[r] += sigmoid(sgate[r]) * shared[r] + sum_k w[r,k] * Y[slot_of[r,k]].
pub unsafe fn dr_moe_combine(x: *mut f32, y: *const f32, slot_of: *const i32, wts: *const f32, sgate: *const f32, shared: *const f32, rows: i32, topk: i32, h: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, topk, h) = (rows as usize, topk as usize, h as usize);
        for r in 0..rows {
            let sg = sigmoid(*sgate.add(r));
            for cc in 0..h {
                let mut acc = sg * *shared.add(r * h + cc);
                for k in 0..topk {
                    let slot = *slot_of.add(r * topk + k) as usize;
                    acc += *wts.add(r * topk + k) * *y.add(slot * h + cc);
                }
                *x.add(r * h + cc) += acc;
            }
        }
    })
}

// ---------------------------------------------------------------------------
// CPU reference: sampling. Distribution matches kernels/drama.cu (top-k / top-p with
// the sglang-omni rule); the Philox stream below is distribution-equivalent to
// curand but NOT bit-identical, so sampled tokens may differ from CUDA runs.
// (Validation compares logits, never samples.)

#[repr(C)]
#[derive(Clone, Copy)]
struct SampleParams {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    pad: i32,
    seed: u64,
}

fn philox4x32(seed: u64, stream: u64, step: u64) -> [u32; 4] {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let mut k0 = seed as u32;
    let mut k1 = (seed >> 32) as u32;
    let mut c = [step as u32, (step >> 32) as u32, stream as u32, 0u32];
    for _ in 0..10 {
        let p0 = c[0] as u64 * M0 as u64;
        let p1 = c[2] as u64 * M1 as u64;
        c = [(p1 >> 32) as u32 ^ c[1] ^ k0, p1 as u32, (p0 >> 32) as u32 ^ c[3] ^ k1, p0 as u32];
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    c
}

/// Uniform in (0, 1], like curand_uniform.
fn philox_uniform(seed: u64, stream: u64, step: u64) -> f32 {
    let x = philox4x32(seed, stream, step)[0];
    1.0 - x as f32 / 4294967296.0
}

/// Sample a logit position: full descending sort (index tie-break), sglang-omni
/// top-k/top-p rule over the full distribution.
fn sample_logits(logits: &[f32], temp: f32, top_k: i32, top_p: f32, u: f32) -> usize {
    let n = logits.len();
    let inv_t = 1.0 / temp.max(1e-5);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let k0 = logits[order[0]];
    let mut cum = 0.0f32;
    let mut cums = vec![0.0f32; n];
    for (i, &idx) in order.iter().enumerate() {
        cum += ((logits[idx] - k0) * inv_t).exp();
        cums[i] = cum;
    }
    let z = cum;
    let mut kept = cums.iter().filter(|&&c| c <= top_p * z).count();
    kept = kept.clamp(1, top_k.max(1) as usize);
    let target = u * cums[kept - 1];
    let rank = (0..kept).find(|&i| cums[i] >= target).unwrap_or(kept - 1);
    order[rank]
}

pub unsafe fn dr_sample_main(logits: *const f32, n: i32, prm: *const c_void, ras: *const i32, ras_len: i32, step: *const u64, eos: i32, sem0: i32, cb: i32, frame: *mut i32, st: Stream) -> i32 {
    launch(st, move || {
        let p = *(prm as *const SampleParams);
        let l = f32s(logits, n as usize);
        let s = *step;
        let u1 = philox_uniform(p.seed, 0, s);
        let u2 = philox_uniform(p.seed, 1, s);
        let r1 = sample_logits(l, p.temperature, p.top_k, p.top_p, u1);
        let r2 = sample_logits(l, 1.0, 30, 0.95, u2);
        let t1 = if r1 == 0 { eos } else { sem0 + r1 as i32 - 1 };
        let t2 = if r2 == 0 { eos } else { sem0 + r2 as i32 - 1 };
        let mut repeat = false;
        for i in 0..ras_len as usize {
            repeat |= *ras.add(i) == t1;
        }
        let main = if repeat && r1 != 0 { t2 } else { t1 };
        *frame.add(0) = main;
        *frame.add(1) = (main - sem0).clamp(0, cb - 1);
    })
}

pub unsafe fn dr_sample_head(logits: *const f32, n: i32, prm: *const c_void, stream: i32, step: *const u64, frame: *mut i32, col: i32, st: Stream) -> i32 {
    launch(st, move || {
        let p = *(prm as *const SampleParams);
        let l = f32s(logits, n as usize);
        let u = philox_uniform(p.seed, stream as u64, *step);
        let r = sample_logits(l, p.temperature, p.top_k, p.top_p, u);
        *frame.add(col as usize) = r as i32;
    })
}

pub unsafe fn dr_frame_end(next: *const i32, in_frame: *mut i32, ras: *mut i32, ras_len: i32, pos: *mut i32, step: *mut u64, hist: *mut i32, hist_len: *mut i32, hist_cap: i32, width: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (ras_len, width) = (ras_len as usize, width as usize);
        for i in 0..width {
            *in_frame.add(i) = *next.add(i);
        }
        for i in 0..ras_len.saturating_sub(1) {
            *ras.add(i) = *ras.add(i + 1);
        }
        if ras_len > 0 {
            *ras.add(ras_len - 1) = *next.add(0);
        }
        *pos.add(0) += 1;
        *step.add(0) += 1;
        let h = *hist_len.add(0);
        if h < hist_cap {
            for i in 0..width {
                *hist.add(h as usize * width + i) = *next.add(i);
            }
            *hist_len.add(0) = h + 1;
        }
    })
}

// ---------------------------------------------------------------------------
// (every dr_* above is implemented; nothing routes to unported() anymore)

// ---------------------------------------------------------------------------
// CPU reference: quantised GEMV (decode)

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// RMS-normalise (with optional fused weight); plain copy when `nw` is None.
fn norm_fused(x: &[f32], nw: Option<&[f32]>, eps: f32) -> Vec<f32> {
    match nw {
        None => x.to_vec(),
        Some(w) => {
            let inv = rms_inv(x, eps);
            x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
        }
    }
}

/// int8 row dot, group-128 f16 scales.
fn dot_q8_row(q: &[i8], s: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let qs = &q[row * k..(row + 1) * k];
    let ss = &s[row * (k / 128)..];
    let mut acc = 0.0f32;
    for (j, &qv) in qs.iter().enumerate() {
        acc += qv as f32 * f16_to_f32(ss[j / 128]) * x[j];
    }
    acc
}

/// int4 asymmetric row dot, group-64 (f16 scale + f16 min per group).
fn dot_q4_row(q: &[u8], sm: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let qb = &q[row * (k / 2)..(row + 1) * (k / 2)];
    let smb = &sm[row * (k / 64) * 2..];
    let mut acc = 0.0f32;
    for (j, &xv) in x.iter().enumerate().take(k) {
        let g = j / 64;
        let qq = (qb[j / 2] >> ((j & 1) * 4)) & 15;
        let w = qq as f32 * f16_to_f32(smb[g * 2]) + f16_to_f32(smb[g * 2 + 1]);
        acc += w * xv;
    }
    acc
}

/// bf16 row dot.
fn dot_bf16_row(w: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let wr = &w[row * k..(row + 1) * k];
    wr.iter().zip(x.iter()).map(|(a, b)| bf16_to_f32(*a) * b).sum()
}

/// CPU: y[n] = W[n] . norm(x) (+ res[n]); fused RMSNorm when `nw` is set.
pub unsafe fn dr_gemv_q8(w: *const i8, s: *const Half, x: *const f32, res: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    launch(st, move || {
        let (n, k) = (n as usize, k as usize);
        let q = std::slice::from_raw_parts(w, n * k);
        let ss = std::slice::from_raw_parts(s, n * (k / 128));
        let xv = f32s(x, k);
        let xn = norm_fused(xv, if nw.is_null() { None } else { Some(f32s(nw, k)) }, eps);
        for row in 0..n {
            let mut v = dot_q8_row(q, ss, k, row, &xn);
            if !res.is_null() {
                v += *res.add(row);
            }
            *y.add(row) = v;
        }
    })
}

/// CPU: y[j] = silu(W[j] . norm(x)) * (W[I + j] . norm(x)).
pub unsafe fn dr_gemv_q8_swiglu(w: *const i8, s: *const Half, x: *const f32, y: *mut f32, i: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    launch(st, move || {
        let (i, k) = (i as usize, k as usize);
        let q = std::slice::from_raw_parts(w, 2 * i * k);
        let ss = std::slice::from_raw_parts(s, 2 * i * (k / 128));
        let xn = norm_fused(f32s(x, k), if nw.is_null() { None } else { Some(f32s(nw, k)) }, eps);
        for j in 0..i {
            let g = dot_q8_row(q, ss, k, j, &xn);
            let u = dot_q8_row(q, ss, k, i + j, &xn);
            *y.add(j) = silu(g) * u;
        }
    })
}

/// CPU: router logits (fused norm) + softmax/top-k; warms nothing, resets counter.
pub unsafe fn dr_router_topk(w: *const Half, x: *const f32, nw: *const f32, eps: f32, logits: *mut f32, e: i32, topk: i32, k: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, counter: *mut u32, _gq: *const c_char, _gs: *const c_char, _dq: *const c_char, _ds: *const c_char, _gqb: usize, _gsb: usize, _dqb: usize, _dsb: usize, st: Stream) -> i32 {
    launch(st, move || {
        let (e, topk, k) = (e as usize, topk as usize, k as usize);
        let wv = std::slice::from_raw_parts(w, (e + 1) * k);
        let xn = norm_fused(f32s(x, k), if nw.is_null() { None } else { Some(f32s(nw, k)) }, eps);
        for row in 0..=e {
            *logits.add(row) = dot_bf16_row(wv, k, row, &xn);
        }
        let l = std::slice::from_raw_parts(logits, e + 1);
        let m = l[..e].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let ex: Vec<f32> = order.iter().map(|&i| (l[i] - m).exp()).collect();
        let zt: f32 = ex[..topk].iter().sum();
        for kk in 0..topk {
            *idx.add(kk) = order[kk] as i32;
            *wts.add(kk) = ex[kk] / zt;
        }
        *sgate.add(0) = l[e];
        *counter.add(0) = 0;
    })
}

/// CPU: decode MoE — routed q4 experts + shared q8 expert.
/// Requires H == 2048, I == 512, topk == 8 (same limits as CUDA).
#[allow(clippy::too_many_arguments)]
pub unsafe fn dr_moe_decode(gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, sgu_q: *const i8, sgu_s: *const Half, sd_q: *const i8, sd_s: *const Half, x: *const f32, idx: *const i32, wts: *const f32, sgate: *const f32, res: *const f32, act: *mut f32, y: *mut f32, h: i32, i: i32, topk: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    if h != 2048 || i != 512 || topk != 8 {
        return 1;
    }
    launch(st, move || {
        let (h, mi, topk) = (h as usize, i as usize, topk as usize);
        let k = h; // K = H = 2048 for gate_up rows
        let gq = std::slice::from_raw_parts(gu_q, 8 * 2 * mi * (k / 2));
        let gs = std::slice::from_raw_parts(gu_sm, 8 * 2 * mi * (k / 64) * 2);
        let dq = std::slice::from_raw_parts(d_q, 8 * h * (mi / 2));
        let ds = std::slice::from_raw_parts(d_sm, 8 * h * (mi / 64) * 2);
        let sgq = std::slice::from_raw_parts(sgu_q, 2 * mi * k);
        let sgs = std::slice::from_raw_parts(sgu_s, 2 * mi * (k / 128));
        let sdq = std::slice::from_raw_parts(sd_q, h * mi);
        let sds = std::slice::from_raw_parts(sd_s, h * (mi / 128));
        let xn = norm_fused(f32s(x, k), if nw.is_null() { None } else { Some(f32s(nw, k)) }, eps);
        // gate_up: routed experts then shared (slot topk)
        for kk in 0..topk {
            let e = *idx.add(kk) as usize;
            for j in 0..mi {
                let g = dot_q4_row(gq, gs, k, e * 2 * mi + j, &xn);
                let u = dot_q4_row(gq, gs, k, e * 2 * mi + mi + j, &xn);
                *act.add(kk * mi + j) = silu(g) * u;
            }
        }
        for j in 0..mi {
            let g = dot_q8_row(sgq, sgs, k, j, &xn);
            let u = dot_q8_row(sgq, sgs, k, mi + j, &xn);
            *act.add(topk * mi + j) = silu(g) * u;
        }
        // down
        let sg = sigmoid(*sgate.add(0));
        let wv: Vec<f32> = (0..topk).map(|kk| *wts.add(kk)).collect();
        for n in 0..h {
            let mut acc = 0.0f32;
            for kk in 0..topk {
                let e = *idx.add(kk) as usize;
                let mut qx = 0.0f32;
                for j in 0..mi {
                    let aval = *act.add(kk * mi + j);
                    let qb = dq[(e * h + n) * (mi / 2) + j / 2];
                    let qq = ((qb >> ((j & 1) * 4)) & 15) as f32;
                    let g = j / 64;
                    let smb = &ds[(e * h + n) * (mi / 64) * 2..];
                    qx += (qq * f16_to_f32(smb[g * 2]) + f16_to_f32(smb[g * 2 + 1])) * aval;
                }
                acc += wv[kk] * qx;
            }
            let sh = dot_q8_row(sdq, sds, mi, n, std::slice::from_raw_parts(act.add(topk * mi), mi));
            *y.add(n) = *res.add(n) + sg * sh + acc;
        }
    })
}

/// CPU: prefill MoE over expert-sorted rows. Requires H == 2048, I == 512.
pub unsafe fn dr_moe_prefill(gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, xp: *const Half, offsets: *const i32, counts: *const i32, act: *mut Half, y: *mut f32, e: i32, h: i32, i: i32, st: Stream) -> i32 {
    if h != 2048 || i != 512 {
        return 1;
    }
    launch(st, move || {
        let (e, h, mi) = (e as usize, h as usize, i as usize);
        let k = h; // K = 2048
        let gq = std::slice::from_raw_parts(gu_q, e * 2 * mi * (k / 2));
        let gs = std::slice::from_raw_parts(gu_sm, e * 2 * mi * (k / 64) * 2);
        let dq = std::slice::from_raw_parts(d_q, e * h * (mi / 2));
        let ds = std::slice::from_raw_parts(d_sm, e * h * (mi / 64) * 2);
        let xpv_len = (0..e).map(|ex| *counts.add(ex) as usize).sum::<usize>().max(1) * k;
        let xpv = std::slice::from_raw_parts(xp, xpv_len);
        for ex in 0..e {
            let cnt = *counts.add(ex) as usize;
            let off = *offsets.add(ex) as usize;
            for t in 0..cnt {
                let slot = off + t;
                let xr: Vec<f32> = xpv[slot * k..(slot + 1) * k].iter().map(|&v| bf16_to_f32(v)).collect();
                for j in 0..mi {
                    let g = dot_q4_row(gq, gs, k, ex * 2 * mi + j, &xr);
                    let u = dot_q4_row(gq, gs, k, ex * 2 * mi + mi + j, &xr);
                    *act.add(slot * mi + j) = f32_to_bf16_rn(silu(g) * u);
                }
            }
            for t in 0..cnt {
                let slot = off + t;
                let ar: Vec<f32> = (0..mi).map(|j| bf16_to_f32(*act.add(slot * mi + j))).collect();
                for n in 0..h {
                    *y.add(slot * h + n) = dot_q4_row(dq, ds, mi, ex * h + n, &ar);
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// CPU reference: embeddings

/// CPU: backbone input rows (prefill gather or decode candidate lookup + codes).
pub unsafe fn dr_embed(rows_emb: *const Half, cand_table: *const Half, frames: *const i32, fstride: i32, cb_emb: *const Half, x: *mut f32, rows: i32, d: i32, sem0: i32, sem1: i32, eos: i32, cb: i32, nb: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, d, fstride) = (rows as usize, d as usize, fstride as usize);
        for r in 0..rows {
            // Decode passes fstride = 0 for its single row; size the row view by use.
            let flen = fstride.max(nb as usize + 2);
            let f = std::slice::from_raw_parts(frames.add(r * fstride), flen);
            let main = f[0];
            let is_sem = (sem0..=sem1).contains(&main);
            let xr = std::slice::from_raw_parts_mut(x.add(r * d), d);
            if !rows_emb.is_null() {
                let er = std::slice::from_raw_parts(rows_emb.add(r * d), d);
                for (o, &v) in xr.iter_mut().zip(er.iter()) {
                    *o = bf16_to_f32(v);
                }
            } else {
                let ci = if main == eos { 0 } else { clampi(main - sem0, 0, cb - 1) + 1 };
                let er = std::slice::from_raw_parts(cand_table.add(ci as usize * d), d);
                for (o, &v) in xr.iter_mut().zip(er.iter()) {
                    *o = bf16_to_f32(v);
                }
            }
            if is_sem {
                let scale = (1.0 + nb as f32).sqrt().recip();
                for c in 1..=nb as usize {
                    let code = clampi(f[1 + c], 0, cb - 1) as usize;
                    let cbr = std::slice::from_raw_parts(cb_emb.add((c * cb as usize + code) * d), d);
                    for (o, &v) in xr.iter_mut().zip(cbr.iter()) {
                        *o += bf16_to_f32(v);
                    }
                }
                for o in xr.iter_mut() {
                    *o *= scale;
                }
            }
        }
    })
}

/// CPU: tower layer input — [rms(e) | rms(h)] with label/codebook gathers.
pub unsafe fn dr_tower_in(h: *const f32, frames: *const i32, fstride: i32, code_col: i32, in_frames: *const i32, in_stride: i32, tok_emb: *const Half, cb_emb: *const Half, enorm: *const f32, hnorm: *const f32, pre_norm: *const f32, y: *mut f32, yb: *mut Half, rows: i32, d: i32, sem0: i32, sem1: i32, cb: i32, ncb: i32, nb: i32, eps: f32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, d) = (rows as usize, d as usize);
        let (fstride, code_col) = (fstride as usize, code_col as usize);
        for r in 0..rows {
            // Decode passes fstride = 0 for its single row; size the row view by use.
            let flen = fstride.max(code_col + 1);
            let f = std::slice::from_raw_parts(frames.add(r * fstride), flen);
            let lab_sem = (sem0..=sem1).contains(&f[0]);
            let code = clampi(f[code_col], 0, cb - 1) as usize;
            let mut ev = vec![0.0f32; d];
            if lab_sem {
                let ter = std::slice::from_raw_parts(tok_emb.add(code * d), d);
                for (o, &v) in ev.iter_mut().zip(ter.iter()) {
                    *o = bf16_to_f32(v);
                }
            }
            let mut hv: Vec<f32> = f32s(h, (r + 1) * d)[r * d..].to_vec();
            if !in_frames.is_null() {
                let ilen = (in_stride as usize).max(ncb as usize + 1);
                let fi = std::slice::from_raw_parts(in_frames.add(r * in_stride as usize), ilen);
                if (sem0..=sem1).contains(&fi[0]) {
                    for c in (1 + nb as usize)..ncb as usize {
                        let cc = clampi(fi[1 + c], 0, cb - 1) as usize;
                        let cbr = std::slice::from_raw_parts(cb_emb.add((c * cb as usize + cc) * d), d);
                        for (o, &v) in ev.iter_mut().zip(cbr.iter()) {
                            *o += bf16_to_f32(v);
                        }
                    }
                }
            }
            if !pre_norm.is_null() {
                let inv = rms_inv(&hv, eps);
                let pn = f32s(pre_norm, d);
                for (o, &w) in hv.iter_mut().zip(pn.iter()) {
                    *o = *o * inv * w;
                }
            }
            let ie = rms_inv(&ev, eps);
            let ih = rms_inv(&hv, eps);
            let en = f32s(enorm, d);
            let hn = f32s(hnorm, d);
            for i in 0..d {
                let a = ev[i] * ie * en[i];
                let b = hv[i] * ih * hn[i];
                if !y.is_null() {
                    *y.add(r * 2 * d + i) = a;
                    *y.add(r * 2 * d + d + i) = b;
                }
                if !yb.is_null() {
                    *yb.add(r * 2 * d + i) = f32_to_bf16_rn(a);
                    *yb.add(r * 2 * d + d + i) = f32_to_bf16_rn(b);
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// CPU reference: full attention

/// CPU: per-(row, head) RMSNorm, rotary embedding, KV-cache write.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dr_qk_norm_rope(qkv: *const f32, rs: i32, rows: i32, h: i32, hkv: i32, hd: i32, qn: *const f32, kn: *const f32, inv_freq: *const f32, rot: i32, interleaved: i32, pos_base: *const i32, pos_off: i32, q_out: *mut f32, kc: *mut Half, vc: *mut Half, eps: f32, st: Stream) -> i32 {
    launch(st, move || {
        let (rs, rows, h, hkv, hd, rot) = (rs as usize, rows as usize, h as usize, hkv as usize, hd as usize, rot as usize);
        for r in 0..rows {
            let pos = (*pos_base.add(0) + pos_off) as usize + r;
            let row = f32s(qkv, (r + 1) * rs);
            let row = &row[r * rs..];
            for hh in 0..h + hkv {
                let is_q = hh < h;
                let src_off = if is_q { hh * hd } else { 2 * h * hd + (hh - h) * hd };
                let src = &row[src_off..src_off + hd];
                let nw = f32s(if is_q { qn } else { kn }, hd);
                let inv = rms_inv(src, eps);
                let mut b = vec![0.0f32; hd];
                for t in 0..hd {
                    b[t] = src[t] * inv * nw[t];
                }
                // Rotary on the first `rot` dims.
                let mut o = b.clone();
                if interleaved != 0 {
                    for i in 0..rot / 2 {
                        let f = *inv_freq.add(i);
                        let (sn, cs) = (pos as f32 * f).sin_cos();
                        o[2 * i] = b[2 * i] * cs - b[2 * i + 1] * sn;
                        o[2 * i + 1] = b[2 * i + 1] * cs + b[2 * i] * sn;
                    }
                } else {
                    let half = rot / 2;
                    for i in 0..half {
                        let f = *inv_freq.add(i);
                        let (sn, cs) = (pos as f32 * f).sin_cos();
                        o[i] = b[i] * cs - b[i + half] * sn;
                        o[i + half] = b[i + half] * cs + b[i] * sn;
                    }
                }
                if is_q {
                    let dst = std::slice::from_raw_parts_mut(q_out.add(r * h * hd + hh * hd), hd);
                    dst.copy_from_slice(&o);
                } else {
                    let kh = hh - h;
                    let at = (pos * hkv + kh) * hd;
                    for t in 0..hd {
                        *kc.add(at + t) = f32_to_bf16_rn(o[t]);
                        *vc.add(at + t) = f32_to_bf16_rn(row[2 * h * hd + hkv * hd + kh * hd + t]);
                    }
                }
            }
        }
    })
}

/// Shared softmax attention over cached keys/values: out = softmax(qK/sqrt(hd))V / L,
/// multiplied by sigmoid(gate). `nrows` query rows starting at `pos0`.
unsafe fn attn_rows(q: &[f32], kc: &[u16], vc: &[u16], gate: &[f32], gate_stride: usize, nrows: usize, h: usize, hkv: usize, hd: usize, pos0: usize, out: &mut [f32]) {
    let scale = (hd as f32).sqrt().recip();
    let g = h / hkv;
    for r in 0..nrows {
        let nkeys = pos0 + r + 1;
        for hh in 0..h {
            let kh = hh / g;
            let qr = &q[r * h * hd + hh * hd..(r + 1) * h * hd];
            let qr = &qr[..hd];
            // Scores then online softmax.
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; hd];
            for j in 0..nkeys {
                let base = (j * hkv + kh) * hd;
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += qr[d] * bf16_to_f32(kc[base + d]);
                }
                s *= scale;
                let mn = m.max(s);
                let corr = (m - mn).exp();
                let p = (s - mn).exp();
                l = l * corr + p;
                for d in 0..hd {
                    acc[d] = acc[d] * corr + p * bf16_to_f32(vc[base + d]);
                }
                m = mn;
            }
            for d in 0..hd {
                let gv = gate[r * gate_stride + hh * hd + d];
                out[r * h * hd + hh * hd + d] = acc[d] / l * sigmoid(gv);
            }
        }
    }
}

/// CPU: prefill attention over keys 0..pos+r.
pub unsafe fn dr_attn_prefill(q: *const f32, kc: *const Half, vc: *const Half, gate: *const f32, gate_stride: i32, rows: i32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32, o: *mut f32, st: Stream) -> i32 {
    if hd != 256 {
        return 1;
    }
    launch(st, move || {
        let (rows, h, hkv, hd) = (rows as usize, h as usize, hkv as usize, hd as usize);
        let pos0 = (*pos_base.add(0) + pos_off) as usize;
        // KV caches are append-only and sized max_len rows; the prompt fits by construction.
        let max_keys = pos0 + rows;
        let kcv = std::slice::from_raw_parts(kc, max_keys * hkv * hd);
        let vcv = std::slice::from_raw_parts(vc, max_keys * hkv * hd);
        let qv = f32s(q, rows * h * hd);
        let gv = f32s(gate, rows * gate_stride as usize);
        let ov = std::slice::from_raw_parts_mut(o, rows * h * hd);
        attn_rows(qv, kcv, vcv, gv, gate_stride as usize, rows, h, hkv, hd, pos0, ov);
    })
}

/// CPU: single-row decode attention (the `part` buffer is unused on this path).
pub unsafe fn dr_attn_decode(q: *const f32, kc: *const Half, vc: *const Half, gate: *const f32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32, _part: *mut f32, o: *mut f32, st: Stream) -> i32 {
    if hd != 256 {
        return 1;
    }
    launch(st, move || {
        let (h, hkv, hd) = (h as usize, hkv as usize, hd as usize);
        let pos0 = (*pos_base.add(0) + pos_off) as usize;
        let max_keys = pos0 + 1;
        let kcv = std::slice::from_raw_parts(kc, max_keys * hkv * hd);
        let vcv = std::slice::from_raw_parts(vc, max_keys * hkv * hd);
        let qv = f32s(q, h * hd);
        let gv = f32s(gate, h * hd);
        let ov = std::slice::from_raw_parts_mut(o, h * hd);
        attn_rows(qv, kcv, vcv, gv, h * hd, 1, h, hkv, hd, pos0, ov);
    })
}

// ---------------------------------------------------------------------------
// CPU reference: linear attention (depthwise conv + gated DeltaNet)

/// CPU: causal depthwise conv (kernel 4) + SiLU; state keeps 3 inputs/channel.
pub unsafe fn dr_conv(input: *const f32, in_stride: i32, w: *const f32, state: *mut f32, out: *mut f32, n: i32, c: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (n, c, ins) = (n as usize, c as usize, in_stride as usize);
        for ch in 0..c {
            let mut h = [*state.add(ch * 3), *state.add(ch * 3 + 1), *state.add(ch * 3 + 2)];
            let wv = [f32_at(w, ch * 4), f32_at(w, ch * 4 + 1), f32_at(w, ch * 4 + 2), f32_at(w, ch * 4 + 3)];
            for t in 0..n {
                let xt = *input.add(t * ins + ch);
                *out.add(t * c + ch) = silu(wv[0] * h[0] + wv[1] * h[1] + wv[2] * h[2] + wv[3] * xt);
                h[0] = h[1];
                h[1] = h[2];
                h[2] = xt;
            }
            *state.add(ch * 3) = h[0];
            *state.add(ch * 3 + 1) = h[1];
            *state.add(ch * 3 + 2) = h[2];
        }
    })
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (x.exp() + 1.0).ln() }
}

/// CPU: gated DeltaNet recurrence with gated RMSNorm output. dk == dv == 128 only.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dr_gdn(conv: *const f32, proj: *const f32, ps: i32, z_off: i32, b_off: i32, a_off: i32, a_log: *const f32, dt_bias: *const f32, norm_w: *const f32, state: *mut f32, o: *mut f32, n: i32, nk: i32, nv: i32, dk: i32, dv: i32, eps: f32, st: Stream) -> i32 {
    if dk != 128 || dv != 128 {
        return 1;
    }
    launch(st, move || {
        const DK: usize = 128;
        let (n, nk, nv) = (n as usize, nk as usize, nv as usize);
        let (ps, z_off, b_off, a_off) = (ps as usize, z_off as usize, b_off as usize, a_off as usize);
        let cs = 2 * nk * DK + nv * DK;
        let qscale = (DK as f32).sqrt().recip();
        for hh in 0..nv {
            let kh = hh / (nv / nk);
            let mut s = vec![0.0f32; DK * DK];
            for k in 0..DK {
                for v in 0..DK {
                    s[k * DK + v] = *state.add(hh * DK * DK + k * DK + v);
                }
            }
            let a = -f32_at(a_log, hh).exp();
            let dtb = f32_at(dt_bias, hh);
            for t in 0..n {
                let cr = f32s(conv, (t + 1) * cs);
                let cr = &cr[t * cs..];
                let pr = f32s(proj, (t + 1) * ps);
                let pr = &pr[t * ps..];
                // q/k RMSNorm (each over 128 dims) then scaled.
                let qv: Vec<f32> = (0..DK).map(|i| cr[kh * DK + i]).collect();
                let kv: Vec<f32> = (0..DK).map(|i| cr[nk * DK + kh * DK + i]).collect();
                let qi = rms_inv(&qv, 1e-6) * qscale;
                let ki = rms_inv(&kv, 1e-6);
                let mut qs = vec![0.0f32; DK];
                let mut ks = vec![0.0f32; DK];
                for i in 0..DK {
                    qs[i] = qv[i] * qi;
                    ks[i] = kv[i] * ki;
                }
                let vv: Vec<f32> = (0..DK).map(|v| cr[2 * nk * DK + hh * DK + v]).collect();
                let decay = (a * softplus(pr[a_off + hh] + dtb)).exp();
                let beta = sigmoid(pr[b_off + hh]);
                let mut outv = vec![0.0f32; DK];
                for v in 0..DK {
                    let mut kvm = 0.0f32;
                    for k in 0..DK {
                        s[k * DK + v] *= decay;
                        kvm += s[k * DK + v] * ks[k];
                    }
                    let delta = (vv[v] - kvm) * beta;
                    let mut oacc = 0.0f32;
                    for k in 0..DK {
                        s[k * DK + v] += ks[k] * delta;
                        oacc += s[k * DK + v] * qs[k];
                    }
                    outv[v] = oacc;
                }
                let zr: Vec<f32> = (0..DK).map(|v| pr[z_off + hh * DK + v]).collect();
                let inv = rms_inv(&outv, eps);
                for v in 0..DK {
                    *o.add(t * nv * DK + hh * DK + v) = outv[v] * inv * *norm_w.add(v) * silu(zr[v]);
                }
            }
            for k in 0..DK {
                for v in 0..DK {
                    *state.add(hh * DK * DK + k * DK + v) = s[k * DK + v];
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// CPU reference: codec (kernels/codec.cu). Activations are time-major [T, C].
// `x` may point into a buffer with readable history rows before it (causal
// convs), so negative offsets are valid and handled with raw arithmetic.

fn snake(x: f32, a: f32) -> f32 {
    let s = (a * x).sin();
    x + s * s / (a + 1e-9)
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, |err| <= 1.5e-7.
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

fn gelu_erf(v: f32) -> f32 {
    0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752))
}

/// Round-to-nearest-even f32 -> f16 bits (matches `__float2half`).
fn f32_to_f16_rn(v: f32) -> u16 {
    let b = v.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7fffff;
    if e <= 0 {
        if e < -10 {
            return s as u16;
        }
        let m = m | 0x800000;
        let shift = (1 - e) as u32;
        let mut out = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rest > half || (rest == half && (out & 1) == 1) {
            out += 1;
        }
        return (s | out) as u16;
    }
    if e >= 31 {
        return if m == 0 { (s | 0x7c00) as u16 } else { (s | 0x7e00) as u16 };
    }
    let m = m + 0xfff + ((m >> 13) & 1);
    if m >= 0x800000 {
        return (s | ((e as u32 + 1) << 10)) as u16;
    }
    (s | ((e as u32) << 10) | (m >> 13)) as u16
}

/// CPU: out[t] = sum_c table[row(c, codes[t, c])].
pub unsafe fn dr_rvq_sum(codes: *const i32, frames: i32, ncb: i32, table: *const f32, out: *mut f32, d: i32, sem_size: i32, res_size: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (frames, ncb, d) = (frames as usize, ncb as usize, d as usize);
        let (sem, res) = (sem_size as usize, res_size as usize);
        for t in 0..frames {
            let orow = std::slice::from_raw_parts_mut(out.add(t * d), d);
            orow.fill(0.0);
            for c in 0..ncb {
                let size = if c == 0 { sem } else { res };
                let code = (*codes.add(t * ncb + c)).clamp(0, size as i32 - 1) as usize;
                let row = if c == 0 { code } else { sem + (c - 1) * res + code };
                let tr = std::slice::from_raw_parts(table.add(row * d), d);
                for (o, &v) in orow.iter_mut().zip(tr.iter()) {
                    *o += v;
                }
            }
        }
    })
}

/// CPU: causal im2col with optional snake -> f16.
pub unsafe fn dr_im2col(x: *const f32, n: i32, c: i32, k: i32, d: i32, s: i32, alpha: *const f32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        let (n, c, k, dil, strd) = (n as usize, c as usize, k as usize, d as isize, s as isize);
        let has_a = !alpha.is_null();
        for i in 0..n * k * c {
            let cc = i % c;
            let tj = i / c;
            let j = (tj % k) as isize;
            let t = (tj / k) as isize;
            let src = t * strd + (strd - 1) - ((k as isize - 1 - j) * dil);
            let mut v = *x.offset(src * c as isize + cc as isize);
            if has_a {
                v = snake(v, *alpha.add(cc));
            }
            *out.add(i) = f32_to_f16_rn(v);
        }
    })
}

/// CPU: one residual-VQ stage per row (argmax cosine, lower-index tie-break).
pub unsafe fn dr_vq_encode(r: *mut f32, rows: i32, d: i32, in_w: *const f32, in_b: *const f32, codebook_n: *const f32, n: i32, table: *const f32, codes: *mut i32, code_stride: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, d) = (rows as usize, d as usize);
        for row in 0..rows {
            let rr = std::slice::from_raw_parts_mut(r.add(row * d), d);
            let mut z = [0.0f32; 8];
            for o in 0..8 {
                let wr = std::slice::from_raw_parts(in_w.add(o * d), d);
                z[o] = wr.iter().zip(rr.iter()).map(|(a, b)| a * b).sum::<f32>() + *in_b.add(o);
            }
            let mut best = 0i32;
            let mut bv = f32::NEG_INFINITY;
            for e in 0..n as usize {
                let er = std::slice::from_raw_parts(codebook_n.add(e * 8), 8);
                let dot: f32 = z.iter().zip(er.iter()).map(|(a, b)| a * b).sum();
                if dot > bv {
                    bv = dot;
                    best = e as i32;
                }
            }
            let tr = std::slice::from_raw_parts(table.add(best as usize * d), d);
            for (a, &b) in rr.iter_mut().zip(tr.iter()) {
                *a -= b;
            }
            *codes.add(row * code_stride as usize) = best;
        }
    })
}

/// CPU: transposed-conv input gather ([snake(x[t]) | snake(x[t-1])]) -> f16.
pub unsafe fn dr_tconv_gather(x: *const f32, n: i32, c: i32, two: i32, alpha: *const f32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        let (n, c) = (n as usize, c as usize);
        let w = if two != 0 { 2 * c } else { c };
        let has_a = !alpha.is_null();
        for i in 0..n * w {
            let t = (i / w) as isize;
            let cc = i % w;
            let ch = cc % c;
            let src = if cc < c { t } else { t - 1 };
            let mut v = *x.offset(src * c as isize + ch as isize);
            if has_a {
                v = snake(v, *alpha.add(ch));
            }
            *out.add(i) = f32_to_f16_rn(v);
        }
    })
}

/// CPU: biased/scaled activation (+residual) -> f32 and/or f16.
pub unsafe fn dr_epilogue(y: *const f32, n: usize, c: i32, bias: *const f32, scale: *const f32, res: *const f32, act: i32, out: *mut f32, out16: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        let c = c as usize;
        for i in 0..n {
            let cc = i % c;
            let mut v = *y.add(i) + if bias.is_null() { 0.0 } else { *bias.add(cc) };
            if act == 1 {
                v = v.tanh();
            } else if act == 2 {
                v = gelu_erf(v);
            }
            if !scale.is_null() {
                v *= *scale.add(cc);
            }
            if !res.is_null() {
                v += *res.add(i);
            }
            if !out.is_null() {
                *out.add(i) = v;
            }
            if !out16.is_null() {
                *out16.add(i) = f32_to_f16_rn(v);
            }
        }
    })
}

/// CPU: depthwise causal conv + bias.
pub unsafe fn dr_dwconv(x: *const f32, t: i32, c: i32, k: i32, w: *const f32, b: *const f32, out: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        let (t, c, k) = (t as usize, c as usize, k as usize);
        for tt in 0..t {
            for cc in 0..c {
                let mut acc = *b.add(cc);
                for j in 0..k {
                    acc += *w.add(cc * k + j) * *x.offset((tt as isize - (k - 1 - j) as isize) * c as isize + cc as isize);
                }
                *out.add(tt * c + cc) = acc;
            }
        }
    })
}

/// CPU: affine LayerNorm per row -> f16.
pub unsafe fn dr_layernorm(x: *const f32, rows: i32, c: i32, w: *const f32, b: *const f32, eps: f32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, c) = (rows as usize, c as usize);
        for r in 0..rows {
            let xr = std::slice::from_raw_parts(x.add(r * c), c);
            let mean = xr.iter().sum::<f32>() / c as f32;
            let inv = (xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32 + eps).sqrt().recip();
            for i in 0..c {
                *out.add(r * c + i) = f32_to_f16_rn((xr[i] - mean) * inv * *w.add(i) + *b.add(i));
            }
        }
    })
}

/// CPU: plain-weight RMSNorm per row -> f32 and/or f16.
pub unsafe fn dr_rmsnorm_plain(x: *const f32, rows: i32, c: i32, w: *const f32, eps: f32, out: *mut f32, out16: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, c) = (rows as usize, c as usize);
        for r in 0..rows {
            let xr = std::slice::from_raw_parts(x.add(r * c), c);
            let inv = rms_inv(xr, eps);
            for i in 0..c {
                let v = xr[i] * inv * *w.add(i);
                if !out.is_null() {
                    *out.add(r * c + i) = v;
                }
                if !out16.is_null() {
                    *out16.add(r * c + i) = f32_to_f16_rn(v);
                }
            }
        }
    })
}

/// CPU: codec qkv split + interleaved rope (cos/sin rounded via bf16, like the
/// reference table) + ring KV-cache write.
pub unsafe fn dr_codec_rope(qkv: *const f32, n: i32, h: i32, inv_freq: *const f32, t0: i32, ring: i32, q: *mut f32, kc: *mut f32, vc: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        let (n, h) = (n as usize, h as usize);
        let w = h * 64;
        for r in 0..n {
            let t = t0 as usize + r;
            let slot = (t % ring as usize) * w;
            for i in 0..w / 2 {
                let head = i / 32;
                let p = i % 32;
                let (sn, cs) = (t as f32 * *inv_freq.add(p)).sin_cos();
                let c = bf16_to_f32(f32_to_bf16_rn(cs));
                let s = bf16_to_f32(f32_to_bf16_rn(sn));
                let base = r * 3 * w;
                let e = head * 64 + 2 * p;
                let q0 = *qkv.add(base + e);
                let q1 = *qkv.add(base + e + 1);
                let k0 = *qkv.add(base + w + e);
                let k1 = *qkv.add(base + w + e + 1);
                *q.add(r * w + e) = q0 * c - q1 * s;
                *q.add(r * w + e + 1) = q1 * c + q0 * s;
                *kc.add(slot + e) = k0 * c - k1 * s;
                *kc.add(slot + e + 1) = k1 * c + k0 * s;
                *vc.add(slot + e) = *qkv.add(base + 2 * w + e);
                *vc.add(slot + e + 1) = *qkv.add(base + 2 * w + e + 1);
            }
        }
    })
}

/// CPU: causal windowed attention over the ring caches, head_dim 64 -> f16.
pub unsafe fn dr_codec_attn(q: *const f32, kc: *const f32, vc: *const f32, n: i32, h: i32, window: i32, t0: i32, ring: i32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        let (n, h) = (n as usize, h as usize);
        let w = h * 64;
        for r in 0..n {
            let t = t0 as usize + r;
            let start = t.saturating_sub(window as usize - 1);
            for hh in 0..h {
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = [0.0f32; 64];
                for j in start..=t {
                    let at = (j % ring as usize) * w + hh * 64;
                    let mut s = 0.0f32;
                    for dd in 0..64 {
                        s += *q.add(r * w + hh * 64 + dd) * *kc.add(at + dd);
                    }
                    s *= 0.125;
                    let mn = m.max(s);
                    let corr = (m - mn).exp();
                    let p = (s - mn).exp();
                    l = l * corr + p;
                    for dd in 0..64 {
                        acc[dd] = acc[dd] * corr + p * *vc.add(at + dd);
                    }
                    m = mn;
                }
                for dd in 0..64 {
                    *out.add(r * w + hh * 64 + dd) = f32_to_f16_rn(acc[dd] / l);
                }
            }
        }
    })
}

/// CPU: y[t, j] = silu(h[t, j]) * h[t, I + j] -> f16.
pub unsafe fn dr_swiglu16(h: *const f32, y: *mut Half, rows: i32, i: i32, st: Stream) -> i32 {
    launch(st, move || {
        let (rows, ii) = (rows as usize, i as usize);
        for r in 0..rows {
            for j in 0..ii {
                let g = *h.add(r * 2 * ii + j);
                let u = *h.add(r * 2 * ii + ii + j);
                *y.add(r * ii + j) = f32_to_f16_rn(silu(g) * u);
            }
        }
    })
}

/// CPU: final layer — tanh over the snake-activated causal conv.
pub unsafe fn dr_conv_out(x: *const f32, n: i32, c: i32, k: i32, alpha: *const f32, w: *const f32, b: *const f32, out: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        let (n, c, k) = (n as usize, c as usize, k as usize);
        for t in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                for cc in 0..c {
                    acc += *w.add(j * c + cc) * snake(*x.offset((t as isize - (k - 1 - j) as isize) * c as isize + cc as isize), *alpha.add(cc));
                }
            }
            *out.add(t) = (acc + *b.add(0)).tanh();
        }
    })
}


