# gale ↔ Python Interop Strategy

**Forward-looking notes on giving gale a Python entry point and interoperating
with the Python CUDA ecosystem (numba / cupy / torch).** Not yet implemented —
this scopes *what's possible* and *what it costs* so the eventual PyO3 work starts
from a clear picture. All cuda-oxide hooks cited are verified against the v0.2.0
tree (2026-06-08). Companion to
[`cuda-oxide-codegen-notes.md`](./cuda-oxide-codegen-notes.md) §5 (monomorphization
/ runtime-dispatch maturity) and [`api-design.md`](./api-design.md).

---

## TL;DR

| Capability | Status | Mechanism |
|---|---|---|
| **Python drives gale** (PyO3 over host API) | Feasible now (work is ours) | Kernels stay AOT in the Rust cdylib; Python calls host launch fns |
| **Zero-copy array exchange** with cupy/numba/torch | Feasible now (glue is ours) | `DeviceBuffer::from_raw_parts` / `cu_deviceptr` + `__cuda_array_interface__`/DLPack |
| **numba `@cuda.jit` kernels co-executing in gale's loop** | Feasible (coarse-grained) | Shared primary context + stream; Python orchestrates launches on shared buffers |
| **Runtime host callbacks** (completion signaling) | Present | `stream.launch_host_function(FnOnce)` (`cuLaunchHostFunc`); `cuda-async` `DeviceFuture` |
| **Python fn called *on the device*** | ✗ Not possible | Device code is AOT; Python must be compiled to PTX (that's what numba does) |
| **numba kernel fused *inside* a gale device kernel** | ✗ Not realistic | Needs runtime device-linking or fn-pointers across modules (unproven) |
| **Runtime device-side dispatch** (fn-ptr table) | Plumbed, unproven on sm_70 | See codegen-notes §5; default to monomorphization |

**The headline:** everything needed for a first-class Python entry point and
zero-copy ecosystem interop **exists in cuda-oxide today** — the remaining work is
host-side PyO3 + array-interface glue that *we* write, not a missing cuda-oxide
capability. The only genuine "can't" is calling a Python function *on the GPU*; the
substitute is exchanging arrays with numba's own compiled kernels.

---

## 1. The entry point — PyO3 over gale's host API

The primary path is unremarkable and robust: a PyO3 module wraps gale's *host-side*
solver objects (`sim.step()`, field accessors, etc.). **Kernels stay AOT-compiled
into the Rust `.so`** — Python never compiles gale kernels at runtime. No callbacks,
no JIT, no device-side magic. This carries the bulk of the value and should land
first.

---

## 2. Array & handle sharing — the interop hinge

Interop with the Python CUDA ecosystem is about **sharing device memory**, not
callbacks. cuda-oxide exposes the raw pointers/handles that make zero-copy exchange
possible (all present in v0.2.0):

| Need | cuda-oxide hook |
|---|---|
| **Adopt** an external device pointer (cupy/numba/torch array) | `DeviceBuffer::from_raw_parts(ptr: CUdeviceptr, len, ctx)` — `cuda-core/src/device_buffer.rs:183` |
| **Hand out** a gale buffer | `into_raw_parts()` (`:195`), `cu_deviceptr()` (`:147`) |
| Share/adopt a context | `ctx.cu_ctx()` → raw `CUcontext` (`context.rs:135`); `CudaModule::from_raw` (`module.rs:281`) |
| Load **foreign** PTX/cubin (e.g. numba-emitted) | `load_module_from_ptx_src` / `load_module_from_file` / `load_module_from_image` (`module.rs:80-130`) |

**To accept** a cupy/torch/numba array: read its `__cuda_array_interface__` (CAI) or
DLPack capsule → `(ptr, shape, strides, stream)` → `from_raw_parts(ptr, len, ctx)`.
**To expose** gale output: implement CAI (or build a DLPack `DLManagedTensor`) on the
PyO3 wrapper from `cu_deviceptr()` + shape/dtype. cuda-oxide ships neither CAI nor
DLPack, but it gives the raw pointer, so each is a thin wrapper on the PyO3 side.

**Lifetime discipline:** `from_raw_parts` adopts a pointer gale does **not** own —
gale must not free it (use `into_raw_parts()` / `mem::forget` to release without
freeing). Conversely, a buffer handed to Python must outlive Python's use of it.

---

## 3. The context/stream gotcha (the thing that bites)

For a shared pointer to be valid on both sides, both must use the **same context**.
cuda-oxide retains the device **primary context** (`cuDevicePrimaryCtxRetain`,
`context.rs:108`).

- **cupy, numba, and torch default to the primary context** → they line up
  naturally; `from_raw_parts` Just Works.
- **pycuda is the exception** — it creates its *own* context
  (`make_default_context`), which won't match, so pointers aren't directly
  shareable without explicit context juggling.

**Steer Python-side interop toward cupy / numba / torch, not pycuda.**

Streams: launch interleaved work on **one shared stream** (gale exposes raw
`CUstream`; numba/cupy accept a stream arg) for correct ordering, or synchronize
explicitly between hand-offs.

---

## 4. Runtime host callbacks — what they're for, and their ceiling

cuda-oxide *does* have a runtime host callback: `stream.launch_host_function(f)`
enqueues an `FnOnce` host closure that fires after prior stream work completes
(`cuda-core/src/stream.rs`, backed by `cuLaunchHostFunc`). `cuda-async` uses it to
bridge stream completion to a Rust `Future` (`DeviceFuture` registers the callback
to wake its waker — `cuda-async/src/device_future.rs:133`).

**Two hard constraints** (from the CUDA contract): a `cuLaunchHostFunc` callback
**must not call any CUDA API** (no kernel launch, no memcpy) and must not block. So
it's for *signaling/bookkeeping* — wake a future, flip a flag, notify Python of
progress/completion — **never** for issuing more GPU work mid-stream.

For Python: you *could* call into Python from such a callback via PyO3, but (a)
acquiring the GIL on a driver callback thread is fragile, and (b) the no-CUDA rule
forbids launching Python-CUDA work from inside it. So host callbacks suit completion
notification; the clean async story is to expose gale's `DeviceFuture` as a Python
awaitable, or just block-and-return.

---

## 5. numba co-execution — JIT Python kernels in gale's loop

**This is feasible, and it's the clean version of "runtime-pluggable physics."** The
shape that works is **co-execution orchestrated from Python**, not numba code embedded
in gale's device kernels:

1. PyO3-wrapped gale exposes its state buffers as CAI/DLPack views (from
   `cu_deviceptr()`).
2. A numba `@cuda.jit` kernel (numba JIT-compiles it to PTX at first call, via its
   *own* NVVM toolchain) reads/writes that same array.
3. Python sequences the launches:
   `gale.stage_a()` → `my_numba_kernel[grid,block](state_view, ...)` → `gale.stage_b()`.

gale never compiles, hosts, or links the numba code — it shares memory and the
surrounding solver. The driver prerequisites all line up (shared primary context §3,
CAI/DLPack array exchange §2, shared stream §3).

**Use case:** runtime-pluggable custom terms — a forcing term, a new constitutive
closure, a diagnostic — written in Python and injected into gale's time loop
**without rebuilding gale**.

**sm_70 bonus:** numba's CUDA target supports Volta cleanly through its own mature
NVVM path, **independent of cuda-oxide's codegen**. A numba-provided kernel does
*not* inherit the pre-Blackwell typed-pointer fork fragility — in that narrow sense
it's *more* robust on the Titan V than gale's own kernels are today.

### Caveats

- **Launch-granularity cost.** A DG solver fires many small kernels per step.
  Dropping to the Python interpreter to launch a numba kernel each substage adds
  host-side latency that can dominate in the hot inner loop. Fine for prototyping and
  coarse/occasional terms; costly at high step counts. Mitigations: keep the hot loop
  in Rust and call out only at coarse boundaries, batch, or wrap the repeated
  sequence in a CUDA graph.
- **Forfeits the CPU-oracle guarantee for that term.** gale's testing posture is
  bit-for-bit agreement with the Rust CPU path; a runtime-injected numba kernel is
  user code with no Rust twin, so it can't be auto-cross-checked. Design/testing
  consideration, not a blocker.
- **Layout agreement is on the user.** A CAI/DLPack view is a flat pointer + shape;
  the numba author must know gale's nodal DOF ordering (element-major, p-dependent).
  Expose layout helpers / document the convention.

---

## 6. Boundaries — what is NOT realistic

- **Python function on the device.** No — device code is AOT-monomorphized. Python
  must be compiled to PTX (numba), then exchanged as a *separate kernel/array*, never
  called as a device function from inside a gale kernel.
- **Fine-grained fusion** (numba snippet inlined into gale's matrix-free operator).
  Needs runtime device-linking (gale AOT vs numba JIT) or cross-module device
  fn-pointers — the deep, unproven path. Composition is **coarse-grained** (separate
  launches on shared buffers).
- **Runtime device-side dispatch in gale's own kernels.** cuda-oxide has *partial*
  fn-pointer plumbing (reify/closure-coercion casts at
  `mir-importer/src/translator/rvalue.rs:529-549`; an indirect-call export path at
  `llvm-export/src/export/ops.rs:809`) but **no validated end-to-end device fn-pointer
  path, and none on sm_70**; virtual/`dyn` device dispatch shows no evidence; dynamic
  parallelism is absent. See codegen-notes §5. **Default to monomorphization**; push
  runtime extensibility to the *orchestration* layer (this doc's §5 numba pattern)
  rather than into device codegen.

---

## 7. Suggested build order (when we get here)

1. **PyO3 over the host API** (§1) — solver lifecycle + field access. The 80% value.
2. **CAI export** on field/buffer wrappers (§2) — lets cupy/numba/torch *read* gale
   output zero-copy. Smallest useful interop increment.
3. **CAI/DLPack import** via `from_raw_parts` (§2) with lifetime discipline — lets
   gale *consume* external arrays.
4. **Shared-stream plumbing** (§3) — thread one `CUstream` through both sides.
5. **numba co-execution demo** (§5) — a custom forcing term as a `@cuda.jit` kernel
   in the gale loop, validated against a Rust reference for that term.

Steps 2–4 are the foundation; everything else (numba, torch, cupy) rides on them.
