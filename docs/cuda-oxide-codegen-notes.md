# cuda-oxide Codegen Notes — Bitcast Fix Scope & #58 Analysis

**Working notes for the fixes gale carries on its cuda-oxide fork
(`ianrgraham/cuda-oxide@pr-101-typed-nvvm`).** Validated on the 2× Titan V box
(sm_70, CUDA 12.9) on 2026-06-01. Companion to
[`milestone-1-probe-results.md`](./milestone-1-probe-results.md) and
[`cuda-oxide-repo-status.md`](./cuda-oxide-repo-status.md) §8.1.

Reproducers (kept as regression gates):
- `src/bin/probe_fp64_math.rs` — exercises the typed-pointer path (libdevice math).
- `src/bin/probe_nested_write.rs` — reproduces #58.

---

## TL;DR

Two bugs that *look* related (both involve `get_mut` / `Option<&mut>`) but have
**different root causes in different compiler stages and code paths**:

| | Typed-pointer **bitcast bug** (ours, on PR #101) | **#58** nested-array dropped writes |
|---|---|---|
| Stage | `dialect-llvm` NVVM-IR **text exporter** | `mir-lower` / `mir-importer` **place lowering** |
| Path | **libdevice only** (NVVM-IR text → libNVVM) | **PTX path** (`llc`); any kernel |
| Symptom | libNVVM parse error, won't compile (typed mode) | compiles & runs, **writes silently dropped** |
| Trigger | `Option<&mut T>` pointer field printed with wrong type | nested `get_mut(i)` into `&mut [T; N]` |
| Fix locus | `export/ops.rs::emit_insert_value` (+ audit call/ret/phi) | place projection through `Option<&mut>` deref+index |

**They must be fixed independently.** Fixing one does nothing for the other. They
share only the *surface syntax* `get_mut`/`Option<&mut>` — a fragile area of the
lowering worth hardening, but the defects are unrelated.

---

## 1. The typed-pointer bitcast bug (our PR #101 follow-up)

### Symptom
With #101's backend on sm_70, a libdevice kernel that writes via
`out.get_mut(idx)` fails at libNVVM:

```
probe_fp64_math.ll (62, 43): parse '%v36' defined with type 'double*'
```
Line 62 is `insertvalue { i8, i8* } %v37, i8* %v36, 1`, but `%v36` is a `double*`
(produced by a GEP at line 60). Legal in opaque-pointer mode (`ptr`), **illegal in
typed-pointer mode**: you can't use a `double*` value where `i8*` is declared
without a `bitcast`.

### Why it only appears on the libdevice path
cuda-oxide emits PTX two ways:
- **No libdevice** → LLVM 22 → PTX via `llc` *in memory*. LLVM handles pointer
  types internally; the `dialect-llvm` **text exporter is never used**. (This is
  why `probe-sm70`'s `get_mut` kernels pass.)
- **libdevice present** → emit **NVVM IR text** via `dialect-llvm/src/export/*`,
  skip `llc`, hand to libNVVM. The text exporter's typed-pointer printing is where
  the bug lives. (This is why only libdevice kernels hit it.)

### Root cause
The exporter tracks the concrete LLVM type each pointer value was *emitted* as in
`State::typed_pointer_value_types: HashMap<Value, String>`
(`export/state.rs:59`), populated by GEP and friends (`export/ops.rs:553,598,1408`).
Memory ops reconcile a pointer operand's emitted type against the type a site
*requires* via **`pointer_operand()`** (`export/ops.rs:1449`), which emits a
`bitcast`/`addrspacecast` when they differ:

```rust
let desired = self.pointer_type_for_pointee(pointee_ty, addrspace)?;
let current = self.pointer_value_type(ptr)?;
if current != desired { /* emit  %ptrcastN = bitcast <current> %v to <desired> */ }
```
`emit_load`/`emit_store`/`emit_gep` all route operands through it (ops.rs:503, 524,
572, 617, 641, 670, 705). **`emit_insert_value` (ops.rs:1276) does not.** It prints
the value with the aggregate's *declared* element type while the value name carries
a *different emitted* type:

```rust
// emit_insert_value, ops.rs:1293-1295
self.export_type(val.get_type(self.ctx), output)?;   // prints "i8*"  (slot type)
write!(output, " ").unwrap();
self.export_value(val, value_names, output)?;        // prints "%v36" (emitted double*)
```

That `{ i8, i8* }` aggregate is what `DisjointSlice::get_mut` → `Option<&mut f64>`
lowers to (discriminant + erased pointer); the pointer field's value is a typed
GEP result.

### Fix scope (proposed; small, self-contained)
1. **Add a general coercion helper** alongside `pointer_operand`, e.g.
   `coerce_pointer_to(&mut self, val, target_type: &str, value_names, output) -> String`
   that, in `TypedPointers` mode, looks up `typed_pointer_value_types[val]`; if it
   differs from `target_type` and both are pointers, emits
   `%ptrcastN = bitcast <emitted> %val to <target>` and returns the cast name;
   otherwise returns the plain name. (`pointer_operand` becomes a thin wrapper that
   computes `target` from `(pointee_ty, addrspace)`.)
2. **Use it in `emit_insert_value`** for the value operand, with `target` = the
   aggregate element type at the inserted index. (Confirmed-broken site.)
3. **Audit & likely apply the same coercion at:**
   - `emit_call` (ops.rs:866) — pointer args vs the callee's declared param types.
   - `emit_return` (ops.rs:437) — pointer return vs the function's return type.
   - phi incomings, and `emit_store`'s *value* operand when storing a pointer.
   These weren't exercised by our kernel but are the same class of site.
4. **Tests:** extend `crates/dialect-llvm/tests/export_test.rs` (#101 already edits
   it) with an `insertvalue` of a typed pointer into an `i8*` slot, asserting a
   `bitcast` is emitted.

Why not "just emit GEP results as `i8*`"? Because LLVM-7 typed-pointer mode needs
typed GEPs/loads; the value genuinely *is* `double*` at its def. The reconciliation
belongs at the **use site**, exactly as `pointer_operand` already does for memory
ops. This fix simply extends that discipline to aggregate/call/return sites.

### gale workaround until landed
Avoid `Option<&mut>` accessors in libdevice kernels: write via
`get_unchecked_mut(i)` after an explicit `i < out.len()` bounds check
(see `probe_fp64_math.rs`). Pure-arith / shared-mem kernels are unaffected.

### Update (2026-06-02): `f64::abs()` is *also* a trigger — not just libdevice math

While porting the hyperbolic operator (`gpu_advection.rs`, linear advection), a
kernel doing only arithmetic + shared memory + `out.get_mut(idx)` hit the **same**
bitcast error:
```
gale (356, 45): parse '%v240' defined with type 'double*'
```
The sole cause was a single `an.abs()` in the Rusanov flux. `f64::abs()` lowers to
the `llvm.fabs.f64` **intrinsic**, which appears to route the *whole* kernel through
the NVVM-IR text exporter (the libNVVM path) instead of the PTX/`llc` path — so the
typed-pointer bitcast defect fires even though no libdevice *function* is called.

**Refinement to §1's characterization:** the path is selected by "contains any LLVM
intrinsic the PTX path can't emit," not strictly "calls a libdevice math function."
Float intrinsics (`abs`, almost certainly `sqrt`/`floor`/`min`/`max`/`copysign`,
fma) are enough.

**gale workaround (confirmed):** open-code the intrinsic. Replacing `an.abs()` with
`if an < 0.0 { -an } else { an }` keeps the kernel on the PTX path; the operator
then matches the CPU oracle to `3.7e-16`. This matters for the upcoming **Euler GPU
port**, which needs `sqrt`/`ln` (genuine libdevice) — that one *cannot* be
open-coded and will require either the §1 exporter fix landed or the
`get_unchecked_mut` + ltoir-loader path.

---

## 2. Issue #58 — nested-array writes through `get_mut` silently dropped

### Reproduced (this hardware, 2026-06-01)
`src/bin/probe_nested_write.rs`, `DisjointSlice<[f32; 4]>`, 64 elements:
```
[FAIL] get_mut(i):  0/256 elements written to 42
[PASS] for e in a:  256/256 elements written to 42
```
Both kernels are **libdevice-free** → PTX path → the text exporter is never
involved. So #58 cannot be the bitcast bug.

### Root cause (from the generated PTX)
`via_get_mut` allocates a stack frame and routes the element store through it:
```
.local .align 8 .b8  __local_depot0[8];
...
cvta.to.local.u64    %rd15, %rd19      ; address taken into LOCAL (stack)
st.b32  [%rd19], 1109917696            ; 1109917696 = 0x42280000 = 42.0f → stack copy
```
`via_iter` instead writes `42.0` through the pointer derived from the global
output parameter. So the nested `get_mut` chain
(`DisjointSlice::get_mut` → `&mut [f32;N]`, then `[f32;N]::get_mut(i)` →
`Option<&mut f32>`) loses the "this is a place in device global memory" property
and operates on a **by-value stack copy** that is never written back — a
place/projection **lowering** defect in `mir-lower`/`mir-importer`, not a
serialization defect.

### Relationship to the bitcast bug
**Cousins, not the same bug.** Both surface through `Option<&mut>`/`get_mut`, so the
`get_mut` lowering is clearly a fragile area — but:
- bitcast bug = how a pointer *field is printed* in typed NVVM-IR text;
- #58 = whether the place *aliases global memory or a stack copy* at the PTX level.
Fixing the text-export bitcast does nothing for #58, and vice versa.

### gale workaround
Prefer iteration (`for e in a { … }` / `iter_mut`) or direct
`get_unchecked_mut`/indexed writes over chained `get_mut(i)` into nested arrays in
device kernels, until #58 is fixed. `probe-nested-write` is the regression gate.

---

## 3. Suggested order of work on the fork

1. **Bitcast fix** (§1) — small, removes the `get_unchecked_mut` workaround for
   libdevice kernels; first upstream PR (folds into / follows #101).
2. **#58 place-lowering fix** (§2) — larger (touches MIR place projection); removes
   a silent-wrong-answer hazard that directly threatens DG DOF-array writes.
3. Re-run `probe-fp64-math` and `probe-nested-write`; both should go green without
   workarounds. Then bump the `rev` in `gale/Cargo.toml` and rebuild the backend.

Both are genuine upstream contributions (issue #58 is open and unassigned; the
bitcast gap is a concrete follow-up to PR #101 with a minimal repro).

---

## 4. GPU-port coverage (evidence: what works vs what's blocked)

As of 2026-06-02, gale has bit-for-bit-validated GPU kernels for a broad set of
operators on sm_70 via the **embedded `#[cuda_module]` path**. The pattern that
emerges is clean and useful as cuda-oxide evidence:

**Ports cleanly on the embedded path (arithmetic + shared mem + `Option<&mut>`):**
| Kernel | bin | GPU vs CPU |
|---|---|---|
| SIPG operator / CG / p-MG PCG (elliptic) | `gpu-poisson-*` | ~1e-16 |
| Hyperbolic weak form (linear advection) | `gpu-advection` | bit-for-bit |
| Split-form entropy-stable (Burgers) | `gpu-burgers-split` | 2.9e-15 |
| Oldroyd-B conformation transport | `gpu-oldroyd` | 0 (exact) |
| Volume-penalization apply (IBM) | `gpu-penalize` | 0 (exact) |

Two independent confirmations worth noting:
- **`f64::abs()` forces the libdevice/NVVM-text path** and trips the §1 bitcast bug
  even in an otherwise pure-arithmetic kernel (see §1 update). Open-coding it
  (`if x<0 {-x} else {x}`) keeps the kernel on the PTX path. So entropy-*stable*
  (Rusanov/LLF) hyperbolic kernels must open-code the wave-speed `abs`; the
  entropy-*conserving* (central) variant needs no `abs` at all.
- In-place `Option<&mut f64>` read-modify-write (single scalar, not nested array)
  is fine — `gpu-penalize` does it and matches exactly. The §2/#58 defect is
  specific to **nested-array** `get_mut` chains.

**Previously blocked on the libdevice path — now UNBLOCKED by the §1 fix:**
- **Compressible Euler** flux/wave-speed (`sqrt`, division-by-pressure).
- **Log-conformation** viscoelastic transport (2×2 symmetric **eigendecomposition**
  → `sqrt`/`atan2`, and the matrix `exp`/`log`).

### §1 fix: IMPLEMENTED & validated (2026-06-02)

The typed-pointer bitcast bug is **fixed** in the fork
(`crates/dialect-llvm/src/export/ops.rs`): a new `coerce_pointer_value` helper (a
sibling of `pointer_operand`) bitcasts a concretely-typed pointer value (e.g. a GEP
`i32*`/`double*`) to the erased field/operand type (`i8*`) before it is consumed,
and `emit_insert_value` now calls it so the cast is emitted *before* the
`insertvalue` line. Validated:
- new regression test `typed_nvvm_export_casts_typed_pointer_before_insertvalue`
  (export_test.rs) — 7/7 export tests pass, all 26 dialect-llvm tests green (no
  regression);
- the real gale kernel **`gpu-advection` with `f64::abs()`** (which previously
  failed `parse '%v240' defined with type 'double*'`) now **compiles and runs**,
  matching the CPU oracle to 4.2e-16, against the rebuilt backend.

**Confirmed by two real ports:**
- `gpu-euler` (compressible Euler weak-form, nv=4, Rusanov flux with in-kernel
  `sqrt(γp/ρ)` + `abs`) — matches CPU to **4.7e-14**.
- `gpu-logconf` (log-conformation Ψ transport — per-node eigendecomposition + matrix
  `exp`, i.e. `sqrt`/`exp` in-kernel) — matches CPU to **5.3e-15**.

The libdevice/NVVM-text path is proven working post-fix on the two operators the bug
previously blocked. **The GPU port of the entire headline operator set is complete.**

### Second finding: `atan2` is not in the device-intrinsic table

While porting log-conformation, `f64::atan2` (`std::sys::cmath::atan2`) caused a
*different* backend panic (`collector.rs:809`: "Device code calls: …atan2") — the
collector's allow-list / libdevice mapping covers `sqrt`/`sin`/`cos`/`exp` but not
`atan2` (no `__nv_atan2` mapping). This is a **separate, smaller upstream gap** from
the §1 bitcast bug. Worked around in gale by computing the 2×2 symmetric
eigenvector **directly** (`v=(μ₁−r, q)`, normalized; `μ₁−r ≥ 0` matches the host
`atan2`-derived `c≥0` branch to round-off) — which avoids the transcendental and is
faster anyway. The clean upstream fix is to add `atan2`→`__nv_atan2` to the
collector's mapping (and likely the other missing libdevice transcendentals).

Result: **Euler and log-conformation GPU ports are now possible.** The fix is local
to the fork checkout (so the loaded `.so` has it); it is a clean upstream
contribution (PR-ready) following PR #101. Remaining audit (same coercion class, not
yet exercised by a failing kernel): `emit_call` pointer args, `emit_return`, phi
incomings, and `emit_store`'s value operand — apply `coerce_pointer_value` there too
when a repro surfaces.

### Third finding: opaque pointers (`ptr`) emitted for some kernels → libNVVM rejects

Porting the **3D hex** hyperbolic operator (`src/bin/gpu_advection3d.rs`, the 3D
analogue of `gpu-advection`) surfaced a new backend bug. The kernel fails at NVVM
compile:

```
Ltoir(Nvvm(Call { operation: "nvvmCompileProgram", code: 9,
                  log: Some("gale (14, 24): parse expected type") }))
```

The emitted NVVM IR (`cargo oxide build --bin gpu-advection3d --emit-nvvm-ir`) shows
the function signature uses **opaque pointers**:

```llvm
define void @advect_rhs(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ... ) {
```

whereas the working 2D kernels emit **typed pointers** (`i8*`), e.g. `gpu-advection`:

```llvm
define void @advect_rhs(i8* %v0, i64 %v1, i8* %v2, i64 %v3, ... ) {
```

libNVVM (pre-opaque-pointer) cannot parse `ptr` and errors at the first parameter
(line 14, col 24 = the `ptr` token). So the backend's typed-pointer lowering — the
same area as the §1 fix — is **applied to the 2D kernels but not to this 3D kernel**;
some IR pattern in it routes around the opaque→typed conversion.

Isolation performed (all reproduce the same `(N,24) parse expected type`, where `N`
tracks the number of shared-array decls preceding the `define`):
- **Not** parameter count: reducing 23→12 params (packing the 9 metrics node-major
  into one buffer + the 4 face floats into another) did not change it.
- **Not** the conditional shared-memory load (`if m < n2 { DS[m]=d[m] }`): replaced
  with an unconditional load over a host-padded diff buffer; no change.
- **Not** shared-array count (5→3 by packing PR/PS/PT into one array): error line
  shifted with the count but persisted.
- **Not** shared-array size: `p=3` (`[64 x double]`, smaller than the 2D kernels'
  working `[81 x double]`) still emits `ptr`.

So the trigger is structural to this kernel, not count/size/signature-width. The CPU
operator it checks against (`Hyperbolic3d`) is correct and unit-tested; the binary is
kept as the **minimal reproducer** for the upstream fix (extend the typed-pointer
lowering to cover whatever pattern this kernel hits — candidates: the 3-way tensor
volume contraction indexing, or the `met[b*9+c]` packed-stride loads). This blocks
the 3D GPU operator (and thus 3D multi-GPU GPU execution) until fixed upstream; the
2D GPU path and the whole CPU 3D stack are unaffected.

#### Resolution

Root cause: **`NvvmIrDialect::for_target` defaulted unknown/`None`/sentinel targets
to `OpaquePointers`.** The 3D hex kernels use `f64::abs` (`__nv_fabs`), so the
pipeline auto-switches to the NVVM-IR path (`needs_libdevice`). In that path the
embedded `.ll` is exported once at codegen time and compiled by `nvvmCompileProgram
-gen-lto` at *runtime* on the host GPU. When `CUDA_OXIDE_TARGET` was not threaded
through, the export saw the `"nvvm-ir"` sentinel / `None` and picked OpaquePointers
→ opaque `ptr` → pre-Blackwell libNVVM `parse expected type`. (The 2D kernels
happened to get a concrete `sm_70` and so rendered typed `i8*`.)

Fix (fork `crates/dialect-llvm/src/export/config.rs`): invert the default so only
an **explicit** Blackwell-or-newer target gets opaque pointers; pre-Blackwell **and
unknown** targets get the typed-pointer dialect, which is always loadable on
pre-Blackwell libNVVM:

```rust
match target.and_then(cuda_arch_major) {
    Some(major) if major >= 10 => Self::OpaquePointers,
    _ => Self::TypedPointers,   // pre-Blackwell AND unknown → typed (libNVVM-safe)
}
```

After rebuilding the backend, `gpu-advection3d` (p=4, 125 nodes/elem, 3375 dofs)
emits typed `i8*` + the legacy NVVM datalayout and matches the CPU `Hyperbolic3d`
operator to **5.078e-16** on the 2× Titan V. This is a clean upstream contribution
(PR-ready, following the §1/§2 typed-pointer fixes). The 3D GPU operator — and thus
the 3D multi-GPU path — is unblocked.

---

## 5. Generic kernels & closures are **monomorphized** (mechanism + gale implications)

**Verified against the v0.2.0 tree (2026-06-08).** Relevant when we consider making
device kernels generic over a physics trait (e.g. a `conf_rhs<M: ConstitutiveModel>`
that swaps Oldroyd-B / Giesekus / FENE-P), or accepting host closures à la the
`host_closure` example. The mechanism is **standard Rust monomorphization**, not
runtime trait-object dispatch or an on-device interpreter.

### Evidence (cuda-oxide source)

- The backend consumes **rustc's already-monomorphized mono-items**:
  `rustc-codegen-cuda/src/lib.rs:471` calls `tcx.collect_and_partition_mono_items(())`.
  By the time codegen runs, every generic is resolved to a concrete `Instance`.
- It only emits **fully-monomorphized** functions: `collector.rs:319-321` gates on
  `MonoItem::Fn(instance)` with `is_fully_monomorphized(tcx, *instance)`
  (`collector.rs:376` = "no unresolved type parameters"). A generic kernel with open
  type params is **never codegen'd** — only concrete instantiations are.
- **Each monomorphization gets its own export name.** `collector.rs:180-181`:
  non-generic kernel → `base_name`; generic kernel with N type args →
  `base_name + "_TID_" + hex32`, where `compute_kernel_export_name`
  (`collector.rs:206`) hashes the instance's concrete type args into the `hex32`.
  So `conf_rhs<OldroydB>` and `conf_rhs<Giesekus>` become **two distinct cubin
  symbols** — which automatically satisfies our crate-wide unique-export-name device
  bundle constraint.
- **Closures** (`host_closure/src/main.rs`): the closure value is pushed as a single
  **byval `.param` struct** (the captures, as runtime bytes), while `F` binds to the
  closure's anonymous type at the call site (`module.map::<f32, _>`). Distinct closure
  type ⇒ distinct `F` ⇒ distinct monomorphization. `FnMut`/`FnOnce` dispatch through
  `<F as FnMut>::call_mut` etc., still monomorphized per `F`.

### The reachability subtlety (why it's compile-time, not runtime-pluggable)

rustc only monomorphizes **reachable** instances, so a generic `#[kernel]` must
actually be instantiated to exist in the cubin — either by a real typed-launch call
site (`module.map::<f32, _>` forces it) or via the macro's explicit instantiation
list (`cuda-macros/src/lib.rs:100-101`: `instantiate_types: Vec<Type>`, "Types to
instantiate generic kernels for", + `INSTANTIATE_PREFIX`). That hook is how we'd
force `{OldroydB, Giesekus, FENE-P}` to compile before any call site exists.

### Net for gale

- **You get one compiled kernel per concrete type-set you instantiate**, each
  uniquely named by a hash of its type args. The split is: **structure = compile-time
  type parameter; numeric coefficients = runtime kernel args** (scalars). For a
  constitutive model that's a clean fit — the model *family* is the type `M`, and
  `λ, η_p, Giesekus α, FENE L²` are just scalar `.param`s, so **no heap-closure
  serialization is needed**.
- **Bound:** the model set is fixed at build time. There is no "user passes an
  arbitrary closure at runtime without recompiling" — a closure/model type that
  doesn't exist at compile time can't become a kernel.
- **Why this is attractive architecturally:** making the device kernel generic over
  the *same* `ConstitutiveModel` / `ConservationLaw` traits the host uses (or a
  device-compatible subset) collapses the current hand-duplicated GPU Oldroyd kernel
  (`gale-gpu/src/operators/oldroyd.rs`, a copy of the host `OldroydB`) into one source
  of truth — keeping the bit-for-bit CPU oracle aligned with the GPU by construction.
- **Caveat:** on sm_70 all of this still rides the pre-Blackwell typed-pointer path
  (§1/§4), so it inherits the fork's typed-pointer dependency.

> **Where closures actually pay off** (host-sampled-to-array vs device functor):
> a quantity that depends only on `(x,y,t)` should stay host-sampled (current pattern
> for BCs in `src/dg/operators/bc.rs`, ICs in `src/sim/state.rs`, forcing in
> `src/sim/term.rs`). Push it into the kernel as a monomorphized functor **only when
> it's evaluated per-thread on per-thread runtime state** — the constitutive
> relaxation term `f(C)` is the prime example; the `ConservationLaw` flux/Riemann
> choice is second; a device-side moving-particle indicator `χ(x,y,t)` for many-body
> IBM is third. BC *values* are the weakest case (cheap to host-sample, no state
> dependence).

### Correction: monomorphization is the *validated* path, not the only conceivable one

The "fixed compile-time set" bound above is about **what's proven in cuda-oxide**,
not a CUDA or Rust-language limit. CUDA C/C++ supports runtime device-side
indirection three ways — `__device__` function-pointer tables, virtual dispatch on
device-constructed objects, and dynamic parallelism — and Rust has `fn` pointers and
`dyn` too. What cuda-oxide actually lowers (verified against v0.2.0):

- **Function pointers / indirect calls — plumbed, unproven.** The MIR frontend
  translates fn-pointer reification + closure→fn coercion casts
  (`mir-importer/src/translator/rvalue.rs:529-549`,
  `dialect-mir/src/attributes.rs:38-40`), and the LLVM exporter has an indirect-call
  path (`CallOpCallable::Indirect`, `llvm-export/src/export/ops.rs:809`). So a
  `__device__` fn-pointer-table style runtime dispatch is *representable end-to-end*,
  but **no example exercises it and it is unverified on sm_70** (and likely fragile on
  the pre-Blackwell typed-pointer path).
- **Virtual `dyn` dispatch on device — no evidence.** Every `dyn`/vtable hit is the
  compiler's own internal architecture, not lowering of user trait objects; the only
  user-facing trace is a fall-through comment for trait-object Unsize coercions
  (`mir-lower/src/convert/ops/cast.rs:328`). Treat as unsupported/unverified.
- **Dynamic parallelism — absent.** Zero `cudaLaunchDevice` / device-launch hits.

**Practical stance:** default to monomorphization (the only validated path). If we
ever want runtime model selection without recompiling, the better lever is the
*orchestration* layer (Python launches / numba co-execution —
[`python-interop-strategy.md`](./python-interop-strategy.md) §5), not device-side
indirection. A device fn-pointer probe on the Titan V is the only way to turn
"plumbed, unproven" into a real yes/no, if it ever matters.
