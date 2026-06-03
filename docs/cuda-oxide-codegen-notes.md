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
