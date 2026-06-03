# Milestone-1 Probe Results — cuda-oxide on 2× Titan V (sm_70)

**Empirical answers to the cuda-oxide go/no-go questions, run on the actual
hardware.** Companion to [`cuda-oxide-repo-status.md`](./cuda-oxide-repo-status.md)
§6 and [`dg-gpu-fluid-simulation.md`](./dg-gpu-fluid-simulation.md) §9.

> **Run date: 2026-06-01.** Probe sources: `src/bin/probe_sm70.rs` (core,
> libdevice-free) and `src/bin/probe_fp64_math.rs` (isolated libdevice math).
> Reproduce with `cargo oxide run --bin probe-sm70` and
> `cargo oxide run --bin probe-fp64-math`.

## Environment

| Component | Value |
|---|---|
| GPUs | **2× NVIDIA TITAN V**, compute capability **7.0 (Volta, sm_70)** |
| Interconnect | PCIe (Titan V has **no NVLink**) |
| CUDA toolkit | 12.9 (V12.9.86) |
| libNVVM | 2.0 · nvJitLink 12.9 · libdevice `libdevice.10.bc` |
| llc / LLVM | 22.1.2 (rust nightly-2026-04-03) |
| rustc | 1.96.0-nightly (2026-04-02) |
| cuda-oxide | git `d989fa4` (v0.1.0 line) |
| Target arch | `sm_70`, auto-detected from device 0 |

## Results

| # | Capability | Result | Notes |
|---|---|---|---|
| 0 | f32 kernel (vecadd, no libdevice) loads & runs | ✅ **PASS** | baseline — basic kernels work on Volta |
| 1 | Device inventory (2 GPUs) | ✅ **PASS** | both Titan V, cc 7.0 |
| 2 | Embedded module load on sm_70 (libdevice-free) | ✅ **PASS** | |
| 3 | **FP64 arithmetic** (+ − × ÷) | ✅ **PASS** | worst relative error **0.0** over 256 elems |
| 4 | **FP64 shared memory + `sync_threads`** | ✅ **PASS** | block-level tile load/sync/neighbor-read |
| 5 | **Multi-GPU peer-to-peer** (dev0 ↔ dev1) | ✅ **PASS** | `can_access_peer` true both ways; **direct dev0→dev1 P2P copy verified over PCIe** |
| 6 | **FP64 libdevice math** (`exp`, `ln`, `powf`) | ✅ **PASS*** | worst rel err **3.35e-15** — *requires PR #101 backend + the `ltoir` loader + avoiding `Option<&mut>` accessors (see Diagnosis)* |

\* Initially failed; now passes once the three conditions in the Diagnosis are met.

## Diagnosis

FP64 libdevice math **works on sm_70** — but only after isolating and working
around three distinct issues. The investigation (toggling a single `sqrt()`, then
swapping loaders, then changing the slice accessor) nailed each one:

**(a) Stock cuda-oxide emits opaque-pointer NVVM IR that pre-Blackwell libNVVM
can't parse.** Any libdevice call (`sqrt`/`exp`/`ln`/`powf`) flips the pipeline
into NVVM-IR mode; the stock backend's opaque-pointer IR fails on the Titan V:
```
nvvmCompileProgram: 9 — parse expected type
```
This is **cuda-oxide issue [#98]**, and **PR [#101]** ("typed NVVM IR for
pre-Blackwell") is the fix. We pinned #101's backend (`CUDA_OXIDE_BACKEND` +
Cargo dep rev) and confirmed it now emits typed-pointer IR (`double*`, legacy
datalayout, `!nvvmir.version = {2,0,3,1}`). **Required.**

**(b) The libdevice path only works through the file-based `ltoir` loader, not
the embedded `#[cuda_module]` loader.** Libdevice kernels need
NVVM IR + libdevice → LTOIR → nvJitLink → cubin, which #101 implemented in
`cuda_host::ltoir` (reached via `load_kernel_module` + `cuda_launch!`, the
`manual_launch_libdevice` pattern). The embedded `kernels::load` path does **not**
do this and fails. So libdevice kernels must use the file-based loader. **Required.**

**(c) PR #101's typed-pointer conversion is incomplete — it misses a bitcast.**
With #101 + the ltoir loader, libNVVM got further and gave a precise error:
```
probe_fp64_math.ll (62, 43): parse '%v36' defined with type 'double*'
```
Line 62 is `insertvalue { i8, i8* } %v37, i8* %v36, 1`, but `%v36` is a `double*`
(from a GEP). #101 emits a typed `double*` into an `i8*` aggregate slot **without
the required bitcast** — legal in opaque-pointer mode (`ptr`), illegal in typed
mode. That `{ i8, i8* }` aggregate is what `DisjointSlice::get_mut` →
`Option<&mut f64>` lowers to. **Workaround:** write via `get_unchecked_mut(i)`
after an explicit `i < out.len()` bounds check, avoiding the `Option<&mut>`
aggregate. With that, the probe passes (worst rel err 3.35e-15).

> Issue (c) is a real **bug in PR #101** and a concrete gale upstream-contribution
> target: teach the typed-pointer export to insert `bitcast`s when a typed pointer
> feeds an `i8*` aggregate/store/call slot. Until then, the `get_unchecked_mut`
> kernel idiom is the workaround.

The intrinsics themselves are present (`__nv_exp/__nv_log/__nv_pow/__nv_sqrt/
sin/cos/tan`); `atan`/`atan2` are not yet mapped (open #77/#78) — avoid them.

[#98]: https://github.com/NVlabs/cuda-oxide/issues/98
[#101]: https://github.com/NVlabs/cuda-oxide/pull/101
[#69]: https://github.com/NVlabs/cuda-oxide/pull/69
[#77]: https://github.com/NVlabs/cuda-oxide/issues/77
[#78]: https://github.com/NVlabs/cuda-oxide/pull/78

## What this means for gale

**The whole foundation gale needs works on the Titan Vs today** — nothing is
hard-blocked:
- FP64 arithmetic is correct; shared-memory matrix-free kernels are viable.
- Multi-GPU P2P primitives (`cuda_core::peer::*`, `memory::memcpy_dtod_async`)
  **exist and work** — a direct dev0→dev1 copy succeeds over PCIe. The inter-GPU
  trace-exchange primitive is in hand. (Not "greenfield" as first assumed.)
- **FP64 libdevice math (`exp`/`ln`/`powf`) works** — including everything the
  log-conformation viscoelastic model needs — subject to the recipe below.

**The working recipe for libdevice (FP64 math) kernels on sm_70:**
1. **Build against PR #101's backend.** Pinned via `Cargo.toml` (dep `rev`) +
   `.cargo/config.toml` (`CUDA_OXIDE_BACKEND` → a local build of #101). Revert
   when #101 (and the bitcast fix) land upstream.
2. **Use the file-based `ltoir` loader**, not the embedded `#[cuda_module]` path:
   top-level `#[kernel]` + `cuda_host::load_kernel_module` + `cuda_launch!`.
3. **Avoid `Option<&mut>` slice accessors** in those kernels: write via
   `get_unchecked_mut(i)` after an `i < out.len()` check (workaround for #101's
   missing bitcast).

Pure-arithmetic / shared-memory kernels (no libdevice) have **none** of these
constraints — the ergonomic embedded `#[cuda_module]` + `get_mut` path works
directly, as `probe-sm70` shows.

**Implication:** Milestones 0–5 (incl. log-conformation viscoelasticity) are
unblocked on cuda-oxide today. The cost is the libdevice recipe above until two
upstream fixes land — which are exactly gale's contribution targets.

## Next actions

- [x] Pin #101's backend + crates; confirm FP64 math passes on sm_70
      (`probe-fp64-math` green, worst rel err 3.35e-15).
- [ ] **Decide on a permanent fork** (see below) vs. the current local pin.
- [ ] **Upstream contribution — fix (c):** patch `dialect-llvm` typed-pointer
      export to insert `bitcast`s when a typed pointer feeds an `i8*` aggregate/
      store/call slot, so `get_mut`/`Option<&mut>` kernels compile. Add to
      `export_test.rs`. This removes workaround #3.
- [ ] **Upstream contribution — validate #101:** comment real sm_70 evidence
      (this probe) on PR #101 to help it land; note the bitcast gap.
- [ ] Optionally raise the embedded-loader libdevice gap (issue (b)) upstream.
- [ ] `probe-fp64-math` is now the **regression gate**: it should stay green; if a
      cuda-oxide bump breaks it, the libdevice recipe changed.

## Current pin (how this box is set up)

- **Fork:** `github.com/ianrgraham/cuda-oxide`, branch **`pr-101-typed-nvvm`**
  (= PR #101 head `a315b72`, pushed from `pr-source` = mohamedsamirx). This is ours
  to carry our own follow-up fixes (e.g. the residual typed-pointer bitcast bug).
- `Cargo.toml`: `cuda-{device,host,core}` → `git =
  "https://github.com/ianrgraham/cuda-oxide.git", rev = "a315b72…"`.
- `.cargo/config.toml`: `CUDA_OXIDE_BACKEND` → `~/src/cuda-oxide-101/crates/
  rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so`, built from `a315b72`
  with gale's toolchain (byte-identical commit, so valid for the fork pin too).
- **Local fork checkout:** `~/src/cuda-oxide-101` on branch `pr-101-typed-nvvm`,
  remotes `origin` = your fork, `upstream` = NVlabs, `pr-source` = mohamedsamirx.
  Develop the bitcast fix here and `git push origin` to update the branch (then
  rebuild the backend and bump the `rev` in `Cargo.toml`).
- **The `cargo-oxide` CLI was *not* reinstalled** — #101 changed only that crate's
  README, and cuda-oxide's auto-fetch is hardcoded to clone NVlabs/main, so a
  fork-based `cargo install` wouldn't change behavior anyway. The
  `CUDA_OXIDE_BACKEND` override is the supported mechanism (backend.rs priority 1).
