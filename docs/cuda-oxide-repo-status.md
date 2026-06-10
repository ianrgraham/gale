# cuda-oxide — Repository Status & Contribution Map

**A snapshot of the [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)
project for the `gale` team, with a gale-specific gap & contribution analysis.**

> **Why this doc.** `gale` is committed to cuda-oxide as its GPU toolchain and
> intends to **contribute upstream** where features are missing (see
> [`dg-gpu-fluid-simulation.md`](./dg-gpu-fluid-simulation.md) §9). This doc tracks
> the live state of the repo so we know what already exists, what's in flight, and
> where our effort is best spent.
>
> **Snapshot date: 2026-06-01** (data pulled directly from the GitHub REST API).
> The repo is **~6 weeks old and changing daily** — numbers below will drift. See
> [§9 How to refresh](#9-how-to-refresh) to regenerate.
>
> ⚠️ **2026-06-08 update — `v0.2.0` is out.** §1–§8 below describe the v0.1.0-era
> repo and are now partially stale. **[§10](#10-v020-released--fate-of-our-fork-patches)
> records what 0.2.0 changed and the fate of each of our 8 fork patches.** Bottom
> line: 0.2.0 is a rearchitecture (not a cherry-pick of our work); it obsoletes ~4
> of our patches but **still does not support pre-Blackwell typed pointers**, so the
> Titan V (sm_70) fork is still required. Left for later — no action taken yet.

---

## 1. Repo at a glance

| Field | Value |
|---|---|
| Full name | `NVlabs/cuda-oxide` |
| Description | "experimental Rust-to-CUDA compiler … compiles standard Rust directly to PTX — no DSLs, no foreign language bindings, just Rust" |
| Created | **2026-04-22** (≈6 weeks old at snapshot) |
| Last push | 2026-06-01 |
| License | Apache-2.0 |
| Default branch | `main` |
| Latest release | **v0.1.0** (the only release/tag) |
| Stars | **2,554** |
| Forks | 162 |
| Watchers | 16 |
| Primary language | Rust (≈4.6 MB; also Cuda, C, Python, Nix, Shell) |
| Topics | compiler-backend, cuda, gpu, heterogeneous-computing, hpc, nvidia, rust, async |

**Read:** young, high-visibility (2.5k stars in 6 weeks), single tagged release at
**v0.1.0** — consistent with the "alpha, expect breakage" framing in the docs.

---

## 2. Project health & governance

- **Velocity is high.** Commits land daily through the snapshot date; many merges
  in the last week (constant memory, Nix flake, closure-launch ABI, embedded IR).
- **Bus factor is a concern.** Contributions are dominated by a single maintainer,
  **@nihalpasham (137 of ~190 commits)**, with @josephglanville (13) and
  @mohamedsamirx (6) the next most active. A long tail of one-off contributors.
  For `gale`, this means: (a) responsive but potentially bottlenecked review;
  (b) an opportunity to become a recognized regular contributor quickly.
- **Healthy external interest.** 162 forks and a steady stream of community PRs
  (math intrinsics, codegen fixes, docs) — the project is accepting outside work.
- **Labels are well-organized** (good for finding work): `good first issue`,
  `help wanted`, `codegen`, `IR-lowering`, `intrinsics`, `host-apis`,
  `cuda-feature`, `miscompile`, `safety`, `interop`, `epic`, `tracking`, etc.

---

## 3. Issue / PR taxonomy (snapshot)

| Bucket | Count |
|---|---|
| Open issues (excl. PRs) | **16** |
| Open PRs | **19** |
| Closed issues (excl. PRs) | 21 |
| Merged PRs | 29 |
| Closed-but-unmerged PRs | 14 |
| **Total PRs ever** | 62 |

**Notable signal: a 19-PR open backlog vs 29 merged.** Many open PRs are marked
*ready* (not draft) and several are small fixes — review throughput, not
contribution supply, is the constraint. A `gale` contribution's time-to-merge will
depend on maintainer bandwidth; well-scoped, well-tested PRs that don't need much
review will move fastest.

---

## 4. Open issues (16)

Grouped by theme; `#n [Nc]` = issue number and comment count.

**Codegen / IR-lowering (the core compiler):**
- `#98 [1c]` **nvvmCompileProgram fails with "parse expected type" when libdevice
  calls (`__nv_expf`) trigger NVVM IR mode** — *directly relevant to gale*, see §6.
- `#76` Cross-file device-fn helper → "Type translation not yet implemented for
  RigidTy(Str)".
- `#58 [2c]` Mutable array element borrow in kernel.
- `#35 [2c]` Compilation fails for `z*z` where `z: num_complex::Complex32`.
- `#29` Kernel taking a reference to an application struct → illegal memory access.

**Intrinsics / math (FP coverage):**
- `#77` Add support for `{f32,f64}::atan{,2}` — *FP64 math gap, see §6*.

**Host APIs / build / tooling:**
- `#99` Owned async launches not cancellation-safe after first poll.
- `#93` Runtime CUDA library discovery ignores `CUDA_TOOLKIT_PATH`.
- `#87` `cargo oxide doctor` can't diagnose missing CUDA headers (panics first).
- `#72 [1c]` `ModuleNotFound` after moving a `#[cuda_module]` into a separate crate.
- `#68` cuda-oxide-book build missing `sphinx-autobuild`.

**Process / meta / perf:**
- `#96 [4c]` **Tracking epic: Tile-to-SIMT Interop** (cuda-oxide as a SIMT
  participant) — labels `tracking/epic/interop`.
- `#85` Classify rustc-codegen-cuda error examples by support status.
- `#84` README example count outdated.
- `#31` feat: introduce `oxide-signals` crate for observability.
- `#30 [6c]` Replace `HashMap` with `FxHashMap`? (label `perf`).

---

## 5. Open PRs (19)

**Math / intrinsics (FP coverage being filled incrementally):**
- `#78` Lower `f32::atan{,2}` / `f64::atan{,2}` via libdevice (fixes #77).
- `#66` Add `cvt_f16x2_f32` intrinsic (f32→f16x2 packing).
- `#63` Emit NaN float literals as hex bit patterns, not bare `nan`.
- `#62` Add `f32::max` / `f32::min` via libdevice `fmax`/`fmin`.

**Architecture / target support — *critical for gale's Volta Titan V (sm_70)*:**
- `#101` **Support typed NVVM IR for pre-Blackwell libNVVM targets** (fixes #98) —
  see §6.
- `#69` **feat: support pre-Ampere GPUs (sm_61+) and `std::sys::cmath`
  transcendentals** — see §6.

**Codegen / IR-lowering:**
- `#102` mir-lower: unit tests for memory op conversion.
- `#90` mir-importer: resolve arithmetic-trait `Output` on aggregate operands.

**Host APIs / memory safety:**
- `#100` (DRAFT) Keep in-flight async future results alive on drop (re: #99).
- `#92` Fix GPU memory leak on `DeviceBuffer` allocation failure.
- `#89` Use `malloc_sync` for `DeviceBuffer` allocations.
- `#95` Honor `CUDA_TOOLKIT_PATH` in runtime discovery (re: #93).
- `#88` Handle CUDA header discovery errors (re: #87).

**Docs / classification:**
- `#91` Update README example count (fixes #84); `#86` classify error examples;
  `#83`/`#81` add `sphinx-autobuild`; `#75`/`#73` docs fixes.

---

## 6. gale-relevant deep dive — what works, what's missing

This is the part that matters for the build. Mapped against `gale`'s needs:
FP64 physics, the Volta Titan V (sm_70), shared-memory matrix-free kernels, and
multi-GPU.

> ✅ **These questions are now answered empirically** — a probe was run on the
> actual 2× Titan V box (2026-06-01). See
> [`milestone-1-probe-results.md`](./milestone-1-probe-results.md). Summary:
> FP64 arithmetic, shared memory, multi-GPU P2P, **and FP64 libdevice math
> (`exp`/`ln`/`powf`) all work on sm_70** — nothing is hard-blocked. Libdevice
> math needs a 3-part recipe: pin **PR #101**'s backend (issue #98), use the
> file-based `ltoir` loader (not embedded `#[cuda_module]`), and avoid
> `Option<&mut>` slice accessors (a residual #101 missing-bitcast bug). The notes
> below are updated to match.

### 6.1 Will it even run on the Titan V (Volta, sm_70)? ⚠️ Track closely

This is the **single highest-priority open question**, and there is **active
in-flight work**:

- **`#69` (open PR): pre-Ampere sm_61+ support.** Its problem statement is
  telling: for pre-Ampere GPUs the pipeline "skips `llc` and emits NVVM IR,
  requiring nvJitLink at runtime — which fails with `ModuleNotFound` on pre-Ampere
  GPUs." It adds libdevice linking and cmath transcendentals across **all sm
  architectures**.
- **`#101` (open PR) + `#98` (open issue): typed NVVM IR for pre-Blackwell
  targets.** `#101` makes NVVM IR export target-aware: **typed-pointer** NVVM IR
  for pre-Blackwell targets (explicitly names `sm_75`), opaque-pointer for
  `sm_100`+. Uses the legacy NVVM datalayout / `!nvvmir.version = {2,0,3,1}`.

**Confirmed by the probe:** the Titan V is both pre-Ampere and pre-Blackwell
(sm_70). Basic kernels run fine. Stock libdevice math fails with
`nvvmCompileProgram: "parse expected type"` (issue **#98**); **PR #101** (typed
NVVM IR for pre-Blackwell) fixes that root cause but is **incomplete** — it omits a
bitcast when a typed pointer feeds an `i8*` aggregate slot, which still breaks
`Option<&mut>` (`get_mut`) kernels. With #101 pinned + the `ltoir` loader +
`get_unchecked_mut`, **FP64 libdevice math passes on sm_70** (rel err 3e-15).
Action:
1. ✅ Milestone-1 probe done (`docs/milestone-1-probe-results.md`) — libdevice math
   validated working on sm_70 via the recipe.
2. **Highest-leverage contributions:** (a) fix #101's missing bitcast in
   `dialect-llvm` typed-pointer export; (b) post sm_70 validation on #101 to help
   it land. We own the exact hardware → ideal validation partner.

### 6.2 FP64 — arithmetic likely fine, math library incomplete

- **Basic f64 arithmetic** rides on LLVM's NVPTX backend and is expected to work
  (no issue reports it broken).
- **The gap is libdevice math-function coverage**, which is being filled PR-by-PR:
  `#62` (f32 max/min), `#78`/`#77` (atan/atan2 incl. f64), closed-unmerged `#12`
  (`rsqrt_{f32,f64}`), and `#69` (transcendentals: `exp`, `tanh`, `sinh`, `erf`).
- `#98` shows the failure mode: a libdevice call (`__nv_expf`) can flip the
  pipeline into NVVM IR mode and then fail to compile — i.e. **transcendentals are
  currently fragile**, especially on non-newest targets.

**Updated by the probe:** the f64 intrinsics gale needs (`sqrt`/`exp`/`log`/`pow`)
are all *mapped* and **now verified working on sm_70** (rel err 3e-15) via the
§6.1 recipe (#101 backend + `ltoir` loader + `get_unchecked_mut`). The blocker was
never a missing mapping — it was the NVVM-IR dialect (#98/#101) plus #101's
residual bitcast gap, not per-function work. (`atan`/`atan2` are *additionally*
unmapped — #77/#78 — so avoid them regardless.)

### 6.3 Multi-GPU / peer-to-peer — primitives exist and WORK (corrected)

**Correction from the live probe:** the earlier "greenfield" read was wrong. While
there are still **zero issues/PRs** discussing multi-GPU, the host crate already
ships working P2P primitives — `cuda_core::peer::{can_access_peer,
enable_peer_access, disable_peer_access}` and `cuda_core::memory::memcpy_dtod_async`
("`src` and `dst` may reside on different devices if peer access is enabled").
The Milestone-1 probe **verified a direct dev0→dev1 P2P copy between the two Titan
Vs over PCIe** — `can_access_peer` is true both ways and the copy round-trips
correctly.

**Implication for gale:** the low-level inter-GPU trace-exchange primitive is
**already usable today** — multi-GPU is *not* blocked. What's missing is *higher
level* and *documentation*: ergonomic multi-context orchestration, a
device-pointer→MPI/NCCL handoff path (the PyFR pattern, `dg-gpu-fluid-simulation.md`
§6), and any docs/examples at all. Those remain good `gale` contributions, but the
foundation is in place.

### 6.4 What's already in place (use directly)

Confirmed present from merged PRs / docs:
- **Shared memory + barriers** (`SharedArray`/`DynamicSharedArray`, `sync_threads`)
  — basis for matrix-free operator kernels.
- **Warp intrinsics** (shuffle, `warp_reduce`) and **scoped atomics**.
- **Constant memory** — `#82` merged (`#[constant]` / `Constant<T>`); resident
  reference-element operator matrices can live here.
- **Streams / async** (`cuda-async`) — basis for compute/communication overlap.
- **Pinned host buffers** (`#42`, merged) — fast H2D/D2H staging.
- **Typed kernel launch + embedded kernels** (`#26`, `#44`, merged).
- **`globaltimer` intrinsic** (`#52`, merged) — useful for in-kernel timing.
- **Auto-detect local GPU target** for `cargo oxide run` (`#39`, merged).

---

## 7. Closed / merged highlights (context)

**Merged (29 total) — what's been built recently:**
- `#82` CUDA constant memory (`#[constant]`/`Constant<T>`).
- `#60`/`#26` typed launch + closure-launch ABI; `#44` embedded IR/binary payloads.
- `#42` pinned host buffers; `#52` `globaltimer`; `#55` `llvm.addressof` exports.
- `#27`/`#41`/`#64` codegen correctness (miscompiles→hard errors, arch handling,
  `step_by` lowering); `#19` `DeviceCopy` bound on transfers.
- `#48` Nix flake; `#25` Docker dev image; `#1`/`#2` security/CI hardening.

**Closed issues (21) — resolved pain points:** constant memory (`#71`),
tuple-returning device fns ICE (`#79`), `SharedArray` undefined-SSA lowering
(`#54`), typed-launch ABI mismatch (`#61`), `index_2d` soundness (`#34`),
several build/env discovery issues (`#16`, `#49`, `#36`, `#50`).

**Closed-unmerged (14) — superseded/declined, worth knowing:**
- `#11` `mma.sync`/`ldmatrix.x4` for sm_75+ tensor cores (verified on sm_120) —
  *tensor-core path was proposed but not merged*; relevant if gale ever wants MMA,
  though it needs sm_75+ (Titan V is sm_70, so no tensor cores anyway).
- `#45` "Fix 37 codegen bugs surfaced by porting vanity-miner-rs" — a large
  codegen-bug haul that didn't land as-is; signals the codegen still has sharp
  edges under real workloads.
- `#12` `rsqrt_{f32,f64}` — math intrinsic, not merged (so still missing).

---

## 8. Contribution roadmap for gale

Ordered by leverage, mapping repo state → `gale`'s needs:

| Priority | Contribution | Why / status | Effort |
|---|---|---|---|
| **P0** | **Validate + help land PR `#101`** (typed NVVM IR for pre-Blackwell) | **THE blocker** — probe proved libdevice math fails on sm_70 (#98); fix exists, we have the HW to validate it | Med (validate) → High (if fix needs work) |
| **P1** | **Multi-GPU: ergonomic multi-context + device-ptr→MPI/NCCL handoff** (PyFR pattern) | Low-level P2P **already works** (probe-verified); higher-level orchestration + docs are missing | Med–High |
| **P2** | Docs/examples for the existing P2P primitives | They work but are undocumented | Low |
| **P2** | Codegen robustness fixes we hit in DG kernels | Codegen has sharp edges (`#45`, `#76`, `#35`, `#29`) | Var |
| **P2** | Async cancellation-safety (`#99`/`#100`) if we use `cuda-async` heavily | In flight | Low |
| **P3** | Docs / example classification (`#84`/`#85`/`#86`) | Easy goodwill, `good first issue` territory | Low |

### 8.1 Fork watchlist — what to cherry-pick / track / avoid

Now that we carry `ianrgraham/cuda-oxide@pr-101-typed-nvvm`, here's the triage of
the **19 open PRs / 16 open issues** through the gale lens (reviewed 2026-06-01).

**Cherry-pick candidates (clean, no #101 overlap):**
- **PR #90** *(fixes #35)* — resolves operator-trait `Output` (`Mul`/`Add`/…) for
  **aggregate operands** (Complex, user numeric structs), in
  `mir-importer/translator/types.rs`. **No overlap with #101.** Pull this the
  moment gale defines operator-overloaded tensor/complex kernel types (likely for
  conformation tensors). Until then, track.
- **PR #63** — emit NaN/inf float literals as hex bit patterns. Codegen-correctness
  hygiene; matters if a kernel uses `f64::NAN`/`INFINITY` constants (e.g. reduction
  seeds). Small, low-risk.
- **PR #92 / #89** — `DeviceBuffer` alloc-failure leak fix / `malloc_sync`.
  Robustness for our many-buffer allocations; `cuda-core` only, low conflict risk.
- **PR #62** — `f32::max`/`min` via libdevice `fmax`/`fmin`. gale wants the **f64**
  versions too — extend it ourselves when needed.

**Reconcile later (conflicts with #101):**
- **PR #69** — pre-Ampere sm_61+ **and** `std::sys::cmath` transcendentals
  (`tanh`/`sinh`/`erf`) + math examples. **Overlaps #101 in
  `mir-importer/src/pipeline.rs`** and is itself `dirty` vs main. This is the *other
  half* of the pre-Blackwell libdevice story; if gale needs `tanh`/`sinh`/`erf`
  (some closure models), we merge #69 onto our branch and resolve `pipeline.rs` by
  hand. `exp`/`ln`/`pow`/`sqrt` we already have working, so not urgent.
- **PR #78** *(fixes #77)* — `atan`/`atan2` (f32/f64). Touches
  `mir-lower/.../call.rs` + `dialect-mir/rust_intrinsics.rs` (minor overlap with
  #69's area). Pull only if a kernel needs `atan`.

**Codegen landmines — design around these until fixed (no merged fix yet):**
- **#58** — writes through `get_mut()` into a nested `[f32; N]` array element are
  **silently dropped** (compiles, wrong result); the `for e in a { *e = … }` form
  works. *Directly threatens DG matrix-free kernels that write per-element DOF
  arrays.* Related to the same `get_mut`/`Option<&mut>` aggregate area as the typed-
  pointer bitcast bug we already hit — fix candidate for our branch.
- **#29** — passing `&SomeStruct` to a `#[kernel]` → runtime
  `illegal memory access`; arg-count mismatches also slip past the build. *gale will
  pass mesh/param structs* — prefer by-value `KernelScalar` / explicit scalar args
  until resolved.
- **#76** — cross-file device-fn helpers hit
  `Type translation not yet implemented for: RigidTy(Str)`. *gale will split device
  helpers across files* — keep an eye out; the trigger involves `&str`/format use.

**Irrelevant to gale (skip):** tensor-core/Hopper/Blackwell (`#96`, closed `#11`),
`f16` packing (`#66`), observability crate (`#31`), compiler-internal perf (`#30`),
async cancellation (`#99`/`#100`, unless we lean on `cuda-async`), and all
docs/tooling/env-discovery PRs (`#91/#83/#81/#75/#73/#86/#88/#87/#95/#102/#84/#68/#85`).

**Net:** nothing else is *required* right now — our probes pass. The first fix to
land on our fork branch is **our own typed-pointer bitcast fix**; **#90** is the
next clean pull once tensor types appear; **#69** is the real reconciliation job
if/when we need `tanh`/`sinh`.

> **Scoped & validated (2026-06-01):** see
> [`cuda-oxide-codegen-notes.md`](./cuda-oxide-codegen-notes.md). The bitcast fix
> is a small, self-contained change in `dialect-llvm/export/ops.rs`
> (`emit_insert_value` + a coercion helper). **#58 was reproduced and shown to be a
> *separate* bug** — a `mir-lower` place-lowering defect on the PTX path (writes go
> to a stack copy), not the text-exporter bitcast issue. They must be fixed
> independently. Both are kept as regression gates: `probe-fp64-math` (#101/bitcast)
> and `probe-nested-write` (#58).

**Workflow note:** open an issue describing the gale use case *before* large PRs
(multi-GPU especially) to align with @nihalpasham's direction — the `epic`/
`tracking` labels show the maintainer thinks in coordinated initiatives (e.g. the
`#96` tile-to-SIMT epic). Small math/codegen fixes can go straight to PR.

---

## 9. How to refresh

No `gh` CLI on this machine; use the REST API directly (unauthenticated is fine —
60 req/hr, this snapshot used ~10). To regenerate the counts and lists:

```bash
R=NVlabs/cuda-oxide
# Summary
curl -s "https://api.github.com/repos/$R" | jq '{stars:.stargazers_count, forks:.forks_count, open_issues_count, pushed_at}'
# Open issues (exclude PRs)
curl -s "https://api.github.com/repos/$R/issues?state=open&per_page=100" \
  | jq -r '.[]|select(.pull_request==null)|"#\(.number) \(.title)"'
# PRs with merge status
curl -s "https://api.github.com/repos/$R/pulls?state=all&per_page=100" \
  | jq -r '.[]|"#\(.number) [\(if .merged_at then "merged" elif .state=="open" then "open" else "closed" end)] \(.title)"'
```

For an authenticated higher-rate-limit pull (and easier scripting), install `gh`
and run `gh issue list` / `gh pr list -R NVlabs/cuda-oxide --state all`.

> **Caveats.** Counts are a 2026-06-01 snapshot of a fast-moving repo; the
> per-PR/issue *interpretations* in §6 are read from titles + a few fetched bodies
> (`#69`, `#98`, `#101`), not a full reading of every thread. The sm_70 and FP64
> conclusions should be **confirmed empirically** by gale's Milestone-1 probe on
> the actual Titan Vs — the repo tells us what's being worked on, not what passes
> on our exact hardware.

---

## 10. v0.2.0 released — fate of our fork patches

**Reviewed 2026-06-08** by diffing our fork branches against the actual `v0.2.0`
git tree (more authoritative than release notes). Source of truth: a local
`git fetch upstream --tags` in `/home/ian/src/cuda-oxide-101`.

### 10.1 State of play

- Upstream tagged **`v0.2.0`** (commit `faea395` bumps the version). Our two fork
  branches — `fix/typed-pointer-insertvalue-bitcast` (8 patches) and
  `pr-101-typed-nvvm` (1 patch) — both branch from merge-base `1f38440` and are
  **60 commits / one full minor release behind** `upstream/main`.
- **0.2.0 is a rearchitecture, not a cherry-pick of our work.** `git cherry`
  reports all 8 of our patches as still-absent by patch-id, but upstream re-did
  several of them independently under their own commits. The three structural moves:
  1. **`dialect-llvm` crate → renamed `llvm-export` + migrated onto upstream
     `pliron-llvm`** (PR #114, commits `838949c` + `a2effe1`). This is why our 4
     `dialect-llvm` patches no longer apply — that crate is gone by that name/shape.
  2. **New `oxide-artifacts` crate** + real cross-crate kernel support in
     `rustc-codegen-cuda/src/collector.rs`.
  3. **New `cuda-core::peer` module** (`can_access_peer` / `enable_peer_access` /
     `disable_peer_access`) + `memcpy_dtod_async` cross-device copy.

### 10.2 Per-patch fate

| Our patch | 0.2.0 status | Action |
|---|---|---|
| `a315b72` typed NVVM IR for pre-Blackwell | ❌ **Still unsupported.** `llvm-export/src/export/config.rs:88-89` says verbatim: *"Currently supports NVVM 20 dialect (Blackwell+, opaque pointers). NVVM 7 dialect (pre-Blackwell, typed pointers) is not yet supported."* Titan V = sm_70 = pre-Blackwell. | **Keep** — re-author onto `llvm-export` |
| `b787932` bitcast before `insertvalue` (typed) | ❌ Same — part of the unsupported typed-pointer path. | **Keep** |
| `8e8d71a` default unknown targets → typed ptrs | ❌ Same. | **Keep** |
| `eb2eab2` NaN constants as IEEE hex | ✅ **Fixed upstream** — commit `03763eb` / PR #116 / #63, identical fix. | **Drop ours** |
| `09a2e25` libm math → libdevice (broad) | ⚠️ **Partial.** Upstream wired atan/atan2/sin/cos/tan/exp/exp2/log/log2/log10/pow/sqrt/fma/min/max via `CALLEE_*` placeholders. **Still missing** the set our patch added: `acos, asin, sinh, cosh, tanh, cbrt, hypot, expm1, log1p, {a}sinh/cosh/tanh`. | **Keep the delta**, re-expressed via the new placeholder mechanism in `dialect-mir/src/rust_intrinsics.rs` |
| `965c436` `memcpy_peer_async` (P2P) | ✅ **Superseded by a cleaner API** — `peer::enable_peer_access` + `memcpy_dtod_async` (memory.rs:163, "may reside on different devices if peer access is enabled"). | **Drop, migrate** |
| `02129b8` embedded artifacts cross-crate link | ✅ **Superseded** by the `oxide-artifacts` crate + collector cross-crate support. | **Drop** |
| `0c65145` create artifact dir before `.ll` | ✅ Folded into the reworked `device_codegen` pipeline. | **Drop** |

### 10.3 Recommendation (deferred)

Rebase the fork onto `v0.2.0`. That collapses our delta from **8 patches → ~2
areas**: (a) the pre-Blackwell typed-pointer trio, **re-authored against the new
`llvm-export`/pliron-llvm internals** (non-trivial — *not* a clean `git rebase`,
budget real time), and (b) the missing transcendental math entries (`acos`/`asin`/
`sinh`/`cosh`/`tanh`/`cbrt`/`hypot`/…). Everything else (NaN, P2P, artifacts,
dir-creation, the common math fns) comes for free and can be deleted.

`pr-101-typed-nvvm` is the load-bearing branch — it maps to upstream **PR #101**,
which is *still the open gap* for pre-Blackwell support. We own the only sm_70
hardware validating it, so pushing that PR remains the highest-leverage upstream
contribution (consistent with §8 P0).

**Status: noted, no code changes made.** Pick up when we next touch the toolchain.
