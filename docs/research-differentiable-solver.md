# Differentiable gale: Enzyme autodiff through the cuda-oxide GPU path

> Status: research note, 2026-06-08. Verified deep-research pass (21 sources, 25 claims
> adversarially verified, 22 confirmed). Confidence tags: ✅ verified (≥2/3 vote) ·
> ⚠️ inference/domain-knowledge · ❓ open. This is a **feasibility study, not a commitment**.
>
> **UPDATE 2026-06-09 — the typed-pointer blocker largely DISSOLVES; see §9.** Direct inspection
> of the cuda-oxide fork shows it pins **LLVM 22** (nightly-2026-04-03) and is opaque-pointer-native
> internally — typed pointers are only a *final text-export shim* for pre-Blackwell libNVVM. So
> Enzyme would run on modern opaque-pointer IR, NOT the fragile typed-pointer zone §2/Obstacle-3
> feared. The revised gating issues are (a) Enzyme's LLVM ceiling is **21** today (cuda-oxide is on
> **22** — a one-version lag), and (b) cuda-oxide is a pliron + text-export backend (no in-memory
> `llvm::Module`), so insertion is via `opt`/LLVMEnzyme on emitted IR, not the rustc `std::autodiff`
> path. Read §1–§8 as the original survey; §9 is the corrected verdict.

## 0. Why this matters for gale

A differentiable viscoelastic DG-SEM solver unlocks three research capabilities that a
forward-only solver cannot:

1. **Rheological parameter inference** — fit Giesekus / FENE-P / Oldroyd-B parameters
   (relaxation time λ, mobility/anisotropy, extensibility b) to experimental flow data by
   gradient descent *through the solver*. This inverts the classic forward problem and points
   straight at the user's rheology background.
2. **Shape / topology optimization** of microfluidic geometries (adjoint w.r.t. boundary or
   IBM geometry).
3. **ML-in-the-loop closures** — learned constitutive corrections / SGS terms trained against
   the resolved solver.

The architectural bet: cuda-oxide compiles **Rust → LLVM-IR → PTX**, and **Enzyme** is an
LLVM-IR-level autodiff engine with demonstrated GPU support. In principle Enzyme slots into
gale's existing compile path as an LLVM pass before PTX codegen. This note tests that premise.

## 1. Verdict

**Feasible only with significant engineering effort — NOT feasible "now" / drop-in.** The
single load-bearing blocker is **LLVM typed-pointer compatibility** on the pre-Blackwell
sm_70 (Titan V / NVVM 7) path. The capability is real and demonstrated; the *specific*
cuda-oxide-on-sm_70-with-typed-pointers path is unproven and sits in a fragile version zone.

| Dimension | Finding | Conf. |
|---|---|---|
| Enzyme differentiates GPU device code at LLVM-IR level | ✅ First fully-automatic reverse-mode AD tool for GPU kernels (CUDA + ROCm); SC'21 Moses et al. | ✅ |
| Enzyme is an LLVM plugin → any LLVM frontend (incl. Rust) / any LLVM backend (incl. NVIDIA) | ✅ cuda-oxide IR→PTX is exactly the target class — a *candidate* insertion point | ✅ |
| CUDA support maturity | ✅ Officially "HIGHLY EXPERIMENTAL, in active development" | ✅ |
| Entry point granularity | ✅ `__enzyme_autodiff` works on `__device__` functions **only**, NOT `__global__` kernels | ✅ |
| DG-SEM-specific GPU overhead | ✅ DG benchmark: **18× overhead on NVIDIA** (vs 5.4× AMD) from register spill to global memory | ✅ |
| GPU/AD-specific preprocessing required to run at all | ✅ Without it: LULESH 2979× overhead, LBM exhausts GPU memory | ✅ |
| Typed-pointer LLVM zone | ✅ Fragile: LLVM 14 typed-default (works), LLVM 15 opaque-default (broke 67 Enzyme tests), LLVM 16 typed best-effort/untested, LLVM 17 typed removed | ✅⚠️ |
| Rust `std::autodiff` / `#[autodiff]` | ✅ Built on Enzyme; unstable nightly-only; RFC pending (#124509); needs custom `enzyme` toolchain | ✅ |
| `std::autodiff` shipping status | ✅ Fully upstreamed into rustc (H2 2025) but **not yet shipped/enabled on nightly** | ✅ |
| Rust-frontend GPU-kernel AD | ✅ Exists at LLVM level but **NOT yet exposed to Rust frontend**; blocked partly by LLVM's lack of multi-arch-target module support | ✅ |

## 2. The three concrete obstacles

### Obstacle 1 — `__device__`-only entry point ✅
Enzyme's `__enzyme_autodiff` differentiates `__device__` functions, not `__global__` kernels
("may be supported in the future"). gale cannot just point Enzyme at a launch kernel. The
documented pattern is to **differentiate the device function from inside a `__global__`
wrapper**, using Enzyme's augmented-forward / reverse split plus custom-derivative
registration for the pieces Enzyme can't see through. This is a structural design constraint
on how gale's kernels would have to be factored (pure `__device__` compute cores, thin launch
wrappers) — not a blocker, but it shapes the kernel architecture.

### Obstacle 2 — DG-SEM register-spill overhead ✅ (the performance risk)
This is the most gale-relevant data point in the whole pass. The SC'21 DG benchmark hit
**18× overhead on CUDA** specifically because reverse-mode "quickly exhausts the amount of
available registers and the CUDA assembler decides to spill a large number of registers into
global memory" — vs only 5.4× on AMD (more registers). gale is a **DG-SEM** solver, so this is
a direct hit, not an analogy.
- ⚠️ Transferability caveat: that benchmark was **Julia/CUDA.jl on RTX 2080/A6000**, not
  Rust/cuda-oxide on Titan V/sm_70. Titan V (Volta) register file is 256 KB/SM (64 K 32-bit
  regs) — register pressure on sm_70 could be better or worse; untested.
- Implication: reverse-mode through the *full* matrix-free DG operator may be too register-hungry.
  This is a strong argument for **forward-mode AD** for the parameter-inference use case (see §4),
  where the differentiated quantity is a handful of scalar rheological parameters.

### Obstacle 3 — typed-pointer LLVM version zone ✅⚠️ (the gating blocker)
Pre-Blackwell sm_70 / NVVM 7 implies **typed-pointer** LLVM IR. The Enzyme × LLVM-version
timeline:
- LLVM 14: typed pointers default — Enzyme works.
- LLVM 15: opaque pointers default — broke **67** of Enzyme's integration tests at the time
  (Enzyme#687, a 2022 snapshot).
- LLVM 16: "Opaque pointers enabled by default. Typed pointers supported on a best-effort
  basis only and not tested."
- LLVM 17: "Only opaque pointers are supported. Typed pointers are not supported."

So the typed-pointer path gale needs for sm_70 is exactly the path LLVM has been deprecating.
**Caveat:** Enzyme#687 is a point-in-time 2022 snapshot and Enzyme has since added
opaque-pointer support. The unresolved question (§5) is whether the LLVM version cuda-oxide
pins for sm_70/NVVM-7 typed-pointer emission is one a *current* Enzyme release still supports.

## 3. The Rust-frontend path is immature — use ClangEnzyme or hand-wire the pass instead

`std::autodiff` (`#[autodiff]`) is built on Enzyme but: unstable nightly-only, RFC unapproved
(#124509), needs a custom `enzyme` rustup toolchain that **isn't shipped on nightly yet** (as of
H2 2025), and — critically — **does not yet expose Enzyme's GPU-kernel differentiation to the
Rust frontend**. Exposing it is an explicitly experimental 2024-onward effort (the Manuel/Jed
two-approaches work referenced in `enzyme.mit.edu/rust/limitations.html`), hindered by LLVM's
lack of support for modules with multiple architecture targets.

⚠️ **Implication for gale:** do not plan on `std::autodiff` for *device-kernel* gradients.
The realistic paths are:
- **(a) Hand-wire Enzyme as an LLVM pass inside the cuda-oxide pipeline** (Enzyme acts on the IR
  cuda-oxide already produces, before PTX codegen). Most aligned with gale's architecture; most
  work.
- **(b) ClangEnzyme C/C++ route** for the differentiated device cores, called from Rust — proven
  but splits the toolchain.
- `std::offload` is a *separate* rustc PTX backend (not cuda-oxide) and as of 2025h2 could only
  move data, not launch kernels — tangential, watch but don't depend on.

## 4. Forward-mode as the pragmatic first target ⚠️

For **rheological parameter inference**, the differentiated inputs are a *small* number of
scalar parameters (λ, Giesekus α, FENE-P b, …). Reverse-mode's advantage (cheap gradient of
scalar-output w.r.t. many inputs) is wasted here, and reverse-mode is exactly where the 18×
register-spill DG overhead bites. **Forward-mode AD** (or tangent-linear, cost ∝ #parameters)
sidesteps the register-spill pathology and the `__global__`-vs-`__device__` checkpointing
complexity. For a few parameters, forward-mode through the solver is likely the cheaper, more
robust first capability — reverse-mode/adjoint is the right tool only once the optimization
variable becomes high-dimensional (shape/topology fields, ML weights).

❓ Not yet evaluated head-to-head — flagged as an open question for the IMEX-adjacent follow-up.

## 5. Riskiest unknowns (verify before any integration work)

1. ❓ **Which exact LLVM version does cuda-oxide pin for sm_70/NVVM-7 PTX, and does a current
   Enzyme release still support typed-pointer IR on it?** This is the decisive blocker. Nothing
   in the pass answered it directly — it's inferred from the LLVM timeline.
2. ❓ **Reverse-mode performance through gale's matrix-free DG-SEM kernels** given the 18× DG
   register-spill signal — performant enough for gradient descent, or must forward-mode be used?
3. ❓ **Checkpointing / forward-trajectory storage** for reverse-mode through PDE time-stepping
   (revolve/binomial checkpointing). No confirmed claim covered this; LBM GPU-memory exhaustion
   is the only adjacent signal. Matters a lot for adjoint-through-time of a transient viscoelastic run.
4. ❓ Which of the two 2024 Manuel/Jed approaches to Rust-frontend GPU-kernel AD was pursued and
   its current status.

## 6. Recommended first feasibility probe

Before any gale integration: **reproduce the official Enzyme CUDA-guide example** (ClangEnzyme
plugin differentiating a `__device__` function, sm_70, `-O2`) **on the exact LLVM version
cuda-oxide pins.** This is a one-day spike that answers the single load-bearing question — does
Enzyme run on typed-pointer IR at sm_70 — before committing to architecture. If that works,
the next probe is forward-mode AD through one small device compute core (e.g. the pointwise
Oldroyd-B relaxation update) and check the generated PTX + register count.

## 7. Caveats on this research

- **Time-sensitive:** the Rust-frontend story (`std::autodiff` shipping, `std::offload` kernel
  launch, the `enzyme` rustup component) is moving month-to-month; several claims are explicit
  snapshots ("as of 2024h2/2025h2") and a Jan 2026 update (PR #150071, "Add dist step for
  Enzyme") already reports further progress without enabling CI distribution.
- **No source tested the exact path:** nobody benchmarked Enzyme on cuda-oxide, on Rust→PTX with
  typed pointers, or on a Titan V (sm_70). The DG/GPU figures are Julia/CUDA.jl on RTX 2080/A6000.
- **Three claims were refuted** in verification — including one asserting cuda-oxide is
  "architecturally the right" Enzyme insertion point (1-2) — so do not over-rely on the strongest
  framing of cuda-oxide compatibility.
- The checkpointing-through-time-integrators and JAX/Diffrax/discrete-adjoint alternative legs
  of the brief **did not surface verified claims** — treated as open (§5).

## 8. Key sources

- Moses et al., **"Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via
  Enzyme,"** SC'21, doi:10.1145/3458817.3476165 — `papers.wsmoses.com/EnzymeGPU.pdf` (the GPU paper)
- Enzyme CUDA Guide — `enzyme.mit.edu/getting_started/CUDAGuide/` (`__device__`-only entry point)
- Enzyme Rust limitations — `enzyme.mit.edu/rust/limitations.html` (GPU-kernel AD not yet in Rust frontend)
- Enzyme#687 — typed/opaque pointer test breakage on LLVM 15
- LLVM 16 OpaquePointers docs; LLVM 17 release notes (typed-pointer removal)
- Rust `autodiff` unstable-book flag; rust#124509 tracking issue; compiler-team#611
- Rust project goals 2024h2 (Rust-for-SciComp), 2025h2 (finishing-gpu-offload)

## 9. RESOLVED (2026-06-09): the real Q3 — does the cuda-oxide LLVM path admit Enzyme?

Direct inspection of the fork (`/home/ian/src/cuda-oxide-101`) + Enzyme's CI, resolving the
"riskiest unknown" from §5.1. **The typed-pointer fear in §2/Obstacle-3 was based on a wrong
premise and largely dissolves.**

### 9.1 What the fork actually pins ✅ (source inspection)
- **LLVM 22**, via rust toolchain **`nightly-2026-04-03`** (`rust-toolchain.toml`; min LLVM 21 for
  TMA/tcgen05/WGMMA intrinsics — `crates/cargo-oxide/src/commands.rs`).
- **Opaque-pointer-native internally.** Typed vs opaque is a *target-driven export decision*, not the
  pipeline's pointer model: `crates/dialect-llvm/src/export/config.rs:82` →
  `major >= 10 (sm_100+/Blackwell) ⇒ OpaquePointers, else TypedPointers`. The `!nvvmir.version`
  shim is `[2,0,3,1]` (NVVM 7) for pre-Blackwell, `[2,0,3,2]` (NVVM 20) for Blackwell — emitted
  only at the **textual `.ll` export** step to satisfy CUDA 12.x pre-Blackwell libNVVM.
- **No `llvm-sys`/`inkwell`.** cuda-oxide uses `rustc_private` + a **pliron MLIR-in-Rust dialect** and
  **text-based LLVM IR export** — there is no in-memory `llvm::Module`.
- No Enzyme references anywhere in the fork yet.

### 9.2 What Enzyme actually supports ✅ (CI inspection)
- Enzyme's **integration CI (`ccpp.yml`) matrix = LLVM 15,16,17,18,19,20,21**; the core CI tests 15–16.
  So Enzyme is maintained against **LLVM ≤ 21 today**; **LLVM 22 is not yet in the matrix**.
- All of 15–21 are **opaque-pointer-era** (opaque default @15, typed removed @17). Enzyme's modern
  regime IS opaque pointers — exactly cuda-oxide's internal IR.

### 9.3 Corrected verdict
The blocker is **no longer "typed pointers"** (the internal IR is opaque LLVM 22). The two real issues:

1. **One-version LLVM lag (minor, time-resolves).** Enzyme tracks LLVM mainline tightly (15→21 all
   tested); 22 support is weeks-to-months away, OR pin a slightly older nightly that ships **LLVM 21**
   to match Enzyme *today*. There is a known transient "clang-20 + Enzyme crash" report (llvm#119917),
   so validate the chosen version rather than assuming.
2. **Architectural insertion (the actual engineering cost).** Because cuda-oxide emits *textual* IR via
   pliron rather than building an `llvm::Module`, the clean rustc `std::autodiff` path **does not apply**
   (that path needs `rustc_codegen_llvm`'s real module; cuda-oxide is a different backend). The realistic
   route is: emit the **opaque-pointer** `.ll` → run **LLVMEnzyme-21/ClangEnzyme** via `opt` to produce
   the differentiated IR → then the typed-pointer text shim → libNVVM/ptxas. I.e. Enzyme slots in
   **before** the `dialect-llvm` typed-pointer print, on opaque IR. Round-tripping pliron-exported text
   through `opt` and back is the integration work to prove.

### 9.4 Revised first probe
Supersedes §6. Two cheap spikes, in order:
1. **Version match:** confirm whether to target LLVM 21 (pin an older cuda-oxide nightly) or wait for
   Enzyme-22. One afternoon checking the Enzyme release tracker + cuda-oxide's nightly→LLVM mapping.
2. **`opt` round-trip:** take a cuda-oxide-emitted **opaque-pointer** `.ll` for one small `__device__`
   compute core (e.g. the pointwise Oldroyd-B relaxation update), run `LLVMEnzyme-21` via `opt` in
   forward mode, and confirm valid differentiated IR → PTX. This tests the real integration seam, not
   the (now-dissolved) typed-pointer question.

> Net: the differentiable direction is **more feasible than the 2026-06-08 verdict implied** — the
> scary blocker was a misread of where typed pointers live. It's now a tractable
> version-pinning-+-`opt`-integration problem, with the §2/Obstacle-2 reverse-mode register-spill cost
> (→ prefer forward-mode for parameter inference) the main *remaining* substantive risk.
