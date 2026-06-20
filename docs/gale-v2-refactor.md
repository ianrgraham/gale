# gale v2 — refactor & consolidation plan

**Status:** design basis (2026-06-20). This document is the authoritative reference for the v2
consolidation: turning the accumulated GPU/CPU optimization work into user-friendly building blocks
exposed through `src/sim/`, *without* losing the measured performance edge. It answers four scoping
questions (portability/autotuning, the Sim/Device design, perf tooling, and the persisted benchmark
ledger) and lays out the phased plan. A companion skill (`.claude/skills/gale-v2-refactor`) carries
the working-discipline rules distilled from the optimization sessions.

Hardware of record for every number below: **NVIDIA Titan V (sm_70, Volta, native FP64, HBM2
~0.65 TB/s, ~4.5 MB L2, 80 SM)** unless stated otherwise. Every benchmark is stamped because the
*balance* between kernels is hardware-specific (see §1).

---

## 0. Where we are (the honest current state)

What we have, from the optimization passes:
- A **fast DG-SIPG operator** (`operator_arith`): the matvec restructured to the critical-path floor.
- A **device-resident elliptic solver**: hp-multigrid (h-coarsen at order p) + a 4th-kind Chebyshev
  smoother, run as a device-resident CUDA while-graph — now the **conforming default** (~1.5× the old
  p-MG+Jacobi). SBM (shifted-boundary) and NC (non-conforming/AMR) variants exist.
- **Device-resident flow integrators** (`GpuDualSplitting`, `GpuViscoelasticDualSplitting`) that
  *do* implement `gale::sim::StateIntegrator`, and a more ambitious fully-resident `GpuResidentNs/Ve`.

What is **not** consolidated (the gap this refactor closes), per the `src/sim/` survey:
- `Device` (the `Cpu | Cuda | MultiGpu` enum) is **metadata only** — `Simulation::run` never reads it.
  To run on GPU today you must *manually* construct `GpuDualSplitting` and `set_integrator` it.
- The fully-resident while-graph path (`GpuResidentVe`) is **not a `StateIntegrator`** and is unreachable
  from `Simulation`. It's a standalone struct + bin.
- GPU stage-hooks (penalization/IBM), GPU AMR, and multi-GPU halo all exist as kernels but are **not
  wired** to the `StateStageHook` / `Updater` / `DomainDecomposition` seams.
- `Term`/`StateTerm` are **host closures**; the GPU integrators bypass them and build their RHS in-kernel.

The framework is logically sound; the wiring is missing and the residency story is two-tiered (below).

---

## 1. GPU portability & autotuning (future NVIDIA parts)

**Thesis: the algorithms port; the magic numbers don't. ~80% of cross-GPU adaptation is autotuning a
small set of knobs; ~20% is a *measured* investigation of 2–3 new algorithmic levers the newest parts
unlock. We are in a good algorithmic spot — not a rewrite.**

### 1.1 What ports unchanged (algorithm-level, hardware-independent)
- **The hp-MG + Chebyshev structural win is an *iteration-count* property** (28→22 PCG iters). That ~1.5×
  is the same on any GPU; only the per-iteration cost re-scales.
- **The dependency-restructured matvec** (balanced-tree FP64 contractions, hoisted neighbour loads,
  arithmetic neighbours) shrinks the *critical path* — useful whether the kernel is BW- or latency-bound.
  nvcc/LLVM still won't reassociate FP64, so the manual restructuring stays the lever on every part.
- **Device-residency** (whole step as a CUDA graph, the loop a while-graph) matters *more* as GPUs get
  faster — the host falls further behind.

### 1.2 What shifts on A100 / H200 / B200 (why we autotune)
| axis | Titan V (sm_70) | A100/H200/B200 | consequence |
|---|---|---|---|
| HBM bandwidth | ~0.65 TB/s | ~2 / ~4.8 / ~8 TB/s | the BW-bound matvec likely flips to **latency/occupancy-bound** |
| L2 cache | ~4.5 MB | ~40–50 MB | more of `u` stays resident → re-tiling pays off |
| SM count | 80 | 108 / 132 / ~160+ | the grid-limited **transfers get worse** (harder to fill) |
| FP64 tensor cores | none | yes (A100+) | a **new** path for the DG contractions |
| consumer FP64 (5090) | n/a | 1/64 rate | **mixed precision becomes mandatory** |

Net: the *dominant kernel* and the *optimal config* both move. The transfers, already grid-limited
(1024 coarse elements can't fill 80 SMs at any block size — measured), get more underfilled on 108–132
SMs; the lever there becomes *bigger problems* or *fuse transfer+smoother*, not block tuning.

### 1.3 The autotuning levers (ranked by observed impact)
1. **Block size / elements-per-block** — the warp-fill ↔ grid-size tradeoff; the biggest single knob
   (it's what made the transfer 2.2× and what's grid-capped on Titan V).
2. **Mixed-precision toggle** (FP32 vs FP64 smoother / matvec intermediates) — ~1.5× on Titan V (kernel,
   validated), iteration count holds; the *right* choice is very hardware-dependent.
3. **Coarse-iteration count + Krylov tolerance** (CFIX, 1e-9→1e-6) — the work-vs-iteration tradeoff.
4. **hp hierarchy balance + coarse-grid floor** — the structural knob (how far h before p).
5. **Graph vs no-graph** — only in launch-bound regimes.

An autotuner needs these to be *swappable*: abstract the **scalar type** (FP64 / FP32 / double-double
for FP64-weak parts) and the **launch config** so there's something to turn. This is a v2 design goal,
not a bolt-on.

### 1.4 Genuinely new algorithmic levers (investigate + measure — do NOT assume)
- **FP64 tensor cores (A100+).** The DG tensor contractions (D-matrix applications) are small dense
  matmuls that could map to FP64 tensor cores — which *don't exist on Volta*. Most likely "new hardware
  unlocks a new algorithm." Prototype and measure before believing it.
- **Mixed-precision iterative refinement.** Solve in FP32/TF32, refine in FP64. We validated the FP32
  smoother (~1.5×) but the structural win superseded it; it's still on the table, **mandatory** for
  consumer FP64 (5090 @ 1/64), and a throughput win on A100+.
- **Communication-avoiding / pipelined CG (s-step).** Irrelevant single-Titan-V (our dots are
  device-resident and cheap); the lever once **multi-GPU** (the dot sync dominates). Parks with the
  multi-GPU roadmap (`docs/research-multi-gpu-elliptic.md`).

### 1.5 Verdict
Core (hp-MG + Chebyshev + restructured matvec + residency) is sound and portable. New-hardware work is
mostly **autotuning** (block/epb, precision, coarse-iters, tolerance) + **two measured investigations**
(FP64 tensor-core contractions; mixed-precision refinement). Design for swappable scalar type + launch
configs so the autotuner has knobs.

---

## 2. The Sim ↔ Device consolidation design

### 2.1 The core insight: three residency *levels*, two execution *models*
The `StateIntegrator` seam (`step(&mut state, hook)` per call) and the production while-graph residency
are **fundamentally different execution models**. The design must name and support both:

- **Level 0 — CPU.** Host loop, host state. The correctness oracle. (`DualSplitting`, `Mol`, …)
- **Level 1 — GPU, host-orchestrated.** Device-resident *state*, but the **host calls `step()` per
  step**. This is what `GpuDualSplitting` is today: it plugs into `Simulation::run`'s loop and keeps
  fields on-device, but the host drives the loop. Flexible (host updaters/writers can run between
  steps) — *but any per-step readback is a RULE ZERO violation*, so host hooks that need state are poison.
- **Level 2 — GPU, fully device-resident (the production bar).** The whole step is a captured CUDA
  graph and the whole time loop is a device-driven while/conditional graph; the host does **no per-step
  work and issues no per-step sync**. This is the `GpuResidentVe/Ns` target and the standing mandate
  (`docs/plan-device-resident-production.md`, the `gale-gpu-perf` skill). It **cannot** be a per-step
  `StateIntegrator` — the loop itself is on the device.

**A Level-1 sim is a prototype; a Level-2 sim is ready for parameter sweeps.** The v2 job is to make
**Level 2 reachable through the Sim abstraction**, and to make the level a *consequence of the Device +
the spec*, not a manual rewrite.

### 2.2 The design: Sim-as-spec, Device-as-compiler, dual-impl blocks
Separate **what** (the physics) from **how** (the Device-chosen execution).

1. **`Sim` is a declarative spec.** Fields + the dynamics (structured integrator kind: `DualSplitting`,
   `Viscoelastic`, …) + stage-hooks (limiter, penalization) + updaters (AMR) + writers (I/O) + physics
   params. No execution decisions baked in.

2. **Building blocks are dual-impl.** Each `Term`/`StageHook`/`Updater`/`Writer` carries a CPU impl and,
   *optionally*, a **device-resident** impl (a kernel + launch wrapper that can be captured into a
   graph). A block declares its residency capability. The closure-based `Term` is CPU-only by
   construction → it *caps a sim at Level 1*. Reducing closure-only blocks (or giving them kernel twins)
   is what lets a sim reach Level 2.

3. **`Device` becomes a compiler, not a tag.** Add `Device::plan(spec) -> ExecutionPlan`:
   - `Cpu` → host loop, CPU impls (Level 0).
   - `Cuda` → the **highest level the spec supports**: if every per-step block has a device-resident
     impl ⇒ capture the step graph + while-loop (Level 2); else ⇒ host-orchestrated device-resident
     state (Level 1), and host-only blocks are forced to I/O boundaries (every N steps, async) so they
     never sync mid-loop.
   - `MultiGpu` → Level 2 + the P2P halo (the `DomainDecomposition` already builds the partition).

4. **Device-aware integrator factory.** `sim.device(Cuda{0}).run(n)` resolves the right backend
   (`DualSplitting` vs `GpuDualSplitting` vs the resident `GpuResidentVe`) from `(spec, device)` — no
   manual `GpuDualSplitting::new`. This is the user-facing win: *one spec, pick a device, get the fastest
   correct path.*

5. **A `ResidentIntegrator` seam for Level 2.** A second trait —
   `run_resident(&mut self, state, nsteps, io_every, writers)` — owns the whole loop (captures the
   while-graph, replays N steps, reads back only every `io_every` for the writers). `GpuResidentVe`
   implements *this*, not `StateIntegrator`. The Device picks `StateIntegrator` (Level 1) vs
   `ResidentIntegrator` (Level 2) based on spec device-completeness.

```
   Sim (spec) ──device(D)──▶ Device::plan ──▶ ExecutionPlan
                                              ├─ Cpu          → host loop  · StateIntegrator (CPU)
                                              ├─ Cuda L1      → host loop  · StateIntegrator (GPU, resident state)
                                              └─ Cuda L2      → while-graph · ResidentIntegrator (GPU, self-driving)
```

### 2.3 The feature flag & the cuda-oxide contract
- A **`gpu` cargo feature** (recommend **default-on**, matching "GPU is first-class") gates the optional
  `gale-gpu` dependency and every `Cuda`/`MultiGpu` arm. Without it: CPU-only, no cuda-oxide needed,
  ordinary `cargo build`.
- With `gpu`: the user must build/run through **cuda-oxide** (`cargo oxide build/run`) exactly as they'd
  need nvcc for CUDA. Document this as a hard requirement (it is — the kernels are NVVM-IR JIT-linked at
  runtime; see the `--arch` gotcha in the skill). `Device::Cuda` on a CPU-only build is a clear compile
  or runtime error, not a silent CPU fallback.

### 2.4 What this refactor touches (from the survey)
- `src/sim/device.rs` — add `plan()` / integrator resolution (keep the enum).
- `src/sim/simulation.rs` — `run()` consults the plan; add the `ResidentIntegrator` dispatch path.
- `src/sim/{ibm,amr,stagehook}.rs` — give the hooks/updaters device-resident twins (or a backend enum)
  so they're capturable; today they're CPU-only and the GPU twins live unwired in gale-gpu.
- `gale-gpu/src/flow.rs` — `GpuDualSplitting` already implements `StateIntegrator` (Level 1); add the
  `ResidentIntegrator` impl for the resident path.
- `gale-gpu/src/resident.rs` — promote `GpuResidentVe/Ns` behind `ResidentIntegrator`.
- `Cargo.toml` (gale) — optional `gale-gpu` dep behind `gpu`.

### 2.5 The residency contract (non-negotiable, carried from the perf mandate)
Any `Cuda` execution must be device-resident with **no per-step host↔device sync** (RULE ZERO). Level 1
is acceptable only when no per-step block reads device state back to the host; otherwise the plan must
demote those blocks to I/O cadence or refuse. The autotuner/validator must check (nsys) that
memcpy/sync counts do **not** scale with steps×stages. This check is part of "is this sim production".

---

## 3. Standardized performance measurement (zero user effort)

Goal: any sim/example can be measured by flipping one switch, emitting a comparable triplet, with the
profiling gate built in — no per-sim plumbing.

### 3.1 The triplet, always
**ms/step · ms/kernel (the hot kernel) · ms/iteration (the solver)** — plus the RULE-ZERO check. These
three catch different regressions (a slower step could be more iterations, a slower kernel, or lost
residency; you can't tell from ms/step alone).

### 3.2 The harness (in the crates, not the bins)
- A `bench` module in `gale-gpu` (and a CPU analogue) exposing a `Profiled<R>` wrapper / `run_benched`
  entry that any `Simulation` or resident integrator can take, switched by **`GALE_PROFILE=1`** (env, so
  examples need zero code change): wall-clock ms/step, the device-resident scalar counters already in the
  solvers (iteration counts), and an optional nsys/ncu launch hook.
- **One canonical invocation**, documented once: build with `--arch sm_70` (or set `CUDA_OXIDE_TARGET`),
  profile the **bare** `target/release/<example>` (not through `cargo oxide run`), `RVP_NOGRAPH=1`
  (or the generalized `GALE_NOGRAPH=1`) to take a solve kernel **out of the while-graph** for ncu — since
  **ncu cannot profile kernels inside conditional graphs** (its numbers are artifacts; use nsys
  `--cuda-graph-trace=node` for the in-situ breakdown, ncu out-of-graph for SOL/occupancy). This gotcha
  is the single most expensive measurement trap we've hit — bake it into the tool, don't leave it to memory.
- The gate from the `gale-gpu-perf` skill (RULE ZERO, the bound, before/after, correctness) becomes a
  `just bench` / `cargo xtask bench` target that runs the standard set and prints the triplet table.

### 3.3 Layout after the refactor
- **`examples/`** — every simulation example (the current `traj-*`, demo sims). One `Device` line picks
  CPU vs GPU. `GALE_PROFILE=1` turns on measurement.
- **`benches/`** — the standardized regression benchmarks (the §4 baselines), the perf-CI surface.
- **`tests/`** — the correctness checks (the current `*-check` bins: VE-vs-host, NS-vs-analytic, etc.).
- **`src/` + `gale-gpu/src/`** — *all* the performance-critical CPU & GPU routines. No perf logic in
  examples/tests; they only *select* and *measure*.

---

## 4. Persisted findings & benchmark ledger (regression baselines)

These are the load-bearing results to defend through the refactor. **Numbers are Titan V / sm_70 / p=3
unless noted.** Treat each as a regression baseline; a refactor that moves one >~5% without an
explanation is a bug.

### 4.1 Kernel-level
| kernel / metric | before | after | lever |
|---|---|---|---|
| DG-SIPG matvec (`operator_arith`, 256²) | 168 µs (`operator_fused`) | **45 µs** (≈ C++ 44) | balanced-tree FP64 contractions, hoisted neighbour loads, arithmetic neighbours, L1 row/col (no u-stage); 80→56 regs, 36→51% occ, BW-bound at ~76% DRAM. **`get_unchecked` is a measured no-op** (LLVM already elides). |
| h-transfer `h_restrict` (64²→32², HMG) | 27.1 µs (16-thread blocks, ~5% SoL, 50% occ) | **12.4 µs** (~2.2×) | elements-per-block (full warps) + `pq` staged in shared. *Grid-limited* beyond this (1024 coarse elems can't fill 80 SM). |
| FP32 vs FP64 smoother (`op_cheby`, microbench) | FP64 25.6 µs @128² | **FP32 17.1 µs** (~1.5×) | mixed precision; BW halved. Iteration count holds (CPU f32 test 16→16). *Validated, not in production* — the structural win superseded it; revisit for FP64-weak parts. |

### 4.2 Solver-level (iteration counts — language-independent)
| config | iters @64² p=3 | note |
|---|---|---|
| plain CG (pressure) | ~470 | grows ∝ 1/h |
| p-MG + damped-Jacobi (legacy default) | **29** | mesh-independent |
| p-MG + Chebyshev | 27 | the structural *ceiling* on p-MG (~6% step) |
| **hp-MG + Chebyshev (new default)** | **22** | the win; the smoother *kind* (1st vs 4th-kind) doesn't matter — structure does |
| C++ pure-h-MG + Chebyshev (prototype) | 16 | the regime that motivated it; **NOT comparable to the Rust p-MG counts** (different algorithm — see skill) |

Mesh-independence of the new default (hp+Cheby): 22 iters at 24²(non-pow2)/32²/48²/64² + p=4. Deflated
singular-pressure p-MG-PCG was 136× fewer iters than CG @64², mesh-independent.

### 4.3 Step-level (the headline — resident-ve-perf, while-graph, identical max tr C)
| grid | legacy p-MG+Jacobi | **default hp+Cheby** | speedup |
|---|---|---|---|
| 32² | 45.6 s / 800 steps | **29.85 s** | **1.53×** |
| 64² | 28.5 s / 300 steps | **19.62 s** | **1.45×** |

Earlier device-residency milestones (defend these too): the while-graph flipped the resident SBM
cylinder from **readback-bound (44.9% util) to GPU-compute-bound (~95% util)**; persistent-handle
fixes took conformation from **0→82% sustained util** (the one-shot `CudaContext::new`/`kernels::load`
anti-pattern). The dual-splitting flow went 38→27 ms/step keeping fields device-resident.

### 4.4 Commits of record (branch `dg-viscoelastic-ibm-gpu-amr`)
- `bb5ef26` — pure-h-at-p3 multigrid + Chebyshev smoother + the 2.2× transfer opt (opt-in).
- `227eaf2` — promote to conforming default + conditional hp-tail + escape hatches (`PMG_LEGACY=1`,
  `JACOBI=1`).

---

## 5. The phased plan

1. **Freeze the baselines.** Land §4 as `benches/` regression tests with the triplet harness (§3) so
   every later step is measured. *Do this first* — refactors without a baseline are how regressions hide.
2. **Feature-flag the GPU.** `gpu` feature, optional `gale-gpu` dep, `Device::Cuda` gated. CPU-only build
   must work with plain `cargo build`.
3. **`Device::plan` + the integrator factory.** Make `sim.device(Cuda{0}).run()` resolve the GPU
   integrator. Level 1 first (it already exists), correctness via the `*-check` tests.
4. **The `ResidentIntegrator` seam.** Promote `GpuResidentVe/Ns` to Level 2 through the Sim; this is where
   the production residency + the ~1.5× live. Gate on the RULE-ZERO nsys check.
5. **Dual-impl the hooks/updaters.** Wire the GPU penalization/IBM/AMR twins behind the
   `StateStageHook`/`Updater` seams so device-complete sims reach Level 2.
6. **Examples/benches/tests reorg** (§3.3). Move the bins; one `Device` line per example.
7. **Autotuning hooks** (§1.3). Make scalar type + launch configs swappable; a per-device autotune pass.

Order matters: **1 → 2 → 3/4 → 5 → 6 → 7.** Each step ends green on the §4 regression triplet.

---

## Appendix — pointers
- Working discipline + measurement gotchas: `.claude/skills/gale-v2-refactor` and `.claude/skills/gale-gpu-perf`.
- Memories: `hp-multigrid-chebyshev`, `phase-e-chebyshev-gpu`, `rust-matvec-matches-cpp`,
  `device-resident-production-mandate`, `gpu-sync-host-device-copies`, `gale-consumer-gpu-precision`.
- Prior plans/research: `docs/plan-device-resident-production.md`, `docs/research-multi-gpu-elliptic.md`,
  `docs/research-mixed-precision.md`, `docs/research-pressure-multigrid.md`, `docs/api-design.md`.
