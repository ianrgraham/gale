# GPU performance: roofline analysis and optimization plan

Reference for the performance/optimization pass on gale's GPU SIPG-Poisson CG solver —
the bottleneck of the dual-splitting flow path (every timestep runs 1 pressure + 2–3
velocity elliptic solves). Measured on a **Titan V** (Volta GV100): FP64 peak ≈ 6.9
TFLOP/s, HBM2 bandwidth ≈ 652.8 GB/s, so the roofline ridge point is at AI ≈ **10.6
FLOP/byte** — every DG kernel here is far below that (AI 0.08–0.66), i.e. firmly
**memory-bound**. The win is therefore *bandwidth utilization*, not FLOP reduction.

Reproduce with `cargo oxide run --bin roofline-poisson` (2D; `bench_poisson_kernels`
times each kernel in isolation via CUDA events, ideal-traffic byte/FLOP accounting).

## Baseline (commit before the optimization pass)

64×64 elements, conforming, ideal traffic. `%BW` = achieved / 652.8 GB/s.

| order | kernel       | µs/call | GB/s  | %BW   | note |
|-------|--------------|---------|-------|-------|------|
| p=4   | gradient     | 13.2    | 497   | 76%   | sum-factorized, well-utilized |
| p=4   | operator     | 57.4    | 143   | 22%   | **single-threaded face loop** |
| p=4   | axpy / xpby  | 3.3     | 755   | ~peak | streaming, fine |
| p=4   | dot_partial  | 92.5    | 18    | **2.7%** | **single 256-thread block** |
| p=8   | gradient     | 33.8    | 628   | 96%   | |
| p=8   | operator     | 141     | 188   | 29%   | |
| p=8   | dot_partial  | 511     | 10    | **1.6%** | costs more than the operator |

Full `helmholtz_cg_solve` wall-clock, per CG iteration (1 apply + 2 dots + 2 axpy + 1 xpby):

| order | iters | µs/iter | host sync+launch overhead |
|-------|-------|---------|---------------------------|
| p=2   | 585   | 653     | **81%** of the iteration  |
| p=4   | 1309  | 542     | **51%**                   |
| p=6   | 2122  | 750     | 36%                       |

(At p=8 the per-iter device kernels are so large the host overhead is hidden behind them;
the attribution is a crude wall − Σ(isolated-kernel) and goes slightly negative there from
stream overlap. The low-order rows are the clean signal.)

## Bottlenecks, ranked by payoff

1. **`dot_partial` reduction (P1, biggest win).** A single 256-thread block reduces the
   whole vector → one SM, ~10–18 GB/s (1.6–2.8% of peak). Two dots per CG iteration, so
   at p=8 the dots alone are ~1 ms/iter. Fix: a proper **multi-block tree reduction**
   (grid-stride partials per block → second pass / atomic combine). A streaming reduction
   (AI 0.12) should reach ~90% of peak BW — a ~30–50× speedup on this kernel.

2. **Per-iteration host syncs (P2).** Each dot did a blocking `partial.to_host_vec()`;
   `alpha`/`beta` were computed on the host and pushed back as kernel scalar args. Fix:
   keep the CG scalars **on-device** (`reduce_scalar` → `cg_alpha`/`cg_beta` device kernels;
   `axpy_s`/`xpby_s` read the scalar from a device buffer), polling the residual to the host
   only every `CHECK` iterations. **NOTE (corrected after measuring):** this was *not* the
   dominant cost — see the §"Measurement correction" below. It is a real but secondary win;
   it makes the iteration 66–82% device-bound, and it is the right structure for once P4
   removes the setup cost that was masking everything.

3. **`operator` face loop (P3).** The SIPG face contribution is computed by thread 0 only
   (`if m == 0 { … }`), so the operator sustains 22–29% BW vs gradient's 76–96%. Fix:
   **parallelize the face loop across the block** (one thread per face node, accumulate
   the symmetry-lift / penalty into shared memory with the existing `sync_threads`).

4. **Per-solve setup (P4).** `poisson_cg_solve` does `CudaContext::new` + `kernels::load`
   + re-uploads *all constant mesh arrays* on every call — 3×/timestep in the flow loop,
   re-sending identical data. Fix: a **persistent solver handle** that owns the context,
   module, and the uploaded mesh metrics, so a step only uploads the RHS and downloads the
   solution. (Architectural — see the research note below before committing the shape.)

5. **Tiny matvec blocks (P5, order-dependent).** `gradient`/`operator` launch one block
   per element with `nn` threads (25 at p=4, 64 at p=3 in 3D) — below Volta's occupancy
   sweet spot, hence gradient only reaching 76% at p=4. Multi-element-per-block packing
   would help low orders but complicates the shared-memory tiling; lower priority since
   higher orders already utilize well and P1/P2 dominate the iteration cost.

## Phase 2 results — kernel wins (P1 + P3)

Same 64×64 benchmark after the two kernel rewrites. **P1** = multi-block grid-stride
`dot_partial` (one block-partial per SM-worth of blocks, host sums ≤1024 partials).
**P3** = SIPG operator face term rewritten from a single-threaded scatter (`if m==0`) to
a race-free per-node gather (each thread sums the faces its node lies on into registers;
drops the `RF/HX/HY` shared arrays and 2 of 3 `sync_threads`).

| kernel / metric        | p=4 before → after | p=6 before → after | p=8 before → after |
|------------------------|--------------------|--------------------|--------------------|
| `dot_partial` %BW      | 2.7% → **50.5%**   | 2.8% → **84.0%**   | 1.6% → **69.6%**   |
| `dot_partial` µs/call  | 92.5 → **5.0**     | 178 → **5.9**      | 511 → **11.7**     |
| `operator` %BW         | 22% → **27%**      | 28% → **37%**      | 29% → **42%**      |
| `operator` µs/call     | 57.4 → **46.0**    | 86.7 → **66.3**    | 141 → **96.1**     |
| **full CG solve wall** | 710 → **464 ms**   | 1591 → **680 ms**  | 3460 → **1024 ms** |
| **CG speedup**         | **1.53×**          | **2.34×**          | **3.38×**          |

`dot_partial` went from 1.6–2.8% to 50–84% of peak bandwidth (10–44× per call). The
operator gather is a smaller win (still below `gradient`'s BW — the per-thread 4·n1 face
scan adds redundant cached metadata reads and some divergence) but it is no longer
serialized on one thread. Validated bit-for-bit: poisson-operator/cg-check, stokes-check,
ns-check, flow-bc-check, flow-nc-bc-check all pass.

**Now dominant: per-iteration host-sync overhead (P2)** — 41–91% of the iteration once the
kernels are fast. That is the Phase 3 target.

## Measurement correction — the real overhead is per-solve *setup*, not per-iter sync

Phase 1 attributed the `wall − Σ(device kernels)` gap to per-iteration host-sync/launch
overhead. Direct measurement (the `roofline-poisson` `axpy` launch probe + a 1-iteration
solve) shows that was **wrong**:

- Kernel launches are async and ~free: 200 back-to-back `axpy` launches with a single sync
  cost ≈ the device time (3.1 µs wall vs 3.0 µs device) → ~0.1 µs host per launch.
- A **1-iteration** `helmholtz_cg_solve` takes **~285–310 ms** — that is the fixed
  per-solve cost: `CudaContext::new` + `kernels::load` (cubin/JIT) + uploading all the
  constant mesh arrays. It is the entire "overhead", and dividing it by the iteration count
  is exactly why the earlier metric *looked* like a per-iteration cost that shrank with
  order (more iters ⇒ more amortization).

Corrected per-solve breakdown (post P1+P2+P3):

| order | iters | setup (ms) | setup % of solve | iteration (µs) | iteration % device |
|-------|-------|------------|------------------|----------------|--------------------|
| p=2   | 600   | 286        | **88%**          | 63             | 77% |
| p=4   | 1325  | 284        | **64%**          | 121            | 66% |
| p=6   | 2125  | 310        | **49%**          | 152            | 74% |
| p=8   | 3075  | 304        | **31%**          | 221            | 82% |

**P4 (persistent solver handle) is therefore the dominant remaining win**, especially at
the p=4–6 we actually run, and *most* of all in the time-stepping loop: every step does 3
solves, each re-creating the context, re-loading the identical module, and re-uploading the
identical mesh — ~0.9 s/step of pure setup. A handle owning the context/module/mesh buffers
makes a solve just RHS-upload + solution-download.

## Phase 3 results — on-device CG scalars (P2)

Standard CG, scalars kept device-resident (`reduce_scalar`/`cg_alpha`/`cg_beta`/`axpy_s`/
`xpby_s`), residual polled every `CHECK = 25` iters. Mathematically identical to the host CG
(≤ `CHECK`−1 extra iters past convergence). Validated bit-for-bit (poisson-cg-check matches
CPU to 4.6e-14, stokes 4.1e-10). The iteration is now 66–82% device-bound; the absolute
solve time barely moves *because per-solve setup still dominates* — the payoff is realized
once P4 lands. (2D `cg_solve_impl` only so far; the deflated/3D/NC CGs still readback per
dot and are updated alongside P4.)

## Plan

- **Phase 1 — measurement (done).** `bench_poisson_kernels` + `roofline-poisson` bin +
  this report.
- **Phase 2 — kernel wins (done, 2D+3D+NC).** P1 (multi-block reduction) + P3 (parallel
  face loop). Validated bit-for-bit; `dot` 10–44×, CG 1.5–3.4×.
- **Phase 3 — on-device CG scalars (done, 2D).** P2; secondary win (setup masks it).
- **Phase 4 — persistent solver handle (P4) — NEXT, the dominant win.** A handle owning the
  CUDA context, loaded module, and uploaded constant mesh arrays, so a solve (and each of
  the 3 solves/timestep) skips the ~0.3 s setup. API change across the GPU flow solvers.
  Removes ~0.9 s/step in the sim loop. Also fold the deflated/3D/NC CGs onto the on-device
  scalar path here.
- **Phase 5 — revisit P5 (multi-element-per-block matvec)** only if the re-measured roofline
  shows the matvec is still the ceiling at the orders we actually run.

3D (`poisson3d`) mirrors every kernel and inherits the same fixes; it is benchmarked and
optimized after the 2D pattern is proven.
