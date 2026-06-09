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

2. **Per-iteration host syncs (P2).** Each dot does a blocking `partial.to_host_vec()`;
   `alpha`/`beta` are then computed on the host and pushed back as kernel scalar args.
   That serializes the CPU and GPU every iteration (up to 81% of iteration time at low
   order). Fix: keep the CG scalars **on-device** — compute `alpha = rs/pAp`,
   `beta = rs_new/rs`, and the convergence test in tiny device kernels, so an entire CG
   iteration is a dependency chain of launches on one stream with **zero host syncs**
   until the final residual check (poll every k iters, not every iter).

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

## Plan

- **Phase 1 — measurement (done).** `bench_poisson_kernels` + `roofline-poisson` bin +
  this report.
- **Phase 2 — kernel wins.** P1 (multi-block reduction) then P3 (parallel face loop).
  Pure kernel changes, validated bit-for-bit against the CPU oracle and re-measured.
- **Phase 3 — solver restructure.** P2 (on-device scalars / sync-free iteration) and P4
  (persistent solver handle). These change the CG/solver API shape, so per the project's
  "research before architecture" rule, run a verified research pass on device-resident CG
  / communication-avoiding patterns first.
- **Phase 4 — revisit P5** only if the re-measured roofline shows the matvec is still the
  ceiling at the orders we actually run.

3D (`poisson3d`) mirrors every kernel and inherits the same fixes; it is benchmarked and
optimized after the 2D pattern is proven.
