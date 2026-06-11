# Mixed precision for gale's GPU elliptic solvers — research reference

Verified deep-research pass (5 search angles, 23 sources fetched, 102 claims extracted, 25
adversarially verified by 3-vote, 21 confirmed / 4 refuted). Authorities cited: Carson, Higham,
Anzt, de Sturler, Warburton, Notay, Tsai. Several key results are **2025 preprints** (cite as such).
This is a *method-decision reference* per [[research-before-architecture]]; it complements the
consumer-GPU precision roadmap (`gale-consumer-gpu-precision`).

## EMPIRICAL ADDENDUM (measured on gale, Titan V) — it does NOT pay off here

Implemented the FP32-gradient-intermediate V-cycle (opt-in `GpuPoissonMg::with_mixed_precision`,
FP64 default bit-exact) and measured it (`mixed-precision-check`). **Correct but no speedup: 0.99×
at 16²–64²** (flat), with the mixed solution matching FP64 to ~the solve tolerance at the same
iteration count. Root cause confirms the report's central caveat: **gale's h-coarsened V-cycle is a
sequence of SMALL matvecs that are latency/occupancy-bound, not bandwidth-bound** (per `ncu`) — so
halving `gx/gy` bytes saves nothing. The cited ~2× speedups are all large stored-CSR/AMG/FD-Poisson
SpMV, a regime gale isn't in *after* the h-coarsening that gave the big algorithmic win. So on the
Titan V (native FP64), **don't pursue mixed precision further** — the headroom isn't there. The
infrastructure (generic two-precision matvec, the precision-boundary rule below, the opt-out knob)
is kept for the **RTX 5090**, where the calculus flips: FP64 is ~1:64, so mixed precision becomes
about *avoiding crippled FP64 throughput*, not bandwidth.

**Critical precision-boundary lesson (caught by the convergence guard via a NaN):** FP32 is safe
only in the V-cycle *smoother* (a robust stationary Jacobi). The **outer PCG `A·p`** and the
**deflated singular coarse-grid CG** must stay FP64 — both are CG, which diverges (→NaN) on an
FP32-perturbed, non-symmetric operator. The coarse grid is tiny (h-coarsened) so FP64 there is free.

## TL;DR verdict — do it, the standard way

Run the **whole V-cycle preconditioner in FP32** (damped-Jacobi smoother, restriction/prolongation,
matvec/operator, coarse solve, and the field + geometric-factor *storage*) inside an **FP64 outer
CG** (residual, dot-products/reductions, solution update, deflation). This is the canonical,
theory-backed pattern, validated on V100/sm_70 (the Titan V's family), on high-order FEM/SEM
patch-smoother multigrid, and on CFD Poisson. It is **memory-traffic-only** speedup — the iteration
count is *essentially* (not exactly) preserved, which is precisely what gale's memory-bound roofline
wants.

**Expected payoff: modest, must be benchmarked, not copied.** Published numbers (1.4–2.5×) all come
from *stored-CSR SpMV / AMG / finite-difference Poisson* — none from a matrix-free high-order SIPG
operator. gale has **no stored index array**, so the FP32 bandwidth ceiling is ~**2×** (8→4 bytes on
the field + geometric-factor traffic) rather than the cited 1.5×, but the actual mix (field vectors +
geometric factors + flux terms) was measured by *no* source here — **re-derive it from the roofline
and benchmark.** Realistically expect a fraction of 2× on the solve once non-matvec work is included.

**Standard PCG first; flexible CG only if it stagnates.** An FP32 V-cycle perturbs the preconditioner
by ~6e-8 (FP32 unit roundoff), which qualifies as "sufficiently small" for ordinary PCG. *Do not
assume the iteration count is identical to FP64* (that specific claim was refuted), and *do not
over-engineer FCG* (the claim that a Polak-Ribière/Axelsson orthogonalization is the required
mechanism was refuted) — FCG is documented insurance, not a prerequisite.

## The precision split (the design)

| Component | Precision | Why |
|---|---|---|
| Outer CG residual `r = b − Ax` | **FP64** | sets the final attainable accuracy |
| Dot products / reductions (`r·z`, `p·Ap`, residual norm) | **FP64 accumulate** (or Kahan/pairwise) | FP32 reductions lose accuracy; this is the residual-gap lever |
| Solution update `x += αp` | **FP64** | accumulates the answer |
| Deflation / constant-nullspace mean removal | **FP64** | *inference* (see §6) — cheap, keep tight |
| V-cycle smoother (damped Jacobi) | **FP32** | bulk of the work; perturbation tolerable |
| Restriction / prolongation | **FP32** | preconditioner-internal |
| Matvec / operator inside the V-cycle | **FP32** | the memory-bound long pole |
| Coarse-grid solve | **FP32** | preconditioner-internal (tiny grid since h-coarsening) |
| Field + geometric-factor *storage* | **FP32** | the actual bandwidth lever (matvec is FP32-storage / FP64-accumulate) |

## §1 — Mixed-precision Krylov: FP32 preconditioner in FP64 outer CG

- **It doesn't compromise final accuracy** for not-too-ill-conditioned systems. Bake, Carson & Ma
  (arXiv:2510.11379, 2025): "applying preconditioners in low precision does not compromise the
  accuracy of the final results, provided that reasonable conditions are satisfied." Mixed-precision
  PCG still reaches `O(u)` backward error and `O(u)·κ(A)^½` forward error; the *attainable* residual
  nears unit roundoff only when `κ(M)^½` and the iteration count `k` are small enough (an extra
  `κ(M)^½` factor appears in the preconditioned backward-error bound). The guarantee imposes
  `O(ν)·κ(A) ≤ ½`, so it **degrades for highly ill-conditioned A** — relevant as the SIPG operator's
  conditioning grows with `p` and `1/h` (§3).
- **Standard PCG generally suffices.** Notay, *Flexible Conjugate Gradients* (SIAM J. Sci. Comput.
  2000, DOI 10.1137/S1064827599362314): FCG handles "preconditioning slightly variable from one
  iteration to the next" and "the convergence rate is essentially independent of the variations …
  as long as the latter are kept sufficiently small." An FP32 V-cycle's ~6e-8 variation qualifies.
  **Refuted sub-claims (do not rely on):** that the FP32-in-FP64 AMG config keeps an *identical*
  iteration count (refuted 1-2), and that FCG's robustness *requires* an Axelsson/Polak-Ribière
  orthogonalization (refuted 0-3). ⇒ treat FCG as a fallback if stagnation shows up.
- **AMP-PCG (Guo, de Sturler, Warburton, arXiv:2505.04155, 2025)** — Warburton is a DG/SEM authority,
  directly relevant. Two precision knobs: one for "matvec and residual vector, which mainly controls
  the final attainable residual accuracy" (start FP64, drop to FP32 by tolerance + a residual-gap
  estimate), a second for "preconditioned residual and search direction vectors, which primarily
  affects the convergence rate" (can go to FP16). Reported **1.63× GPU** speedup at FP64-comparable
  accuracy, savings from reduced memory.
- **Three-precision IR framing** (Carson & Higham, SIAM SISC 2018, DOI 10.1137/17M1140819): condition
  number sets the precision budget — low-precision-LU IR needs `κ(A) ≲ 1e4`, **GMRES-IR relaxes it to
  `≲ 1e8`**. The takeaway for gale isn't LU-IR itself but the principle: *monitor* the FP32
  preconditioner's effectiveness as `p`/resolution rise.

## §2 — Mixed-precision multigrid V-cycle

- The **V-cycle goes in low precision; the final solution stays FP64** via the outer
  iterative-refinement / residual scaling. FP16 geometric MG as preconditioner + FP64 outer gives
  **up to 2.5× on V100** with "the iteration count is almost not affected by using lower accuracy"
  (arXiv:2007.07539). High-order FEM patch-smoother MG: a dedicated "exploiting mixed precision"
  scheme runs the MG preconditioner in single while the outer iteration stays double, **+~70%
  throughput** (arXiv:2405.19004). GPU Stokes MG has a dedicated mixed-precision section
  (arXiv:2410.09497). CFD Poisson: outer CG/CA-Lanczos kept FP64, only the Chebyshev smoother +
  transfers FP32/FP16 (ORNL ScalA 2021).
- **The smoother can be the lowest precision in the cycle** — lower than residual/restriction/
  prolongation/correction (Vacek, Anzt, Carson, Kohl, Rüde, Tsai, arXiv:2511.04566, 2025; measured
  **up to 1.43×**, energy to 71%). *Caveat:* scoped to **incomplete-Cholesky** smoothing; gale uses
  **damped Jacobi**, so this ordering is suggestive, not automatic — validate gale's smoother
  precision empirically.
- **Decouple working-vector precision from storage precision** (Tsai, Beams, Anzt, OSTI 2581015,
  2023): "we employ the common mixed precision technique of decoupling the working vector precision
  from the matrix storage precision … higher precision for the vectors helps avoid zero residuals and
  Jacobi smoother overflow, but we retain most of the benefit … as the matrix accounts for the bulk
  of the memory movement."

## §3 — High-order DG-SEM / SIPG specifics

- No source measured FP32 preconditioning *for a matrix-free SIPG operator specifically* — this is the
  **biggest transferability gap**. What the literature gives is the governing principle: attainable
  accuracy and FP32-preconditioner viability are bounded by `κ(A)` / `κ(M)` (§1), and the SIPG
  operator's conditioning **grows with `p` and `1/h`** and with the interior-penalty parameter. So:
  FP32 should be fine at the orders we run (validated empirically up to p≈8 for the FP64 MG already),
  but **the FP32 preconditioner's effectiveness must be monitored as `p` and resolution climb** — at
  very high `p` the `κ(M)^½` factor can erode the attainable accuracy and/or add outer iterations.
  Mitigation if it bites: keep the *coarse solve* or the *penalty-heavy face terms* in FP64, or fall
  back to FCG.

## §4 — Consumer GPUs (RTX 5090, FP64 ~1:64) and FP64 emulation

- **Plain FP32-preconditioner + FP64-outer already captures most of the benefit.** The question on
  consumer cards is the *outer* FP64 parts (residual, dots, update) where hardware FP64 is crippled.
- **Double-single / Dekker / two-product double-double** (error-free transforms; Ogita–Rump–Oishi
  accurate summation/dot, TUHH OgRuOi05; the Ozaki scheme) emulate ~FP64 from FP32 ops, but at a
  real cost (several FP32 ops per emulated FP64 op). Community FP64-emulation-on-consumer-GPU efforts
  exist (NVIDIA forums "weekend project").
- **Verdict for gale: double-double is likely NOT worth it.** For a *time-accurate* viscoelastic sim
  the per-step elliptic solve only needs ~time-discretization accuracy (**~1e-4 to 1e-6**, as the
  `solve-tol-sweep` confirmed). FP32-class accuracy with **FP64-emulated (Kahan/compensated)
  dot-products** for the few reduction operations is the pragmatic migration target; reserve
  double-double for the rare case where a true FP64-equivalent residual is genuinely required. *(This
  consumer-path verdict is engineering inference from the attainable-accuracy theory + the stated
  tolerance, not a directly-cited result — validate when the 5090 work starts.)*

## §5 — Matrix-free GPU implementation

- **FP32-storage / FP64-accumulate matvec** is the recommended strategy: store fields + geometric
  factors in FP32 (halves their traffic on the memory-bound kernel), accumulate the tensor
  contractions in FP64 registers (avoids zero-residual / smoother-overflow failures, costs nothing
  since we're bandwidth-bound). gale's affine-metric matvec already shrank the geometric traffic to
  a few scalars + the GLL mass; the remaining FP32 lever is the **field vectors** (`u`, `gx`, `gy`)
  and the per-level scratch.
- **Reductions are the accuracy trap.** FP32 dot products lose accuracy and set the residual gap →
  keep the on-device reductions **FP64-accumulate** (or Kahan/pairwise). gale already does the dots
  on-device via `dot_partial`→`reduce_scalar`; the partial/accumulate should stay FP64 even if the
  input vectors are FP32.
- **Scalar-type abstraction:** the matvec/V-cycle kernels should be generic over the storage scalar
  (FP32) with FP64 accumulation, while the outer-CG vectors/scalars stay FP64 — the hook flagged in
  `gale-consumer-gpu-precision`.

## §6 — Singular pure-Neumann pressure (deflation)

No surviving source addresses deflation precision directly. **Inference:** the constant-nullspace
mean removal (`r ← r − mean(r)`, the range projection) is cheap (one reduction + one axpy) and
directly affects whether the residual stays in the operator's range — keep it **FP64** with an FP64
reduction. It's a negligible fraction of the work, so there's no reason to risk it in FP32. Flag as
engineering judgment to validate.

## §7 — Recommendation for gale + first increment

**Component split:** as the table above. **Outer solver:** standard PCG; add flexible CG only if a
stagnation/residual-floor appears. **Reductions:** FP64 accumulate always. **Deflation:** FP64.

**First implementation increment (lowest risk, isolates the win):** an **FP32-storage / FP64-accumulate
matvec** — store the operator's input field + `gx,gy` in FP32, contract in FP64, used *inside the
V-cycle only*, with the outer CG and all reductions unchanged in FP64. Validate bit-approx against the
FP64 oracle (expect ~FP32-level agreement in the *preconditioner*, FP64 in the *solution*), then
benchmark the matvec bandwidth with the existing `ncu` flow to measure the real (matrix-free, ~2×
ceiling) gain before extending FP32 to the smoother/transfers/coarse solve.

**Migration to FP64-scarce GPUs (RTX 5090):** the same split, with the FP64 outer reductions done via
Kahan/compensated FP32 (FP64-emulated) sums; double-double only if a true FP64 residual is required —
likely unnecessary at the ~1e-4..1e-6 per-step tolerance.

## Pitfalls (from the verified findings)

- **Don't assume identical iteration count** vs FP64 (refuted claim) — measure it.
- **Don't over-specify FCG** (refuted) — plain PCG first, FCG as fallback.
- **Don't copy the published speedups** — they're stored-CSR/AMG/FD-Poisson, not matrix-free SIPG;
  the gale ceiling is ~2× (no index array) but the realized number must be benchmarked.
- **Watch attainable accuracy at high `p`/resolution** — the `κ(M)^½` factor erodes it; monitor.
- **Keep matvec/residual precision ≥ the target residual accuracy** — it's the controlling knob; the
  preconditioned-residual/search-direction can go lower.

## Sources (primary unless noted)

- Bake, Carson & Ma, *Error analysis of preconditioned CG in mixed precision*, arXiv:2510.11379 (2025, preprint)
- Guo, de Sturler & Warburton, *AMP-PCG* (adaptive mixed-precision PCG), arXiv:2505.04155 (2025, preprint)
- Carson & Higham, *Iterative refinement in three precisions*, SIAM SISC 40(2), DOI 10.1137/17M1140819 (2018)
- Notay, *Flexible Conjugate Gradients*, SIAM J. Sci. Comput., DOI 10.1137/S1064827599362314 (2000)
- *FP16 geometric multigrid preconditioner on V100*, arXiv:2007.07539
- Vacek, Anzt, Carson, Kohl, Rüde, Tsai, *Mixed-precision IC-smoother multigrid*, arXiv:2511.04566 (2025, preprint)
- Tsai, Beams & Anzt, *Three-precision algebraic multigrid on GPUs*, OSTI 2581015 (Elsevier 2023)
- *Mixed-precision CFD Poisson (Chebyshev smoother)*, ORNL ScalA 2021
- *High-order FEM patch-smoother multigrid, mixed precision*, arXiv:2405.19004; GPU Stokes MG, arXiv:2410.09497
- Ogita, Rump & Oishi, *Accurate sum and dot product*, TUHH (EFT / double-double summation)

## Caveats

Strong source quality (mostly primary peer-reviewed or arXiv from recognized authorities), but: (1)
several core results are 2025 preprints; (2) **all quantitative speedups are from non-matrix-free,
non-DG-SEM operators** — gale's realized speedup must be re-derived/benchmarked; (3) the
lowest-precision-smoother ordering is IC-smoother-specific, not proven for damped Jacobi; (4) the
deflation-precision and consumer-GPU double-double verdicts are engineering inference, not directly
cited; (5) four related claims were refuted — see the pitfalls. Re-validate the gale-specific numbers
empirically; the *architecture* (FP32 preconditioner / FP64 outer + reductions) is well-grounded, the
*magnitudes* are not yet.
