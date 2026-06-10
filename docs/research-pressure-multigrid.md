# Multigrid for the singular pressure-Poisson & per-region-BC DG elliptic solves

Verified deep-research reference (2026-06) for extending gale's Dirichlet p-multigrid-PCG
(`PMultigrid` + `poisson_pcg_solve`) to (a) the **singular pure-Neumann pressure-Poisson**
of the dual-splitting scheme and (b) **per-region mixed-BC** elliptic solves. Run via the
deep-research harness (22 primary sources fetched, 25 claims adversarially 3-vote verified,
20 confirmed / 5 killed). Confidence tags and citations are per-finding.

## Bottom line for gale

Our existing design is **the right foundation** — the research confirms each of: MG used as
a *preconditioner* for CG (not a standalone solver), **rediscretization** of the operator
per level (we already do `Poisson::with_*` per level — *not* Galerkin RAP), damped-Jacobi
smoothing, and **deflation/projection (subtract the mean), not pinning a DOF**. To handle
the singular pressure solve we need three focused changes, in priority order:

1. **BC-aware coarse operators (required).** `PMultigrid` must build each level with the
   *same* boundary conditions as the fine problem — for pressure that's pure-Neumann
   (`Poisson::with_bc(mesh, alpha, 0, all_tags)`); for per-region it's the same neumann-tag
   set at every p-level. We **rediscretize** per level (already our approach), which sidesteps
   the refuted "Galerkin RAP automatically inherits BCs/penalty" claims — three RAP/operator-
   coarsening claims were killed 0-3, so do NOT switch to RAP. The SIPG interior penalty must
   stay coercive (~`α p²/h`) at every coarsened order — our `τ = α(p+1)²/h` already recomputes
   per level. [Stiller JCP 2016 arXiv:1603.02524; Fortunato-Rycroft-Saye SISC 2019
   arXiv:1808.05320]

2. **Nullspace handling for the singular case (required).** Confirmed strategy (high
   confidence): keep the SPD V-cycle preconditioner **unmodified**, and handle the constant
   nullspace in the **outer PCG** by projecting the residual onto the range = subtracting its
   mean (`r ← r − mean(r)`) at init and after each residual update. We already deflate this
   way in `pressure_cg_solve`/`cg_deflated`. **The one addition:** the **coarsest-level solve
   must itself be made nullspace-consistent** — project the coarse RHS onto the range (subtract
   mean) each coarse-CG iteration (Kaasschieter). Our GPU `coarse_cg!` currently runs *plain*
   CG; on the singular Neumann coarse operator that must gain the same mean-projection or it
   won't converge. [McAdams-Sifakis-Teran SCA 2010 (MGPCG); Stiller arXiv:1603.02524 (coarse
   pseudoinverse via Kaasschieter); Min et al. JSC 2016 (pinning has a Laplace pole ⇒ less
   accurate)]

   - SPD-preconditioner conditions to preserve: `R = Pᵀ` up to scaling (our prolong/restrict
     are transpose pairs), symmetric pre/post smoothing (automatic for Jacobi — no reversal
     needed), and an SPD/consistent coarse solve. [MST10; Tatebe 1993; Bramble-Pasciak]
   - Mixed-BC pressure (an outflow pins `p=0`) is **non-singular** ⇒ no deflation needed; the
     code already branches `has_outflow` (deflate ⇔ no outflow). The MG path must inherit that
     branch: deflate the outer PCG *and* the coarse solve only when the configuration is truly
     pure-Neumann.

3. **Smoother (optional upgrade).** Damped Jacobi (`ω ≈ 0.8`; undamped `ω=1` *diverges* for
   DG — we use `ω = 4/(3 λ_max)`) is the cheap GPU-friendly baseline and is what we have. The
   literature gets true **p-robustness up to P=32** only with **overlapping-Schwarz** (element/
   face block) or a **fourth-kind Chebyshev (Lottes) polynomial** smoother — the latter is the
   lowest-friction matrix-free upgrade (diagonal + matvecs only, no damping tuning, no block
   solves) if Jacobi proves p-sensitive on the pressure solve. Not needed to start. [Stiller
   arXiv:1603.02524 (Schwarz, P=32); Lei-Zhang-Zheng JSC 2025 arXiv:2509.13669 (4th-kind
   Chebyshev, O(p) vs O(p²) sweeps, W-cycle proof); LDG MG arXiv:2412.12506 (`ω=0.8`)]

## Confirmed findings (3-vote verified unless noted)

- **Projection/deflation, not pinning.** Pinning a DOF introduces a Laplace-fundamental-
  solution pole at the pinned point and is provably less accurate; projection out of the
  nullspace costs only a dot+AXPY per iter and preserves accuracy. *(high)* [Min et al. JSC
  2016; arXiv:1512.01756; arXiv:1607.01323]
- **Outer-CG mean-projection suffices for the global nullspace; the V-cycle itself needs no
  nullspace modification** — but this is *V-cycle-as-preconditioner only* (zero inner initial
  guess); a singular V-cycle as a standalone solver can diverge. *(high; the over-strong
  "nullspace handled entirely on the CG side, coarse solve untouched" variant was refuted
  0-3.)* [MST10]
- **Coarsest solve must be nullspace-consistent** (RHS projected to range / pseudoinverse).
  *(high)* [Stiller arXiv:1603.02524; LDG MG arXiv:2412.12506]
- **MG-as-preconditioner tolerates a cheap/inexact V-cycle** and removes the boundary-
  sensitivity that singular standalone V-cycles suffer. *(high)* [MST10; Tatebe 1993]
- **p-MG for nodal IPDG/LDG Poisson is robust in h and p up to P=32** — but that robustness
  is a property of the *smoother/cycle* (Schwarz / Chebyshev / W-cycle), **not** guaranteed
  for a damped-Jacobi V-cycle. *(high)* [Stiller; Lei et al. 2025]
- **Precedent for the exact target:** block-Jacobi + deflation gives grid-size-independent
  Krylov convergence for the discontinuous high-order Poisson-Neumann Schur complement in an
  incompressible-NS operator-splitting scheme. *(medium; Schur-interface deflation, not a
  p-MG V-cycle — transfer is partial.)* [Joshi-Thomsen-Diamessis JCP 2016 arXiv:1512.01756]

## Killed claims (do not rely on)

- "Galerkin/RAP operator-coarsening automatically inherits SIPG fluxes/penalty/BCs at coarse
  levels" — refuted 0-3. **Rediscretize per level instead** (gale already does).
- "SIPG penalty is inherited unchanged (same value) at every coarse level" — refuted 1-2;
  recompute `τ ~ α p²/h` per level (gale already does).
- "Deflation+block-Jacobi gives ~30 iters / half the work vs additive Schwarz" (specific
  quantitative claim) — refuted 1-2.

## Open questions (resolve empirically during implementation)

1. Is damped-Jacobi V-cycle p-robust enough for the pressure-Poisson, or is the 4th-kind
   Chebyshev smoother needed — and at what order does Jacobi break down? (Measure iters vs p.)
2. Per-region BC tag inheritance across p-levels: verify rediscretization-per-level gives a
   consistent SPD coarse operator for the mixed-BC case (it should, since we rebuild
   `Poisson::with_bc` per level).
3. Cheapest matrix-free GPU range-projection on Titan V: global mean-subtraction reduction per
   outer CG iter (we already have the multi-block dot for this) vs a deflation-space projection.
4. Singular vs non-singular detection: the pressure solve is singular (pure-Neumann) or
   non-singular (has an outflow) per configuration — the MG path must switch deflation on/off
   accordingly (mirror the existing `has_outflow` branch).

## Implementation plan (grounded by the above)

1. `PMultigrid`: add a boundary spec (neumann-tag set) so each level is
   `Poisson::with_bc(mesh, alpha, reaction, neumann_tags)`; pressure = all-tags Neumann,
   reaction 0. (The colored-probing diagonal already handles the Neumann operator — boundary
   faces just carry the `NEU` sentinel.)
2. GPU `poisson_pcg_solve`: add a `deflate` flag — when set, subtract the mean of the residual
   in the outer PCG (init + each iter) **and** project the coarse-CG RHS to the range each
   coarse iter.
3. Validate: a `pcg-pressure-check` mirroring `pcg-helmholtz-check` — MG-PCG vs deflated CG on
   the pure-Neumann pressure operator, expecting mesh-independent iterations and a matching
   (mean-zero) solution.
4. Then wire into the flow pressure solve (persistent handle + `from_mesh`), deflation gated on
   `!has_outflow`.

Sources (primary): arXiv 1603.02524, 1808.05320, 2412.12506, 1512.01756, 1607.01323, MST10
(SCA 2010), Min et al. JSC 2016, Lei-Zhang-Zheng JSC 2025 (2509.13669), plus the GPU
matrix-free MG set (1910.03032, 2405.18982, 1805.11930, 2410.09497).
