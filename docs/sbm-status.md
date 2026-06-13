# Sharp-interface (SBM) work stream — status & pin (2026-06-13)

**Pinned / paused here.** Design reference: `docs/research-sharp-interface.md`. This file is
the "where we left off" so the stream can resume cleanly.

## Done & validated
- **Step 1 — geometry** (`gale::dg::shifted`, commit `cc38e47`): `LevelSet`/`CircleLevelSet`,
  `ShiftedBoundary` (active-element classification, surrogate boundary, per-node shift vector
  `d`, `faces_by_elem`). 4 analytic tests (shifts land on the true boundary, `max|d|→0` as O(h)).
- **Step 2a — first-order embedded BC** (`ShiftedPoisson`, commit `5ebbc18`): SIPG restricted
  to active elements + surrogate-Nitsche Dirichlet + identity on inactive dofs. MMS: rel L2
  7.1%→2.7% (plateaus ~2.7% — first-order, ≈ penalization).
- **Step 2b — Taylor correction, HIGH-ORDER** (commit `f0ecef2`): exact symmetric SBM-Nitsche
  (arXiv:2006.00872 eq. 2.14, `S_h u = u+∇u·d`). MMS: 3.6e-3→2.36e-3 (order ≈1.47), **~8×
  better than penalization's ~2% floor**. *This is the validated headline: SBM is sharper than
  penalization.*
- **Parallel apply** (commit `c3fc92c`): record-replay over elements (rayon).

## Step 2c — UNBLOCKED (2026-06-13)
The "stall" was diagnosed (bin `sbm-pressure-diag`) as **pure ill-conditioning**, NOT an operator
bug: the SBM pressure operator is symmetric (asym ~1e-15), non-singular (outflow Dirichlet pins
it, ‖A·1‖≫0), and well-posed — unpreconditioned CG just needed O(10³) iters (≈861 at ny=8, 1800
for the unit-box MMS). Two solver pieces fixed it:

1. **`ShiftedMultigrid`** (`src/dg/shifted_multigrid.rs`) — a p-multigrid using the **SBM operator
   itself** at every level (p-only coarsening; per-level `ShiftedBoundary` rebuilt from the level
   set; checkerboard-colored diagonal + damped-Jacobi smoother + Lagrange p-transfers + V-cycle
   with deflation/coarse band-aid). The *standard* full-mesh `PMultigrid` was tried first and
   FAILED — it ignores the active mask/surrogate, so it stagnates on the cylinder-local modes and
   hits maxit on the physical RHS. The SBM-aware MG converges: MMS validated (Helmholtz 783→96
   iters, Poisson 1800→108, matches direct to ~1e-8).
2. **`ShiftedPoisson::solve_pcg_from`** — preconditioned CG taking an external preconditioner
   closure; the cylinder drives both pressure and velocity with their matching `ShiftedMultigrid`.

**Result (`sbm-cylinder-check`, ny=16, D/h≈3.9):** the cylinder now runs — warm-started pressure
**5–15 iters/step**, velocity **~18** (was a stall at maxit=5000) — and C_D **settles to ≈5.221**
(transient 4.2→7.5→5.8→5.5→5.22, flat in the 4th digit by step ~175). That's **6.4% below** the
DFG reference 5.5795 — STABLE and physically correct, but **not yet beating penalization's ~2.1%**.

The residual error was localized (NOT the solver, which is converged): the **viscous drag recovery
was only first-order** — `drag_x` Taylor-extrapolated the *pressure* to the true circle
(`p_t = p + ∇p·d`) but used **∇u at the surrogate point**, so the viscous traction carried an O(h)
error the high-order SBM BC doesn't.

## High-order drag recovery — DONE (SBM now BEATS penalization)
`drag_x` now also Taylor-extrapolates the velocity GRADIENT to the true circle via the velocity
Hessian: `∂u/∂x(x̃+d) ≈ ∂u/∂x(x̃) + ∇(∂u/∂x)·d` (second derivatives by element-local nodal
differentiation — clean, not noisy in practice). Same flow field, both recoveries printed
(`C_D` = high-order, `(lo …)` = first-order):

| recovery | ny=16 settled C_D | err vs DFG 5.5795 |
|----------|-------------------|-------------------|
| first-order (∇u at surrogate) | 5.219 | −6.5% |
| **high-order (Hessian-extrapolated ∇u)** | **5.637** | **+1.0%** |

**+1.0% beats penalization's ~2.1% floor** (`cylinder-drag-check`) — the headline step-2c result:
**SBM is sharper than diffuse volume penalization for the cylinder drag.**

Also fixed a general CPU-MG bug found en route: `PMultigrid::coarse_solve` ran 500 tight CG
iters/V-cycle on non-h-coarsenable (odd-dim, e.g. 43×8) grids; now uses a loose-tol/cap band-aid
like the GPU.

## Remaining (later)
- Resolution sweep (ny) to confirm the high-order drag converges (order estimate vs penalization).
- Tune the SBM-MG smoother (96–108 iters is ~5× a clean p-MG's ~20 — functional but not optimal).
- The first-order *pressure* surrogate (natural-Neumann at the surrogate location) — a higher-order
  Neumann surrogate could shave the residual further.
- Moving boundary, viscoelastic surface BC, AMR surrogate faces, and the cut-cell DG backend.
