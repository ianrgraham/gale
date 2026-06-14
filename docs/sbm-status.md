# Sharp-interface (SBM) work stream — status & pin (2026-06-13)

**Pinned / paused here.** Design reference: `docs/research-sharp-interface.md`. Method landscape +
**fallback plan** for the uncharted work (moving / many-body / viscoelastic):
`docs/embedded-boundary-methods.md`. This file is the "where we left off" so the stream resumes cleanly.

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

## GPU port — DONE (full cylinder on device, validated)
All four increments complete; the SBM cylinder runs entirely on the GPU and reproduces the CPU
result. **ny=16: GPU C_D = 5.6375 (+1.0%), identical to CPU to 4 digits**; full 260-step settled
run in **~23 s wall-clock** (incl. build/setup, ~3 CPU cores) vs the CPU's ~15–18 min — a large
per-step speedup. The GPU solve is **warm-started** (`solve_from`, x0 = previous step), so it
runs ~4–5 iters/step near steady, matching the CPU's warm-started counts (was ~30/step cold —
warm-start cut the settled run 35 s → 23 s). Validation bins all pass: sbm-poisson-check (apply 7.6e-15),
sbm-pcg-check (pressure MG 4.2e-8), sbm-velocity-check (Dirichlet+Taylor 9.2e-14 / 3.8e-12),
and the standard Poisson path is unaffected (pcg-pressure-check 4.3e-10). Run the GPU cylinder
with `SBM_GPU=1 cargo oxide run --bin sbm-cylinder-check`.

### Increment log
Extending the existing on-device p-MG-PCG (`gale-gpu` `poisson.rs`) to SBM. The affine-collapse +
`fnbr` sentinel design makes the natural-Neumann (pressure) case nearly free: an active→inactive
face is just `NEU`, an inactive element is an identity block.
- **Increment 1 DONE** (commit `00fbf09`): `sbm_operator` kernel (= `operator` + `active_elem`
  mask → inactive=identity) + `sbm_flatten_mesh` (active→inactive faces `NEU`) + `sbm_poisson_apply`.
  Bin `sbm-poisson-check`: GPU matches CPU `ShiftedPoisson::apply` to rel 7.6e-15 (bit-for-bit).
- **Increment 2 DONE** (commit `9584ec9`): `sbm_operator_jacobi` + an optional `active` mask on
  `MgConst` + `MgConst::build_sbm` (p-only, from a CPU `ShiftedMultigrid`) + the 4 matvec/smoother
  macros branch on `active` (Poisson path byte-identical, re-validated). `sbm_pcg_solve` + bin
  `sbm-pcg-check`: GPU SBM-MG-PCG matches CPU `ShiftedMultigrid::pcg` to rel 4.2e-8, 19=19 iters.
- **Increment 3 DONE** (commit `9fad3e9`): velocity operator (Dirichlet + Taylor surrogate). New
  `SURR` sentinel + per-face-node shift vectors `sdx/sdy`; in the kernel `su = u + gx·sdx + gy·sdy`,
  consistency+penalty + symmetry lift + the Taylor penalty lift folded as `pr = rx·(wx − hx + px)`.
  `sbm_flatten_mesh` takes the `ShiftedBoundary` + (surrogate_dirichlet, taylor); `MgConst` bundles
  per-level `SbmData{act,sdx,sdy}`. Bin `sbm-velocity-check`: apply 9.2e-14, MG-PCG 303=303 / 3.8e-12.
- **Increment 4 DONE** (commit `cf44f77`): `GpuPoissonMg::new_sbm` (persistent SBM handle) + the
  cylinder `SBM_GPU` flag (per-step pressure + velocity on device). Reproduces the CPU trajectory
  to 4 digits; ~35 s settled run (see above).

## Freely-moving SBM — DONE (settling disk, CPU + GPU, validated)
First moving-body increment (`sbm-moving-disk-check`). A heavy disk (ρs/ρf=4, explicit
Newton-Euler) settles in a closed no-slip box; each step rebuilds the surrogate at the new center,
solves the dual-splitting NS step with the body's RIGID velocity as the surrogate no-slip BC, and
advances `FreeBody::advance` from the recovered force/torque.
- New: `sbm_force_torque` (force vector + torque on the true circle, high-order Taylor recovery —
  generalizes `drag_x`); `ShiftedMultigrid::pcg_deflated` (singular closed-box pressure) + a
  trivial-RHS guard (a body from rest gives a zero step-1 RHS ⇒ unguarded α=0/0=NaN).
- GPU: `GpuPoissonMg::rebuild_sbm` re-uploads the moving surrogate onto a persistent handle each
  step (no per-step CUDA-context churn) + warm-started `solve_from`.
- **Full GPU settling (ny=16):** terminal velocity reached, **v≈−0.0505**, drag **Fy≈0.134 ≈
  f_net 0.1357** (98.7% force balance), ω≈0 (symmetric fall); 250 steps in ~91 s.
- **CPU↔GPU cross-check (ny=12, 25 steps):** identical trajectory to 4 digits (v=−0.0186,
  y=0.8994, Fy=+0.0905, ω=+0.001 on both). GPU ~3–6×/step (muted vs the fixed case because the
  per-step host MG setup — diagonal probing — is shared; amortizing it is the next lever).

Commits: `f9fc256` (CPU force/torque + Newton-Euler), `346f0be` (GPU rebuild_sbm + bin).
Next moving-SBM work: amortize the per-step MG setup (reuse while the active mask is unchanged),
strong coupling for light/neutrally-buoyant particles (added-mass), then many-body suspensions.

## Remaining (accuracy / features, later)
- Resolution sweep (ny) to confirm the high-order drag converges (order estimate vs penalization).
- Tune the SBM-MG smoother (96–108 iters is ~5× a clean p-MG's ~20 — functional but not optimal).
- The first-order *pressure* surrogate (natural-Neumann at the surrogate location) — a higher-order
  Neumann surrogate could shave the residual further.
- Moving boundary, viscoelastic surface BC, AMR surrogate faces, and the cut-cell DG backend.
