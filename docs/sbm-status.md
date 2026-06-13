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

## Built but BLOCKED
- **Step 2c — SBM dual-splitting NS cylinder** (`sbm-cylinder-check`, commit `e77899b`): full
  solver — convection + pressure projection (natural-Neumann surrogate) + SBM-Nitsche velocity
  no-slip + true-circle drag recovery. Velocity path + geometry sound (stable, umax≈0.33), but
  **the natural-Neumann pressure-Poisson under unpreconditioned CG STALLS** (hits maxit every
  step) — no converged C_D produced. Triple-confirmed at ny=12 (>2s/step, never past step ~1,
  even with warm-start + loose per-step tol).

## To resume (remaining 2c work)
1. **Fast solver for the SBM operator** — the real blocker. The whole SBM stack is **CPU +
   unpreconditioned CG**; the penalization path solves the same-sized pressure-Poisson fast
   because it has the GPU p-MG-PCG. Options: an MG preconditioner for the SBM operator, or a
   GPU port reusing the existing MG machinery.
2. **Debug the pressure stall** — hitting maxit (not just slow) hints at near-singularity in the
   natural-Neumann surrogate + inactive-identity spectrum; check correctness/conditioning of the
   `surrogate_neumann()` pressure operator (a small-problem eigen/condition check, or compare its
   apply to a reference) before/with the MG work.
3. Then the C_D run + comparison: SBM should converge to DFG `C_D = 5.5795`, beating
   penalization's ~2.1%-low plateau (`cylinder-drag-check`).
4. Later: moving boundary, viscoelastic surface BC, AMR surrogate faces, and the cut-cell DG
   backend (the second sharp-interface method per the research).

## Note
The pressure-stall finding fed directly into the next work stream: **gale's CPU path needs a
fast elliptic solve (preconditioner / CPU-MG) in general**, not just for SBM.
