# CPU flow-path performance — findings & changes (2026-06-13)

Goal (user): make gale's **CPU** flow path fast *without over-engineering*; the user's
hypothesis was that **core utilization** is the primary lever. Data tool: `gale-gpu`'s
`cpu-profile` bin (`CPU_N` sets the grid; CPU-only, uses `gale::dg`).

## What the profiling actually showed

1. **Cores are already well used.** A bare `Poisson::apply` and the CG/MG solves run across
   all physical cores (`/usr/bin/time -v` "Percent of CPU" ≈ 3900–7200%). SMT (128 logical vs
   64 physical) buys ~nothing for this FP-dense DG work. Core utilization is *not* the lever.

2. **The lever is the pressure-solve ALGORITHM.** The dual-splitting pressure-Poisson is the
   dominant per-step elliptic cost, and it was solved with **unpreconditioned CG**, whose
   iteration count grows `O(1/h)`. The real operator is the **singular pure-Neumann** pressure
   (closed box / no outflow), solved with *deflated* CG. Measured, deflated CG vs the existing
   (but unused on CPU) **deflated p-MG-PCG**:

   | grid (p=4) | deflated CG | deflated MG-PCG | speedup | CG iters | MG iters |
   |-----------:|------------:|----------------:|--------:|---------:|---------:|
   | 8²         | 1944 ms     | 2348 ms         | 0.8× (loss) | 369  | 21 |
   | 16²        | 4925 ms     | 4153 ms         | 1.2×    | 739      | 22 |
   | 32²        | 14143 ms    | 5669 ms         | 2.5×    | 1444     | 22 |
   | 64²        | 50347 ms    | 10314 ms        | **4.9×**| 2773     | 22 |

   MG-PCG iterations are **mesh-independent (~22)**; CG grows `O(1/h)`. The win grows with size;
   below a **crossover (~16²)** the V-cycle overhead makes MG-PCG a *net loss* on small meshes.

## Changes made

- **`PMultigrid::pcg_deflated`** (`src/dg/operators/multigrid.rs`): deflated MG-PCG for the
  singular pure-Neumann pressure (removes the constant from `r` and from `z = M⁻¹r` each iter).
- **Singular coarse-solve fix** (same file): the coarsest-level CG now deflates its residual
  when the operator is singular (new `singular` flag, set at construction). Without this the
  singular coarse solve returned a **nullspace-polluted correction** that *stalled* the outer
  deflated PCG for divergence-type RHS (it converged for smooth MMS RHS, masking the bug).
  This mirrors the already-validated GPU `deflate_c!` in `gale-gpu`'s `poisson_pcg_solve`.
- **Wired MG-PCG into the CPU pressure solve** (`src/dg/operators/stokes.rs`,
  `project_and_diffuse`): a persistent `pressure_mg: Option<PMultigrid>` (built once via
  `PMultigrid::from_mesh`) drives `pcg` (outflow ⇒ non-singular) or `pcg_deflated` (closed box ⇒
  singular); falls back to plain/deflated CG when absent. **Velocity Helmholtz keeps plain CG**
  (the profile shows MG-PCG is *slower* there — the mass term makes it well-conditioned, ~80 CG
  iters). Viscoelastic (`OldroydB`/`Giesekus`/`FENE-P`) inherits this for free via
  `Stokes::step_ns_forced`.
- **Size gate** (`PRESSURE_MG_MIN_ELEMENTS = 1024`): build the MG preconditioner only at/above
  the crossover. Small meshes (all current unit/oracle tests: 4×4, 5×3, 16²…) use CG exactly as
  before — **byte-identical solver path, no regression** (verified: `uniform_flow_through_outflow`
  is 32 s on both baseline and this branch). Real sims (32²+) get the 2.5–4.9× win.

## The apply allocation pass — DONE

Diagnosis (from the V-cycle profile): one V-cycle was ~27× a single `apply` (205 ms vs 8 ms at
64²), and the cause was **not** operator reconstruction (~0.5 ms, negligible). The matrix-free
`Poisson::apply` was **allocation-bound, so p-coarsening bought no per-apply speedup** — a p=1
apply on 4096 elem (9.6 ms) cost *more* than p=4 (8.1 ms), because the old path used THREE
separate rayon passes (`elem_grads` → `volume_with_grads` → face records), and `grad_x`+`grad_y`
each recomputed `diff_r(ue)`/`diff_s(ue)` (so the same two sum-factorizations ran twice), all
allocating fresh `Vec`s per element.

**Fix (the "fused" pass):** `Poisson::apply` now does ONE parallel pass that, per element,
computes the physical gradients into FLAT `gx`/`gy` buffers (shared with the face terms below)
AND the volume stiffness `r[e]`, sharing a single `diff_r`/`diff_s` pair and reusing per-thread
scratch (`for_each_init`, no per-element allocation). Added scratch-writing `diff_{r,s,r_t,s_t}_into`
to `Reference2dQuad` (single source of truth; the allocating versions delegate). **Bit-for-bit
identical** to the prior path (same per-element arithmetic; IEEE `+` is commutative) — guarded by
the SIPG symmetry / MMS / harmonic-patch / non-conforming tests, all green.

Measured (64², p=4, quiet machine), old → new:

| apply level | old | new | gain |
|---|---:|---:|---:|
| p=4, 4096 elem | 8.11 ms | 6.06 ms | 25% |
| **p=1, 4096 elem** | 9.60 ms | 6.93 ms | 28% (and no longer > p=4) |
| p=1, 256 elem | 1.82 ms | 1.03 ms | 43% |
| p=1, 16 elem (floor) | 1.18 ms | 0.84 ms | 29% |
| **V-cycle** | **205 ms** | **139 ms** | **32%** |
| `Poisson::apply` | 7.41 ms | 6.08 ms | 18% |

The V-cycle (dominated by the coarse p=1 applies, which got the biggest win) is **32% cheaper**,
so the deflated MG-PCG pressure solve at 64² is now ~6.1× faster than deflated CG (was ~4.9×).

**Tried and reverted:** a second stage gave the face-record loop per-thread scratch (a
`lift_t_into` + `FaceScratch`/`map_init`) to remove the remaining per-face `Vec`s. On a quiet
machine it measured *within noise* of the fused pass alone (the face allocations weren't the
bottleneck once the gradient `Vec`-of-`Vec`s and the redundant `diff_*` were gone), so it was
reverted to keep the hot path simple (per "without over-engineering").

**Not done:** the 3D operator (`poisson3d.rs`) still uses the old allocating pattern — the CPU
flow V-cycle is 2D, so 3D was out of scope; the same fusion would apply there if 3D becomes hot.
