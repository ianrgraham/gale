# hp-Adaptivity & Immersed-Boundary Accuracy — Gap Analysis & Roadmap

**Status:** planning / deferred. Captured 2026-06-04 after reviewing Nayak & Mavriplis,
*"Immersed boundaries in the discontinuous Galerkin spectral element method through
hp-adaptivity"*, Computers & Fluids 2025 (HONOM-2024 special issue),
<https://www.sciencedirect.com/science/article/pii/S0045793025003007>.

This doc records what that paper does, where gale already matches or exceeds it, the
genuine feature gaps, and a concrete plan for what we'd add. **Nothing here is
scheduled yet** — it's the reference for when we return to adaptivity after the
AMR-with-GPU-flow work.

---

## 1. What the paper actually does

A focused method/accuracy study, narrower than its title:

- **Discretization:** nodal DG spectral-element method (DG-SEM), same family as gale.
- **IBM:** the **Brinkman volume-penalty method** — obstacles modelled as low-porosity
  porous media via a masking function χ(x,y) and a porosity parameter. This is *exactly*
  gale's `VolumePenalization` approach.
- **Adaptivity:** **hp-adaptive** — both `h` (element subdivision/agglomeration) **and**
  `p` (per-element spectral order), with **Hilbert space-filling-curve dynamic load
  balancing** for the adaptive mesh on HPC.
- **Physics / testbed:** the **2D acoustic wave equation** (linear hyperbolic), explicit
  (Euler-forward / RK) time stepping. *No* Navier–Stokes, Euler, or viscoelasticity —
  those appear only in the related-work citations.
- **Hardware:** **CPU/HPC** (results in "Core hours"; Hilbert-curve partitioning). A
  GPU body-fitted DG-SEM result (Tousignant) is used only as an external comparison
  baseline; the paper's own hp-adaptive IBM code is not GPU.
- **Contribution:** combining **low porosity + hp-refinement localized to the immersed
  boundary** suppresses the spurious oscillations and accuracy loss that plague
  volume-penalty IBM, localizing error to the boundary vicinity. Reported 83–88%
  CPU-time savings vs uniform refinement at matched error.

**One-line takeaway:** it's a recipe for making the (cheap, simple) volume-penalty IBM
*accurate* by throwing hp-refinement at the immersed boundary, demonstrated on a 2D
linear-acoustics testbed, on CPU.

---

## 2. Feature comparison vs gale

### Shared — gale already has these
| Feature | gale |
| --- | --- |
| Nodal tensor-product DG-SEM | ✅ `dg::reference`, `dg::quad`/`hex` |
| Brinkman volume-penalty IBM | ✅ `dg::VolumePenalization` (2D+3D), on GPU (`gale_gpu::penalize_apply`/`penalize3d_apply`) |
| h-adaptivity / AMR | ✅ 2D non-conforming (2:1 mortar) + `SmoothnessIndicator` + dynamic remap (CPU) |
| Explicit time integration | ✅ SSP-RK3 |

### Paper has it, gale does **not** (the genuine gaps)
| Gap | Why it matters | gale today |
| --- | --- | --- |
| **p-adaptivity** (variable per-element order) | Resolve thin features by raising order locally instead of paying uniform high-p everywhere | `Mesh2d`/`Mesh3d` carry a single uniform `order`; AMR is **h-only** |
| **hp-adaptivity** (combine h+p) | The paper's core result; h and p have complementary error-reduction regimes | absent (no p axis to combine) |
| **IB-targeted refinement** | Lock refinement to the immersed boundary, where volume-penalty error concentrates | `SmoothnessIndicator`-driven only; not coupled to the IB mask |
| **Hilbert space-filling-curve load balancing** | Keep adaptive meshes balanced across ranks/GPUs | block partitioning only; code comments mark a SFC partitioner as a "drop-in replacement" TODO (`dg::distributed`, `sim::device`) |
| **Volume-penalty accuracy/porosity characterization** | Know the porosity↔error↔order trade-off for our penalized IBs | we use penalization but haven't done the convergence/porosity study |

### gale has it, the paper does **not** (we already exceed it)
- **GPU** — entire stack incl. IBM; the paper's code is CPU.
- **Multi-GPU** (P2P halo exchange).
- **3D** — paper is 2D only; gale has the full 3D hex stack.
- **Incompressible Navier–Stokes / Stokes** — paper solves only the linear acoustic
  wave equation.
- **Viscoelastic** (Oldroyd-B + log-conformation, 2D+3D) — the headline goal, absent
  from the paper.
- **Elliptic solvers** (SIPG Poisson/Helmholtz CG, p-multigrid PCG) — not needed for
  their explicit hyperbolic testbed.

**Bottom line:** we overlap on the core (DG-SEM + volume-penalty IBM + h-AMR) and go far
beyond on physics, dimensionality, and hardware. We trail on exactly one axis they
specialize in: **p-/hp-adaptivity** (plus SFC load balancing for adaptive meshes).

---

## 3. What we'd add to gale

Ordered by value-to-gale and by dependency. Each builds on the existing `dg::amr`
machinery (`RefineQuad`/`RefineHex`, `SmoothnessIndicator`, mortar projections) and the
non-conforming solver core (`NcMesh`, mortar-coupled `Poisson`/`Stokes`).

### 3.1 p-adaptivity (variable per-element polynomial order) — the headline gap

**Goal:** allow each element its own order `pₑ`, not one global `order`.

**Why it's worth more to gale than to the paper:** our targets — immersed-boundary
layers *and* high-Weissenberg viscoelastic stress boundary layers — are exactly the thin,
localized features where raising `p` locally pays off, and we'd get the win on the GPU +
viscoelastic stack where uniform high-p is most expensive.

**Design sketch:**
- `Mesh2d`/`Mesh3d`: replace the single `order: usize` with per-element order (or a
  global default + per-element overrides). Reference elements become a small cache keyed
  by order (`Reference2dQuad`/`Reference3dHex` per `p`).
- **Mortar on the p-interface:** the existing `RefineQuad::mortar_*` already does
  L2-projection between non-matching edge traces for `h`-non-conforming faces; a
  `p`-non-conforming face (orders `pₗ ≠ pᵣ`) is the same machinery with a rectangular
  1D projection `P(pₗ→pᵣ)` between the two GLL traces. Generalize the mortar to a
  `(p_from, p_to)` projection; `h`-mortar becomes the special case.
- **Operators:** the SIPG `Poisson` and weak-form `Hyperbolic`/`Stokes` face loops
  already consume mortar projections; they need to look up per-element order and pick
  the right reference operators + mortar.
- **Indicator → action:** `SmoothnessIndicator` already says *how* under-resolved a cell
  is; the policy maps smooth-but-under-resolved → `p`-refine, sharp/discontinuous →
  `h`-refine (standard hp decision). Add a `p`-remap (order change with conservative L2
  re-projection of the state, analogous to `remap_scalar`).
- **GPU:** harder — the current GPU kernels assume a uniform `n1` (block_dim = nodes)
  baked into the launch. Per-element order means per-element `n1`, i.e. variable block
  sizes / grouping elements by order into separate launches (a "by-order batched" launch
  per `p`). Sequence p-adaptivity *after* the AMR-with-GPU-flow (h-non-conforming GPU)
  work so the GPU mortar plumbing already exists.

### 3.2 IB-targeted hp-refinement (the paper's actual contribution)

**Goal:** drive refinement from the immersed-boundary mask, not only the smoothness
indicator, so error concentrates and resolves at the obstacle surface.

**Design sketch:**
- A refinement criterion that flags elements the IB mask χ cuts through (0 < χ̄ₑ < 1) or
  that border them, and `h`- or `p`-refines a band around the surface.
- Fold into the existing `AmrUpdater` as an additional (composable) indicator alongside
  `SmoothnessIndicator` — `max`/union of the two criteria.
- Run the porosity↔order↔error study (§3.4) on a known case (cylinder/sphere) to pick
  defaults.

### 3.3 Space-filling-curve load balancing

**Goal:** keep adaptive meshes balanced across ranks/GPUs as cells refine/coarsen.

**Design sketch:**
- A Hilbert (or Morton/Z-order) index over leaf cells → contiguous SFC ranges per
  partition. Slots into `dg::distributed` / `sim::device::DomainDecomposition` where a
  "graph/space-filling partitioner is a drop-in replacement" is already noted.
- Needed for *dynamic* multi-GPU adaptive runs; not needed for single-GPU or static.

### 3.4 Volume-penalty accuracy / porosity characterization

**Goal:** quantify, for gale's penalized IBs, the porosity ↔ resolution ↔ error
trade-off the paper studies, so we choose `η_b`/porosity and local order with confidence
(and document the IBM's order of accuracy under refinement).

**Design sketch:** an MMS/known-solution study (flow past a cylinder/sphere) sweeping
porosity and (once §3.1 lands) local order, reported like the paper's CPU-time-vs-error
tables — but on our GPU + incompressible/viscoelastic stack.

---

## 4. Sequencing

1. **AMR-with-GPU-flow** (in progress, separate thread): get the GPU SIPG operators +
   flow solvers running on 2:1 `h`-non-conforming meshes (GPU mortar coupling). This is
   the prerequisite that puts the GPU mortar plumbing in place.
2. **p-adaptivity (§3.1)** on the CPU first (generalize the mortar to `(p_from,p_to)`,
   per-element order in `Mesh2d`), validated against MMS; then the GPU batched-by-order
   launch.
3. **IB-targeted hp-refinement (§3.2)** + **porosity study (§3.4)** — the paper's actual
   result, now on our richer stack.
4. **SFC load balancing (§3.3)** when dynamic multi-GPU adaptive runs are needed.

The physics gale already has (GPU, 3D, incompressible, viscoelastic, IBM) means landing
p-/hp-adaptivity would put gale strictly ahead of the reference paper on every axis.
