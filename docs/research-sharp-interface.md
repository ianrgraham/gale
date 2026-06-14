# Sharp-interface embedded boundary for gale — implementation-design reference

**Status:** verified deep-research pass (2026-06-13). 5 angles → ~24 primary sources →
adversarial 3-vote verification. This is the *implementation-design* follow-up to
`docs/research-moving-particle-coupling.md` (which compared coupling methods); here we
design HOW to build a high-order **sharp** immersed boundary for gale's matrix-free GPU
nodal DG-SEM incompressible solver, to beat the diffuse volume-penalization accuracy floor
(validated: ~2% on the fixed Schäfer-Turek cylinder, ~13% terminal velocity on a small
moving settling disk — the diffuse mask gives the body an effective radius `r+√η_b`, so the
*flow* sees the wrong body).

---

## TL;DR — recommendation

**Prototype the Shifted Boundary Method (SBM) first; hold cut-cell DG as the
high-accuracy target.**

- **SBM** (Main & Scovazzi, JCP 2018) enforces the no-slip BC weakly (Nitsche) on a
  **surrogate boundary made of existing whole mesh faces** near the true interface, with a
  **Taylor-expansion correction** that recovers optimal high-order convergence. It **avoids
  cut-cell quadrature entirely** and is **provably immune to the small-cut-cell problem**.
  For gale this is the fastest robust path to a *sharp moving* boundary on the **existing**
  matrix-free GPU machinery — no per-cell quadrature generation, no agglomeration, no new
  geometry kernels.
- **Cut-cell / XDG DG** is the rigorously high-order (`h^{k+1}`) target with **exact
  surface-stress force/torque recovery**, but it is heavier and carries a real GPU risk
  (below). Adopt it later, where exact forces matter most.

The decisive factor for our GPU stack: cut-cell **quadrature generation** (root-finding +
small linear solves + Caratheodory pruning, data-dependent branching) is **not verified to
be a libdevice-free, per-cell-parallel GPU kernel** — it would likely run **host-side and
upload per timestep**, which fights the matrix-free / CUDA-graph design. SBM has no such
generation step. That asymmetry, plus SBM's small-cut-cell immunity, is why it goes first.

---

## The two viable families

### A. Cut-cell / embedded-boundary DG (XDG)
- **Accuracy:** optimal `h^{k+1}` for curved/immersed boundaries (cylinders, airfoils) via a
  level-set description, even with discontinuous coefficients (BoSSS/Kummer). *[high, 2-1 —
  the curved-cylinder high-order result is an attributed citation, not the paper's own
  incompressible study.]*
- **Quadrature (the core primitive):** two strong options, both giving **interior-volume AND
  embedded-surface** rules from one level-set cut cell — exactly the dual quadrature needed
  for the operator *and* for sharp surface-stress force/torque recovery:
  - **Saye / Algoim** (SISC 2015): dimension-reduction to a height function + 1D Gaussian
    quadrature; **strictly positive weights**, arbitrarily high order (order `2q`; ~8th at 4
    nodes; demonstrated to 20th–22nd); **restricted to quad/hex — which matches gale's
    meshes**. *[3-0]*
  - **Hierarchical Moment Fitting (HMF)** (Müller/BoSSS): fixes node positions, solves a
    small linear system for weights; all cell types; **MIT-licensed reference impl**
    (`FDYdarmstadt/MomentFitting`). *[3-0]* (Note: HMF was *not* shown to beat
    sub-triangulation in accuracy — claim refuted 0-3 — so prefer Saye on our quad meshes.)
- **Small-cut-cell problem (the defining liability):** tiny cells inflate the SIPG/Nitsche
  penalty (`η ~ k²/h'`) → severe ill-conditioning — precisely what gale's SIPG+Nitsche
  operators would hit. *[3-0]*
- **Remedy = cell agglomeration** (BoSSS/XDG, *preferred*): merge cells with volume fraction
  below a threshold `0≤α<1` into the max-volume-fraction edge-neighbor. **Triple-purpose and
  uniquely suited to moving particles:** removes small cells, **freezes per-timestep topology
  changes** as the body moves (new/vanished cells), **and builds the multigrid hierarchy**.
  *[3-0]*

### B. Shifted Boundary Method (SBM)
- Reformulates the BVP on a **surrogate domain of whole, uncut elements**; the BC is moved to
  the surrogate boundary and made consistent with a **Taylor-expansion correction** applied
  weakly via Nitsche. **Without the correction → only 1st order; with it → optimal high
  order.** **Immune to the small-cut-cell problem by construction.** *[3-0, but single
  primary source — Main & Scovazzi JCP 2018 — not independently corroborated here.]*
- For gale: no cut-cell quadrature, no agglomeration, reuses the existing element quadrature
  and operator structure → the **least new machinery** and the most GPU/matrix-free-friendly.

### C. Others
Immersed-interface (IIM), Nitsche-XFEM, ghost-fluid — surfaced but not competitive with A/B
for this framework in the surviving evidence. **CutFEM / ghost-penalty unfitted FEM**
(Burman–Hansbo–Massing) is the mathematically most-mature relative — a different (continuous-FEM)
philosophy, but its *ghost-penalty* stabilization is the idea to borrow if cut-cell DG conditioning
ever bites. Full landscape + the **fallback decision matrix** for the uncharted work
(freely-moving / many-body / viscoelastic) is in `docs/embedded-boundary-methods.md`.

---

## What is NOT de-risked (read before building)

- **No source demonstrates the full target workflow:** incompressible **dual-splitting
  projection** + Nitsche embedded no-slip + **viscoelastic (Oldroyd-B/Giesekus/FENE-P) stress
  BCs** on a **moving** sharp boundary. Demonstrated results are scalar elliptic/Poisson
  (Saye), inviscid compressible (Müller), or hyperbolic (Taylor & Chan). Composing Nitsche
  with our **SIPG penalty** and the **singular pure-Neumann pressure-Poisson** is genuine
  research.
- **GPU-friendliness of cut-cell quadrature generation: UNVERIFIED** (refuted 1-2). Likely
  host-side generation + upload per step → cost + CUDA-graph compatibility unknown. *Strong
  argument for SBM first.*
- **Persistent p-MG-PCG + CUDA-graph reuse under a moving operator:** addressed only
  indirectly (agglomeration freezes per-step topology and builds the MG hierarchy); whether
  handles can be reused vs partially rebuilt each step, and at what cost, is **unmeasured**.
- **SBM specifics for gale untested:** high-order surrogate→true-boundary force/torque
  recovery, the Neumann/pressure transfer for our projection, and viscoelastic surface stress
  BCs — the SBM evidence covers only Poisson/Stokes and flags high-order Taylor-extrapolation
  conditioning.

These four are the implementation risk register — they are the real work, and they're why we
prototype on the cheapest method (SBM) first.

---

## Incremental build path

1. **SBM geometry (host, validatable):** given the circle level-set, classify elements
   in/out/surrogate; build the surrogate boundary (mesh faces adjacent to the true interface)
   and, per surrogate face quadrature point, the distance vector `d` to the true boundary
   (for the Taylor correction). Validate the surrogate set + `d` against analytic geometry.
2. **SBM Nitsche no-slip, fixed body, Stokes:** add the Nitsche velocity-BC terms on the
   surrogate boundary (1st-order first, then the Taylor correction for high order) to the
   existing SIPG Helmholtz/operator. Confirm the deflation/compatibility of the singular
   pressure-Poisson survives the embedded pressure BC.
3. **Validate vs the cylinder benchmark:** reproduce **Schäfer-Turek 2D-1, C_D=5.5795**
   (already wired as `cylinder-drag-check`) and show SBM **converges to it** — the direct test
   that the sharp boundary beats the penalization ~2% floor, and the small moving-disk ~13%.
4. **Moving body:** recompute the surrogate set + `d` each step (cheap, host); reassess
   p-MG-PCG handle reuse vs rebuild (open question 3).
5. **Sharp force/torque recovery** on the true boundary; then **viscoelastic** surface stress
   BC (open question 4).
6. **Only if SBM is insufficient** (accuracy/robustness for the projection or VE): build
   **cut-cell DG** = Saye/Algoim quadrature (quad-native, positive weights) on quad cells +
   **cell agglomeration** (small cells + moving topology + MG hierarchy) + Nitsche. Expect
   host-side quadrature generation + upload; re-evaluate the CUDA-graph story.

Each step is independently validatable; steps 1–3 settle whether SBM alone closes the
benchmark gap before any cut-cell investment.

---

## Primary sources
- Kummer et al., *BoSSS: a package for multigrid extended DG* / XDG — agglomeration,
  `h^{k+1}`, MG hierarchy: ScienceDirect S0898122120301917; Wiley nme.70013.
- Saye, *High-Order Quadrature on Implicitly Defined Surfaces and Volumes in Hyperrectangles*,
  SISC 2015: math.lbl.gov/~saye/96629.pdf; Algoim: algoim.github.io; arXiv:2105.08857.
- Müller et al., **Hierarchical Moment Fitting** (cut-cell quadrature), Wiley nme.4569;
  MIT-licensed impl: github.com/FDYdarmstadt/MomentFitting.
- Taylor & Chan, entropy-stable SBP-DG on cut elements: arXiv:2412.13002.
- Main & Scovazzi, **Shifted Boundary Method**, JCP 2018: doi 10.1016/j.jcp.2017.10.026.

*Verification: adversarial 3-vote per claim; cut-cell findings 3-0 (convergence 2-1), SBM 3-0
on a single primary; three sub-claims refuted (cut-cell-quadrature GPU-friendliness 1-2; HMF
DG-order 1-2; HMF>sub-triangulation 0-3). Machine record: workflow wf_9f63a71a-a5e.*
