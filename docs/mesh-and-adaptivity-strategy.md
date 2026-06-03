# Mesh & Adaptivity Strategy

**Design guidance for `gale`'s mesh model, element types, boundary treatment, and
adaptivity — oriented toward DG-on-GPU for viscoelastic / incompressible flow with
first-class immersed boundaries.** Companion to
[`dg-gpu-fluid-simulation.md`](./dg-gpu-fluid-simulation.md).

> **Confidence:** **[V]** = backed by the verified research pass (see the DG doc's
> References); **[E]** = engineering/domain judgment. *Last updated 2026-06-01.*
>
> **Headline application target (drives these choices):** viscoelastic
> **particle-laden microfluidic suspensions** — deformable elastic/viscoelastic
> particles in a (possibly viscoelastic) matrix. This makes **IBM + AMR
> first-class** (§9). The material/method taxonomy is §8. See memory
> `gale-application-target`.

---

## 1. Two independent axes of "unstructured"

"Unstructured" conflates two things that have very different consequences:

- **Connectivity** — structured grid vs. arbitrary element adjacency. DG handles
  arbitrary connectivity *for free*: elements communicate only through numerical
  fluxes on shared faces, so a face-based mesh with arbitrary neighbors is the
  native model. No global-structure assumption exists to break. **[E]**
- **Element shape** — tensor-product (quad/hex) vs. simplex (tri/tet) vs. polytope
  (Voronoi/agglomerated). This is where GPU performance is decided:
  - **Quad/hex:** retain the tensor-product reference map → **sum factorization
    applies** → high arithmetic intensity, the core GPU win. **[V]**
  - **Simplex:** maximal meshing flexibility (mature generators), but **tensor
    structure is lost** → dense per-element operators, lower intensity.
    Bernstein–Bézier bases recover near-optimal complexity *at high order*. **[V]**
  - **Polytope:** see §6.

**Sweet spot: unstructured *hex/quad* meshes** — arbitrary connectivity *and*
per-element tensor-product structure (each hex maps to the reference cube). This is
how libParanumal supports tri/quad/tet/hex from one framework. **[V]** It keeps
unstructured flexibility without surrendering sum factorization.

> **The standing tension:** the GPU win is strongest on tensor-product elements at
> high order **[V]**. Geometric flexibility (simplices, polytopes) costs
> throughput. Prefer unstructured hex/quad; reserve simplices/polytopes for
> geometry that forces them. **[E]**

---

## 2. Boundary treatment: body-fitting vs. immersed

### Static body-fitting — **yes, support it**
For fixed complex geometry, body-fitted curved meshes are the accuracy gold
standard, and the reason is gale-specific: **near-wall and stagnation-region stress
is where viscoelastic sims live and die (HWNP)**. A body-fitted mesh clusters
resolution and aligns high-order elements with the boundary layer and stress
gradients; an immersed method smears the interface across a background grid and
degrades near-wall stress exactly where it matters most. **[E]**

The canonical viscoelastic benchmarks — **4:1 contraction, flow past a fixed
cylinder, cross-slot** — are all fixed geometry, so static body-fitting covers
gale's most important validation cases *and* its hardest numerical regime. On the
solver side it is "just a conforming curved mesh," which the face-based abstraction
(§7) already supports; the only real cost is **offline** high-order *curved* mesh
generation (curving near the body without element tangling — a Gmsh / mesh-curving
pipeline problem, not a solver problem). **[E]**

### Dynamic body-fitting — **avoid as the moving-body workhorse**
- **ALE** (mesh deforms with the body): keeps body-fitted accuracy for
  *small/moderate* motion, but adds a mesh-motion solver, the **Geometric
  Conservation Law** burden (free-stream preservation under mesh motion — a subtle
  correctness pitfall), progressive mesh-quality decay, and **no ability to handle
  topology changes** (contact, large rotation, many bodies). **[E]**
- **Remeshing / overset**: the real dealbreaker for gale. Dynamic mesh generation
  is irregular and GPU-hostile; **solution transfer between meshes** each remesh
  costs accuracy and conservation; and **re-partitioning across GPUs every remesh
  destroys the validated performance model** (static partition, persistent device
  buffers, P2P halo exchange — all assume a fixed mesh). **[E]**

### Immersed boundaries — the moving-body answer
- **Volume penalization (forcing):** element-shape-agnostic, mesh stays fixed, body
  is a mask field, topology changes are free, the static-partition GPU model
  survives. Pairs naturally with AMR-around-the-body. **Recommended moving-body
  default.** **[V for VP+DG; E for the synthesis]**
- **Sharp cut cells:** higher near-wall accuracy, but the small-cut-cell explicit
  time-step restriction must be mitigated, and on a moving body the cut geometry is
  recomputed each step. **[V]**

### The honest tension (no free lunch)
Immersed methods smear the wall → worse near-wall viscoelastic stress → potentially
worse HWNP precisely where it bites. For **moving viscoelastic walls** this is a
genuine compromise. Claw-back options: high-order IBM / cut-cell with careful
interface treatment, or AMR banded at the interface — *not* dynamic remeshing. **[E]**

### Decision matrix
| Geometry | Choice |
|---|---|
| Fixed, complex (contraction, fixed cylinder, cross-slot) | **Static body-fitted, curved** |
| Moving body | **Immersed (volume penalization) + AMR** |
| Small-deformation FSI | ALE — *optional later feature*, not a foundation |
| Best-of-both | Static body-fitted fixed domain **+** immersed moving objects inside |

---

## 3. Per-cell diagnostics — anticipating trouble

Compute these as a cheap per-element pass (mostly local reductions over an
element's nodes — GPU-friendly). They serve triple duty: **limiter input,
AMR refine/coarsen criteria, and HWNP early warning.** **[E]** except where noted.

### Viscoelastic / HWNP health
- **Min eigenvalue / determinant of the conformation tensor** — SPD/positivity
  check; det→0⁺ or a negative eigenvalue in the *reconstructed* field is HWNP onset.
  (Log-conformation protects the *evolved* variable; monitor the reconstruction.)
  Log-conformation alleviates but does not eliminate HWNP. **[V for LCR; E for the diagnostic]**
- **Local Weissenberg number** Wi_loc = λ·(local strain rate) — where elasticity spikes.
- **Trace(conformation)** vs. FENE-P limit L² — extensional stretching toward the
  finite-extensibility wall (or unbounded growth for Oldroyd-B).
- **Conformation condition number** — anisotropy / stretching severity.

### DG numerical health
- **Modal decay rate (Persson–Peraire indicator)** — per-element high-order modal
  energy should decay; flat/growing spectrum = under-resolution or Gibbs
  oscillation → troubled-cell flag for limiting *or* refinement. The single most
  useful DG diagnostic.
- **Inter-element face-jump magnitude** — natural DG error estimator; drives h/p
  refinement.
- **Local CFL / cell Péclet** — stability / stiffness flags.

### Cell geometry quality (critical once curved / moving / AMR are in play)
- **Min scaled Jacobian** — standard quality metric; ≤ 0 = tangled/invalid element
  (the failure mode of moving meshes and curved AMR).
- **Aspect ratio / skewness** — distortion that amplifies the viscoelastic
  conditioning issues above.

### Immersed boundary
- **Cut / volume fraction** per cell (cut-cell time-step flag), or **signed distance
  to interface / penalization band width** (volume penalization).

---

## 4. Adaptive mesh refinement (AMR)
DG is arguably the *best-suited* method for AMR: element-locality means
non-conforming / hanging-node interfaces are handled by the same numerical-flux
(mortar) machinery — far easier than continuous FEM. **h** (split), **p** (raise
order), and **hp** adaptivity all fit. References: p4est (forest-of-octrees),
Trixi.jl, deal.II, MFEM. GPU AMR adds dynamic data structures + load rebalancing
but is well-trodden. **[E]**

The §3 diagnostics are the refinement criteria; design AMR and diagnostics together.

## 5. Moving objects + adaptivity
**Volume penalization + dynamic refinement around the body** is far more tractable
than body-fitted remeshing: the body is a mask, AMR refines the band around it, and
the body is re-flagged each step — topology changes are free and the GPU partition
model survives. Body-fitted ALE is the accuracy alternative but only for small
deformation (§2). **[E]**

## 6. Exotic elements (Voronoi / polytopal) — later, research-grade
Needs **polytopal DG (agglomeration-based PolyDG)** or **Virtual Element Methods**
(Cangiani/Houston; Beirão da Veiga). Two costs: (1) quadrature on arbitrary
polytopes (sub-tessellation), and (2) **no tensor structure → dense operators → the
weakest GPU story** (no sum factorization). Upside: natural coarsening by
agglomeration (nice for AMR). **Verdict: a later extension, not a foundation;** keep
the abstractions general enough (§7) that it *could* be added. **[E]**

---

## 7. Architectural recommendations — keep the doors open

Keep three things abstract from the start; this preserves everything above without
building it now. **[E]**

1. **Mesh = elements + faces, arbitrary connectivity.** Face → (two elements +
   orientation). Supports structured, unstructured, hybrid, hanging-node (AMR),
   body-fitted, and immersed (mask field) with no rework.
2. **Per-element `{reference type, geometric map / Jacobian, basis, quadrature,
   polynomial order}`.** Order and type are *per-element data*, not global
   constants → enables mixed meshes and p-adaptivity.
3. **Operator evaluation behind a trait.** A tensor-product / sum-factorization
   backend (quad/hex — the GPU fast path) and a dense backend (simplex/polytopal)
   coexist.

**Then start concrete and narrow:** conforming, single-order, affine→curved
**quads in 2D** (Milestones 0–3); volume-penalization IBM (Milestone 4);
log-conformation viscoelastic (Milestone 5). Add static body-fitted curved meshes
for the fixed-geometry viscoelastic benchmarks; defer AMR, ALE, and polytopal until
the core is proven.

---

## 8. Phases, materials & method families

The methods above (body-fitting, immersed boundaries, two-phase, ...) are not
competing alternatives — they partition by **what kind of material the secondary
region is**. One question organizes the whole space. **[E] throughout.**

### The litmus test: does the material return to a reference shape?
Every immersed or secondary material sits on a spectrum, and *one* property decides
the method family: **does it elastically recover an undeformed reference
configuration?**

- **Yes → it is a SOLID** (rigid · elastic · viscoelastic-*solid* · elastoplastic).
  It stores elastic energy relative to a rest shape, so track *material
  deformation* (Lagrangian) and couple with the **immersed-solid methods** (§2):
  rigid volume-penalization / cut-cell / Peskin force-spreading / fictitious-domain.
- **No → it is a FLUID** (Newtonian or viscoelastic-*fluid* drop). No rest shape; it
  flows indefinitely and memory fully relaxes. Track the *interface* (Eulerian phase
  field or Lagrangian front) — this is **two-phase flow** (§8.2), a different family.

So immersed-solid methods are for rigid (movable) bodies and deformable-but-solid
bodies. A particle that is *itself a fluid* (a droplet) is a two-phase problem, not
an immersed-solid one. Don't force a fluid drop through IBM, and don't track a
flowing material with a reference configuration.

### 8.1 The elastic → plastic → fluid continuum
Real soft matter spans the boundary:
`elastic → viscoelastic-solid → elastoplastic → yield-stress fluid (Bingham / Herschel–Bulkley)`.
The dividing line is still the reference config:
- **"Mostly elastic, a bit plastic"** → **solid** with an *evolving* reference
  configuration (plastic flow permanently updates the rest shape). Handled by the
  immersed-solid methods with an **elastoplastic** stress update (yield criterion +
  return-mapping). ✅
- **"Yields and then *flows*"** (gels, pastes, yield-stress fluids) → no persistent
  reference shape above yield → treat as a **non-Newtonian fluid phase**, not an
  immersed solid.

### 8.2 Two fluids (VE+Newtonian, VE+VE) → two-phase flow, not IBM
A two-fluid system is a multiphase problem. For gale's stack (high-order DG, GPU,
AMR, viscoelastic phases, droplets that break/merge) the natural choice is
**phase-field / diffuse-interface (Cahn–Hilliard)**: a *smooth* PDE (pairs with
high-order DG — no sharp reconstruction), handles **topology change automatically**,
and is conservative. Extend to two viscoelastic phases by carrying a constitutive
model per phase, blended by the phase indicator (log-conformation each phase), with
AMR banded on the interface. Alternatives — level-set (mass loss, needs reinit),
VOF (low-order interface reconstruction fights high-order DG), front-tracking
(sharp but manual topology surgery). VE+Stokes is the special case of one phase
having relaxation time → 0. *(Two-phase **viscoelastic** flow is hard, active
research — interfacial stress singularities — but coherent with the rest of gale.)*
**This choice is now [V]-confirmed — see §8.5.**

### 8.3 The capsule hybrid sits on the seam
A capsule/cell is an elastic **membrane** (solid, reference config → IBM) enclosing
an **interior fluid** distinct from the exterior (→ two-phase). So gale's
particle-laden target already straddles both families: membrane elasticity is an
immersed-solid concern *and* the interior/exterior contrast is a two-phase concern.
Design for the combination.

### 8.4 The unification (architecture)
All of this collapses to: **everything is a constitutive model on a region, with
the region's boundary tracked by some method.** Two pluggable traits over the shared
DG + AMR + diagnostics core (extends §7):

1. **Constitutive model** — shared across fluids *and* solids: Newtonian,
   Oldroyd-B / FENE-P (VE fluid), neo-Hookean (elastic), viscoelastic-solid,
   elastoplastic, yield-stress.
2. **Region / interface representation** — rigid body · deformable solid (reference
   config + Peskin/fictitious-domain) · fluid phase (phase-field/level-set/front) ·
   composite (membrane + interior fluid).

Build order stays incremental: single-phase VE fluid → rigid immersed → deformable
elastic capsule → then **two-phase (phase-field)** and **elastoplastic** as
*parallel later modules* (each a separate constitutive/coupling effort, not a
rewrite).

### 8.5 Verified method selection (research pass, 2026-06-01)

A dedicated research pass (25 sources, 25 claims verified, 0 refuted) confirmed the
method choices above. *(The pass's auto-synthesis output was malformed; the
conclusions below are reconstructed from its verified-claim log + source list — so
treat the **specific citations** as solid and the **phrasing** as my reconstruction.)*
**[V]** = a claim that passed 3-0; **[V, 2-1]** = passed 2-1 (weaker).

**Two-phase interfaces → phase-field (Cahn–Hilliard) on DG is confirmed the right
choice.**
- Phase-field two-phase flow is established on DG / coupled with incompressible
  NS–Cahn–Hilliard, with **mass-conservative, monotone, discretely energy-stable**
  DG/LDG pressure-correction schemes. **[V]** (Manzanero et al. DG phase-field,
  arXiv:1910.11252; S0021999120301376; S0021999124002328; arXiv:2003.12373.)
- **VOF is prone to excessive mass/accuracy error** and fights high-order; **coupled
  level-set/VOF (CLSVOF), incl. adaptive ACLSVOF**, is the better sharp-interface
  hybrid; level-set alone loses mass; surface tension in LS/VOF via continuum-surface-
  force (CSF). **[V / V,2-1]** (Lowengrub ACLSVOF; arXiv:1307.8248.)

**Deformable particles → immersed-FE / DLM (or composite-B-spline Peskin); the
Peskin accuracy penalty is real.**
- The **classical Peskin IBM is ~1st-order at the interface** — the penalty we
  flagged is confirmed; **composite B-spline regularized delta functions** restore
  higher near-interface accuracy. **[V]** (S0021999116302649.)
- **Distributed-Lagrange-multiplier (DLM) and immersed-finite-element (FE-IBM)
  methods are proven for deformable-particle suspensions in *viscous AND
  viscoelastic* media** — directly gale's case. **[V]** (RG "Immersed-finite-element
  method for deformable particle suspensions in viscous and viscoelastic media";
  S0045782520305776.) A **high-order IBM via volume penalization for FSI** also
  exists (arXiv:2512.05733, cited earlier).
- **Capsule = RBC/vesicle membrane model**; the interior/exterior contrast (different
  viscosity/constitutive model) is the two-phase aspect layered under the membrane.
  **[V]** Spectral boundary-integral handles blood cells beautifully but is
  **Stokes-only** (consistent with §8.2: a viscoelastic matrix rules BIM out).

**Viscoelastic phases → log-conformation, confirmed.** Viscoelastic two-phase /
high-Wi-at-interfaces is handled with the log-conformation tensor technique;
adaptive (AMR) viscoelastic incompressible solvers exist. **[V]**
(S0377025718300752; S0009250911005422; **rheoTool**; a fully-coupled high-order DG
viscoelastic solver, RG 347539614.)

**Reference codes to mine:** **Basilisk** (VOF + octree **AMR**, the closest
adaptive-two-phase reference); **HemoCell** (IB-LBM **RBC/cell suspensions**);
**rheoTool** (OpenFOAM viscoelastic: log-conf, SRCR, many models).

**Still open / under-verified (carry as risks):**
- **Elastoplastic immersed solids + the solid↔fluid criterion specifically in
  high-order DG** was *not* strongly settled by the pass — the §8.1 litmus stands as
  reasoned **[E]**, not verified. (Touched by arXiv:1803.09563, S0377025718301162.)
- **GPU dynamic load balancing for moving-body AMR**, and **IMEX for stiff membranes
  at low Re**, remain open (also flagged in `implicit-solver-strategy.md` §6).
- **Meta-caveat (verified):** *no single code* combines DG + AMR + GPU + deformable
  viscoelastic particles + viscoelastic two-phase — the integrated stack is research
  frontier; gale assembles separately-validated parts.

---

## 9. Deformable & particle-laden flows (first-class target)

gale's headline application — **viscoelastic particle-laden microfluidic
suspensions** — escalates the immersed-boundary requirement beyond rigid bodies.
This section captures what that implies; method *selection* for the deformable
stage still wants a verified research pass. **[E] throughout unless noted.**

### "IBM" is two different families
- **Rigid** bodies: cut-cell / **volume penalization** (forcing) — §2. The mesh is
  fixed; the body is a mask / prescribed motion.
- **Deformable** bodies (capsules/cells/soft particles): **Peskin-style IBM**
  (Lagrangian membrane markers spread elastic force to the Eulerian fluid,
  velocity interpolated back — the deformable-capsule workhorse), or
  **fictitious-domain / fully-resolved FSI**. Different machinery, same
  "background Eulerian mesh + immersed body" philosophy.

So **two pluggable axes**, both feeding §7's abstractions:
1. **Coupling mechanism** — rigid VP · Peskin force-spreading · fictitious-domain · cut-cell.
2. **Body physics** — rigid · elastic membrane · viscoelastic solid.

### Why this justifies the volumetric DG path
Boundary integral methods are the efficient choice for *Stokes* (Newtonian)
capsule suspensions but are **restricted to linear bulk physics**. A **viscoelastic
matrix** rules them out → a volumetric solver (DG/FEM/FV/LBM) is required. The
target's own physics justifies the DG-on-GPU approach.

### The central tension (decide at the deformable stage)
High-order DG ↔ deformable interfaces: Peskin force-spreading is inherently
~1st–2nd order at the interface, capping the high-order payoff exactly where the
particle dynamics live. Sharp-interface / fictitious-domain DG preserve order but
are harder and more research-frontier. **This method choice is non-obvious and
should be settled by a verified research pass, not assumed.**

### Extra per-particle diagnostics (add to §3)
- **Deformation / Taylor index**, **membrane stress** — particle health.
- **Enclosed area/volume drift** — the classic Peskin-IBM conservation watchdog.

### Staged build (keeps the vision; validate each layer)
1. Rigid VP particles in **Newtonian** fluid **+ AMR tracking** — IBM+AMR machinery,
   particle motion (Newton's eqns), basic collision/lubrication.
2. **Viscoelastic matrix** (log-conformation) + rigid particles.
3. **Deformable elastic** capsules — add the Peskin/fictitious-domain coupling axis,
   membrane constitutive model, two-way coupling, volume-conservation care.
4. **Viscoelastic** particles — constitutive model *inside* the deformable particle.

> **Honest scope note:** the full stack (DG + deformable IBM + AMR + viscoelastic
> matrix + viscoelastic particle, on GPU) is research-frontier — each piece exists
> in the literature, but the combination is novel. Build and validate layer by
> layer; don't attempt it monolithically.

### Low-Reynolds (microfluidic) consequence
At low Re the advection term is weak and the **implicit Stokes / elliptic solve
dominates** (the multigrid-preconditioned CG of the DG doc §4.1 becomes central),
and stiff membrane elasticity pushes toward **implicit/IMEX** coupling. The
explicit-RK emphasis of the early milestones shifts accordingly for this target.

---

## 10. Summary recommendation for gale

| Topic | Recommendation | Confidence |
|---|---|---|
| Mesh connectivity | Face-based, arbitrary (unstructured-capable) from day one | [E] |
| Primary element | Unstructured **hex/quad**, affine→curved (keeps sum factorization) | [V]/[E] |
| Simplices | Support via dense backend where geometry forces it; Bernstein at high order | [V] |
| Fixed complex geometry | **Static body-fitted, curved** (best for viscoelastic walls) | [E] |
| Moving bodies | **Volume penalization + AMR**, not dynamic remeshing | [V]/[E] |
| Dynamic body-fitting | Avoid; ALE only as later small-deformation FSI option | [E] |
| Material/method choice | Litmus = reference config? Solid→immersed-solid; fluid→two-phase (§8) | [E] |
| Deformable particles | **Immersed-FE / DLM** (or composite-B-spline Peskin); Peskin alone ~1st-order at interface (§8.5) | [V] |
| Two fluids (VE+N, VE+VE) | **Phase-field (Cahn–Hilliard)** on DG + per-phase log-conformation (§8.2, §8.5) | [V] |
| Elastoplastic / "a bit plastic" | Immersed solid w/ evolving reference (return-mapping); yield-stress *fluids* → fluid phase (§8.1) — *high-order-DG specifics under-verified* | [E] |
| Reference codes | Basilisk (VOF+AMR), HemoCell (RBC/IB-LBM), rheoTool (viscoelastic) (§8.5) | [V] |
| Diagnostics | Per-cell pass (§3): drives limiting + AMR + HWNP early warning | [E] |
| AMR | Design-for via §7 abstractions; implement after core | [E] |
| Polytopal/Voronoi | Keep door open via §7; research-grade later extension | [E] |

---

## As-built: AMR foundation (2026-06-03)

`src/dg/amr.rs` implements the `h`-adaptive operator toolkit (validated, CPU):
- **Where to refine** — `SmoothnessIndicator` (Persson–Peraire spectral decay):
  fraction of element L2 energy in the highest Legendre modes. Smooth field ≈0,
  pure top-mode =1, under-resolved oscillation ≈0.5.
- **How to refine** — `RefineQuad::{prolong, restrict}`: parent↔4-children transfer.
  Prolong exact for degree ≤ p; restrict is the conservative mass-weighted L2 adjoint
  (constant-preserving, cell-integral-conserving; round-trip machine-exact for low modes).
- **How to couple** — `RefineQuad::{mortar_to_fine, mortar_to_coarse}`: non-conforming
  face projection. A coarse↔2-fine mortar **flux** is verified **conservative**
  (coarse-edge integral = Σ fine half-edge integrals) and **consistent** (uniform field
  ⇒ uniform flux, no spurious interface jump).

**Not yet built (the large remaining step):** the `Mesh2d` connectivity refactor for
2:1 non-conforming neighbors (a `Neighbor::NonConforming` variant + 2:1 balance) and
wiring the mortar flux into the `Poisson`/`Hyperbolic`/`Stokes` face loops so the
adaptive solver actually runs. The operators above are its validated prerequisites.

## As-built: non-conforming (2:1) solver core (2026-06-03)

`src/dg/nonconforming.rs` — `NcAdvection`: a working linear-advection DG operator on
a 2:1 hanging-node mesh (one coarse element whose East edge faces two fine
neighbors). The mortar flux (compute on the fine resolution, project back to the
coarse edge with `RefineQuad::mortar_*`) is wired into the weak-form face loop.
Validated:
- **free-stream preserved** across the hanging node (uniform state ⇒ ‖∂ₜu‖ < 1e−10);
- **linear advection exact** across the hanging node (high order retained, < 1e−9);
- **conservation exact**: Σ Jw ∂ₜu equals the analytic outer-boundary flux to 1e−9
  (the discrete volume telescopes; the 2:1 interface cancels because coarse-Jacobian
  ½ × mortar-restrict ½ = fine-Jacobian ¼).

This proves the non-conforming coupling works in an actual solver. The remaining step
to a *general* adaptive solver is a `Mesh2d` constructor that refines arbitrary
elements (2:1 balance + a `Neighbor::NonConforming` variant) and the same mortar
wiring in the `Hyperbolic`/`Stokes`/`Poisson` face loops.

### Generalized (2026-06-03): `NcMesh` arbitrary Cartesian refinement

`NcMesh::cartesian_refined(order, nx, ny, .., refine)` builds a Cartesian mesh with
any set of cells single-level-refined, producing correct 2:1-balanced connectivity
(`NcNeighbor`: Boundary / Conforming / CoarseToFine / FineToCoarse). `advection_rhs`
solves linear advection over the whole mesh, mortar-coupling every 2:1 interface.
Validated on a 3×3 mesh with the centre cell refined (12 elements, 4 hanging-node
interfaces + 4 child-child conforming faces): free-stream preserved (<1e-9) and
linear advection exact (1.9e-13) everywhere.

Remaining for full integration: lift this into the shared `Mesh2d`/`Neighbor` enum so
the multi-law `Hyperbolic` (Euler/Burgers), `Stokes`, and SIPG `Poisson` run on
adaptive meshes (a cross-cutting refactor of their face loops), and dynamic
refine/coarsen driven by the `SmoothnessIndicator`.
