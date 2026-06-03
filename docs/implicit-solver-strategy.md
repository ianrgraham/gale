# Implicit Solver, Preconditioning & Time Integration Strategy

**The hardest part of gale: solving the stiff incompressible + viscoelastic
system at low Reynolds number on GPU.** This doc is the verified-research-backed
plan for the formulation, the GPU linear-solver/preconditioner stack, and the time
integration. Companion to [`dg-gpu-fluid-simulation.md`](./dg-gpu-fluid-simulation.md)
§4 and [`mesh-and-adaptivity-strategy.md`](./mesh-and-adaptivity-strategy.md).

> **Confidence:** **[V]** = backed by the verified research pass (24/25 claims
> survived 3-0 adversarial verification; sources cited inline); **[E]** =
> engineering synthesis. *Pass run 2026-06-01. One claim was **refuted** — see
> §5.* **Caveat that frames everything below:** no single source demonstrates the
> full incompressible + viscoelastic + high-order-DG + multi-GPU + AMR + immersed
> stack end-to-end. This is an assembly of separately-validated components, and the
> coupled-system robustness at high Weissenberg number is essentially uncharted. **[V]**

---

## 0. TL;DR recommended spine

1. **Time integration:** **high-order stiffly-stable dual-splitting**
   (Karniadakis–Israeli–Orszag): implicit BDF for the stiff/implicit terms,
   explicit extrapolation for advection. Decouples velocity & pressure,
   **circumvents the LBB/inf-sup constraint → equal-order interpolation**, no
   per-step iterative coupling. This is exactly what the verified SRCR-DG
   *viscoelastic* DG solver uses. **[V]**
2. **The one hard solve becomes a scalar pressure-Poisson** (elliptic), which
   dominates cost. Solve it **matrix-free** with **CG + p-multigrid using
   Chebyshev-accelerated Schwarz/Jacobi smoothers** (nekRS-proven), plus a
   low-order-refined (LOR) coarse correction. **[V]**
3. **Viscoelastic constitutive:** evolve a **positivity-preserving reformulation** —
   **log-conformation** (Fattal–Kupferman) or **square-root-conformation (SRCR)** —
   to push past the high-Weissenberg breakdown. **[V]**
4. **Upgrade path (not v1):** if pressure-robustness / exact local conservation /
   divergence error limit accuracy (likely for particle forces & VE stress),
   upgrade the spatial discretization to **H(div)-conforming HDG** (exactly
   pointwise divergence-free). **[V]** — weigh against its complexity and the open
   AMR-non-conformity question (§6).

---

## 1. Formulation — the divergence constraint (Topic 1)

DG does not naturally enforce divergence-free velocity; how you handle it
determines stability, accuracy, and conservation.

### Options & verdict
- **Dual-splitting / fractional-step (recommended start).** Splitting in time
  decouples velocity and pressure and **removes the LBB requirement, allowing
  equal low-order interpolation for all variables with no iterative coupling per
  step** (SRCR-DG, *Engineering with Computers* 2022, doi:10.1007/s00366-022-01707-5).
  Built on the Karniadakis–Israeli–Orszag (1991) high-order stiffly-stable
  splitting. **Tradeoff:** you trade the LBB constraint for **splitting error,
  time-order reduction, and pressure-boundary-condition errors** (the consistent
  high-order pressure BC matters). **[V]**
- **HDG with H(div)-conforming (or relaxed-H(div) + reconstruction) — the
  accuracy upgrade.** Gives **exactly pointwise divergence-free, pressure-robust,
  momentum-conserving, energy-stable** velocity, and **static condensation reduces
  the globally-coupled DOFs** for the implicit solve (Lehrenfeld & Schöberl, CMAME
  2016, arXiv:1508.04245; Rhebergen & Wells, JSC 2018, arXiv:1704.07569;
  Lederer/Lehrenfeld/Schöberl, SINUM 2018 — relaxed H(div) recovers
  superconvergent-HDG DOF counts then a *cheap reconstruction operator* restores
  pressure-robustness + pointwise-div-free). "Pointwise div-free" is a discrete-space
  property, materially stronger than weakly div-free. **[V]**
- **Unified view.** A single Galerkin framework yields CG / H(div) / DG incompressible
  NS by varying the **viscous stress tensor and penalty terms**, with
  **pressure-robustness tied to discretization design** (symmetric-gradient /
  physical viscous stress — the Linke insight), validated on Taylor–Green,
  Kovasznay, cylinder (Chen et al., *JCP* 422, 2020, arXiv:2008.09485). Implication:
  build the viscous operator so pressure-robustness is a configuration, not a rewrite. **[V]**
- **Operator split that fits low-Re.** Explicit **Upwind-DG for the hyperbolic
  advection** subproblem + implicit **HDG for the unsteady Stokes** subproblem —
  "rather natural," since at low Re the advective CFL relaxes and the stiff
  Stokes/viscous part is what must be implicit (Lehrenfeld & Schöberl 2016). **[V]**

### Why not the others
- **Artificial compressibility:** adds an acoustic time-scale stiffness — wrong for
  low-Re incompressible. **[E]**
- **Fully-coupled saddle-point every step:** maximal accuracy/robustness but the
  hardest solve (block preconditioning, §2); reserve for when splitting errors bite. **[E]**

### Pressure pitfalls to instrument
Pressure null-space (all-Dirichlet-velocity → pressure defined up to a constant),
open/outflow BC artifacts, and the splitting-scheme pressure-BC error. **[V/E]**

---

## 2. GPU linear solvers & preconditioners (Topic 2 — the bottleneck)

**The pressure-Poisson / elliptic solve dominates incompressible high-order cost.** **[V]**

### What works
- **Matrix-free is mandatory at high order.** **Sum-factorization** drops operator
  application from `O(p^{2d})` to `O(p^{d+1})` and runs at a significant fraction of
  peak; assembling the matrix at high p is "prohibitively expensive" in both compute
  and memory (Bastian et al., *JCP* 379, 2019, arXiv:1805.11930; Kronbichler &
  Kormann, ACM TOMS 2019). On the **Titan V's 12 GB**, this strongly favors
  matrix-free over assembled AMG. **[V]**
- **p-multigrid + Chebyshev-accelerated Schwarz/Jacobi smoothers**, as a
  preconditioner inside a Krylov projector, is "robust and effective" for the
  high-order Poisson solve on GPU (Phillips, Kerkemeier & Fischer / **nekRS**,
  arXiv:2110.07663). **[V]**
- **Hybrid multigrid:** matrix-free block-smoothers in the high-order DG space +
  **low-order coarse-grid correction via AMG where only the low-order components are
  assembled** (Bastian et al. 2019). I.e. AMG lives only at the cheap coarse level. **[V]**
- **Low-order-refined (LOR) matrix-free preconditioning** lets the whole high-order
  incompressible/Stokes solve run matrix-free on GPU; the **saddle-point Stokes
  block-preconditioning is robust in mesh size, polynomial degree, time step, and
  viscosity** (Franco/Camier/Andrej/Pazner, *Comput. & Fluids* 2020,
  arXiv:1910.03032; Pazner & Kolev, SISC, arXiv:2103.11967 — uniform MINRES
  convergence independent of h and p). **[V]**
- **Parameter-robust HDG-Stokes preconditioner:** for the statically-condensed
  HDG time-dependent Stokes system, iteration counts/condition number are
  **uniformly bounded across viscosity, time step, and mesh size** (Henriquez, Lee &
  Rhebergen, arXiv:2604.07202, 2026). **[V]**
- **Saddle-point block preconditioners** (Schur-complement approximations — PCD,
  LSC, SIMPLE-type) are the route if you ever solve the coupled system directly. **[V/E]**

### What's risky / doesn't cleanly work
- **The winner is mesh-dependent.** Cheap Chebyshev p-MG smoothers win on regular /
  moderately-skewed meshes; **SEMFEM/AMG overtakes on highly-skewed meshes** and very
  large cases — nekRS ships a **poly-algorithmic autotuner** rather than one fixed
  choice (arXiv:2110.07663). gale should expect to need a small selector, not a
  single hard-coded preconditioner. **[V]**
- **AmgX** gives drop-in GPU AMG + Krylov and ~**2–5× single-GPU vs CPU** (Naumov et
  al., SISC 2015) — but that figure is vendor-reported, 2015 hardware, unspecified
  baseline (survived only 2-1), **and the claim that AMG setup+solve scale well
  across multiple nodes sustaining the single-GPU advantage was REFUTED (0-3).**
  → **Assembled-AMG multi-GPU scaling is a real risk for the two-Titan-V target;
  prefer matrix-free.** **[V]**
- **Conditioning degrades** with polynomial order p and on distorted/non-shape-regular
  meshes (mild iteration-count growth even for the "robust" LOR results), and the
  robustness proofs are for **Stokes / low-Re (≤ ~Re 100), NOT viscoelastic
  extra-stress coupling or high-Wi** — that regime is uncharted (§6). **[V]**

### Krylov choices
CG for the SPD pressure-Poisson; GMRES/FGMRES/BiCGStab for the nonsymmetric coupled
or viscoelastic systems; **flexible** preconditioning (FGMRES/flexible-CG) so the
multigrid V-cycle can vary per iteration. **[E, standard]**

---

## 3. Time integration for stiff viscoelastic + low Re (Topic 3)

### Stiffness sources & what must be implicit
Incompressibility constraint (algebraic/DAE), viscous term, and **polymer stress
relaxation (~1/Weissenberg)**. At low Re, **advection is the non-stiff part** →
explicit; the rest → implicit. **[V/E]**

### Recommended scheme
**Dual-splitting (implicit BDF + explicit extrapolation)** (SRCR-DG 2022): decouples
velocity/pressure, circumvents LBB, equal-order interpolation, no per-step iteration.
Advance momentum → pressure-Poisson → conformation/stress separately. **[V]**

### Constitutive stabilization (the HWNP)
The **high-Weissenberg-number problem** (numerical breakdown above a limiting Wi) is
addressed by reformulating the conformation tensor to **guarantee positive
definiteness**:
- **Log-conformation** (matrix log; Fattal–Kupferman; Hulsen/Fattal/Kupferman,
  *JNNFM* 2005) — "allows … high Weissenberg numbers which are impossible to solve
  with the typical three-field formulation"; in DEVSS/DG-FEM the improvement for the
  **Giesekus** model is "rather dramatic." **[V]**
- **Square-root-conformation (SRCR)** — the alternative used by the SRCR-DG solver. **[V]**
- **Both preserve positive-definiteness but do NOT fully solve the HWNP** —
  localized non-convergence near geometric stress singularities persists. **[V]**

Treat the linear stress-relaxation term with an integrating-factor/exponential or
implicit substep; advect the conformation tensor with upwind-DG (or
characteristic/semi-Lagrangian). **[E]**

### Coupling strategy
Monolithic (Newton) vs partitioned/segregated (the dual-split decoupling is
effectively partitioned). Start partitioned; escalate to Picard/Newton only if the
flow↔stress coupling fails to converge at high Wi. **[E]**

---

## 3b. Local / multirate time stepping (future capability)

A different time step in different cells — **local time stepping (LTS)** /
spatio-temporal adaptivity. Not v1, but the time-integration abstraction should
**reserve a per-cell time-level** so this is addable without a rewrite.

> **Confidence:** §3b.1 is now **[V]** — the dedicated research pass was re-run with
> working adversarial verification (10 findings, all 3-0; one refutation noted
> inline). Design rules below are **[E]**.

**The families (what exists):**
- **Berger–Oliger time subcycling** (block-structured AMR): refined patches take
  power-of-2 smaller substeps; **Berger–Colella refluxing** restores conservation
  at coarse↔fine *time* interfaces (the hard, bug-prone part).
- **ADER-DG clustered LTS** (SeisSol/ExaHyPE): the space-time predictor lets each
  element integrate neighbor fluxes over its local time window; elements binned into
  a few **power-of-2 time-step clusters**. *The most directly relevant reference.*
- **Tent pitching** (Gopalakrishnan–Schöberl–Wintersteiger): causal space-time
  tents respecting the local domain of dependence → intrinsic local dt.
- **Space-time DG/AMR**: refine the (d+1)-D space-time mesh (general, expensive).
- **Multirate integrators** (multirate RK / MIS / multirate IMEX): the
  method-of-lines framing of fast/slow coupling.

**Two design rules that already hold [E]:**
1. **CFL vs stiffness — different cures.** If a cell's limit is a *local CFL* (small
   cell, fast local speed) → **LTS** (cluster-based power-of-2 subcycling). If it's
   *stiffness* (stress relaxation ~1/Wi, HWNP near singularities) → **local
   implicitness / IMEX** (implicit only in stiff cells, à la locally-implicit DG for
   Maxwell) + the log/SRCR positivity reformulation. High-stress viscoelastic regions
   are mostly stiffness+under-resolution → lead with **spatial AMR + local implicit**,
   not smaller explicit dt.
2. **GPU form = cluster-based, level-batched.** Per-cell *asynchronous* LTS is
   GPU-hostile (divergent control flow, load imbalance). Advance all cells at a given
   power-of-2 level as one **batch** (SeisSol's design) — the only SIMD/GPU-viable form.

**The gale-specific caveat:** our incompressible spine is **implicit (dual-splitting
+ global pressure-Poisson)**. A global implicit solve couples the domain every step,
so LTS's payoff is **bounded to the explicit substeps — advection and
conformation-stress transport** — not the flow solve. LTS shines for *fully explicit*
hyperbolic systems (why SeisSol is its poster child); in gale it's a targeted
optimization of the explicit transport, and the natural companion to spatial AMR
(refined small cells would otherwise force a global tiny dt). **[E]**

### 3b.1 What the literature says (verified — [V])

The pass split the topic into **two distinct problems with two distinct cures** —
keep them separate:

**A. Local bottleneck = CFL (small/refined cells) → cluster-based power-of-2 LTS.**
- **Cluster-based (not per-element async) LTS is the production-proven, GPU/SIMD-viable
  form**: bin elements into power-of-2 time-step clusters and batch each — **2.3–4.1×**
  (SeisSol, petascale; Breuer/Heinecke/Bader, IPDPS 2016) to **6–10×** for 3D
  poroelastic with *stiff source terms* (Wolf et al., JCP 2022). **[V]**
- Per-element **asynchronous** LTS works (CPLTS for DG Maxwell, Angulo et al., JCP 2014)
  but is **GPU-hostile** — it "amplifies all task balancing and scheduling difficulties"
  (Charrier/Hazelwood/Weinzierl, SIAM 2020), which is *why* clustering exists. **[V]**
  *(That paper's specific enclave-tasking speedup figures were refuted 1-2 — cite the
  qualitative point, not the numbers.)*
- **2:1 temporal grading** ("maximum difference property"): neighbors are in the same
  cluster or differ by at most a factor-2 dt — the temporal analogue of 2:1 spatial
  AMR balance, and the coupling that makes time-refinement track h-refinement. **[V]**
- **ADER-DG enables it cleanly** via an element-local space-time predictor (solve a
  medium 1k–10k-unknown system per element, no neighbor coupling; only the flux step
  couples), and it can absorb stiff source terms locally. **[V]**
- **Tent-pitching / Mapped Tent Pitching** gives *intrinsic* LTS via causal space-time
  tents (causality constraint on tent height, **distinct from CFL**), mapped to a
  tensor-product reference for explicit (or locally-implicit) in-tent solves
  (Gopalakrishnan/Schöberl/Wintersteiger 2017). **Sharp caveat [V]:** naive RK in the
  degenerate tent map **silently loses order** (p=2 DG drops 3rd→1st order for RK2 *and*
  RK4); **Structure-Aware RK (SARK)** is required to recover it (PDE&A 2020). Research-grade.

**B. Local bottleneck = STIFFNESS (high-stress/relaxation, not small cells) →
locally-implicit / IMEX.** A *different* mechanism: treat **only the stiff cells
implicitly**, rest explicit, giving an unconditionally-stable small-cell update so the
global step is set by the coarse cells — proven **globally 2nd-order** (CNLF: explicit
Leap-Frog + implicit Crank-Nicolson) for Maxwell. **[V]** *Caveats:* the global scheme
still retains the **coarse-cell CFL** (not globally unconditionally stable); the
2nd-order proof is for **linear** Maxwell — **not** established for gale's *nonlinear*
viscoelastic conformation-stress transport; and naive IMEX blends can order-reduce. **[V]**
→ This is the right family for **high-Weissenberg stiff-stress regions**, likely paired
with the log/SRCR positivity reformulation (§3).

**Two hard limits for gale specifically:**
- **Amdahl ceiling on subcycling.** AMR time-subcycling gives **limited gains unless
  only a small fraction of the mesh is refined** (zone-counting argument, Dursi &
  Zingale 2003; the favorable exception is sparse-but-deep refinement). With a **global
  implicit pressure-Poisson solve every step**, classical explicit-LTS speedups are
  further Amdahl-bounded — the global solve can dominate. **[V]**
- **Incompressible precedent — and a warning.** A 4th-order projection AMR scheme *does*
  subcycle for incompressible NS and preserves 4th order, but **only via custom
  coarse–fine interface conditions + a composite synchronizing projection — explicitly
  NOT classical Berger–Colella refluxing** (Zhao & Zhang 2025, arXiv:2506.02663). So for
  a projection/dual-splitting solver, **refluxing is the wrong correctness mechanism.**
  Caveat: that result is brand-new, finite-volume, structured AMR, Dirichlet BCs, not yet
  independently reproduced. **[V]**

**Genuine gaps (no source covered these):** multirate method-of-lines coupling
(order conditions for a fast-explicit-transport / slow-implicit-pressure split), and
**LTS × immersed-boundary-method interaction** — both open for gale.

**Net (verified):** the cautious stance holds. **Lead with spatial AMR + the global
implicit solve; reach for locally-implicit/IMEX (not LTS) for high-Wi stiff regions;
add cluster-based power-of-2 LTS on the explicit advection/stress substeps only if
profiling shows that substep dominates and refinement is sparse.** Reference
implementation: **SeisSol** (clustered ADER-DG LTS, power-of-2, GPU); for incompressible
subcycling, Zhao & Zhang 2025 (and *not* refluxing).

## 4. Failure modes to instrument against (build these as diagnostics)

The user's explicit ask — what bites, and how to catch it early. Tie into the
per-cell diagnostics of `mesh-and-adaptivity-strategy.md` §3.

| Watchdog | Why | Trigger |
|---|---|---|
| **‖∇·u‖ per cell** (divergence error) | Dual-splitting is only *weakly* div-free; HDG is pointwise. Divergence error also *drives* time-order reduction. | grad-div stabilization; consider HDG upgrade |
| **Conformation min-eigenvalue / det** | HWNP onset; positivity loss in the *reconstructed* field | log/SRCR reformulation; refine; limit |
| **Solver iteration count** per step | Preconditioner robustness **degrades as Wi rises — uncharted** (§6); also p- and distortion-sensitive | switch preconditioner (poly-algorithm); investigate |
| **Observed temporal order (via MMS)** | **High-order (≥3) IMEX/RK suffer order reduction under time-dependent BCs** (Guesmi et al., IJNMF 2023, arXiv:2112.04167) | grad-div stab.; pick robust 3rd-order RK; consistent pressure BC |
| **Splitting / pressure-BC error** | Fractional-step pressure BC is a classic accuracy sink | high-order consistent pressure BC |

**[V]** for the divergence↔order-reduction link, the Wi-robustness gap, and the
order-reduction-under-time-dependent-BC result; **[E]** for the mitigations.

---

## 5. Refuted / contested (record so we don't rely on it)

- **REFUTED (0-3):** "AMG setup+solve scale well across multiple nodes, sustaining
  the single-GPU advantage." → Do **not** assume assembled-AMG scales across the two
  Titan Vs; design the multi-GPU pressure solve around **matrix-free** p-multigrid
  and validate scaling empirically. **[V]**
- **Weak (2-1):** the AmgX "2–5× vs CPU" figure (vendor, 2015, unspecified baseline).

---

## 6. Open questions (genuinely unsettled — gale will be partly first to answer)

1. **Preconditioner robustness with viscoelastic coupling as Wi rises** — the LOR /
   HDG-Stokes / block-Schur preconditioners are proven for Stokes/low-Re only;
   whether iteration counts stay bounded once elastic extra-stress is coupled and Wi
   grows is **essentially uncharted**. Instrument iteration counts from day one. **[V]**
2. **Multi-GPU scaling of matrix-free p-multigrid / Chebyshev-Schwarz across 2 Titan
   Vs** (PCIe, no NVLink), given assembled-AMG multi-node scaling was refuted and
   12 GB constrains assembled operators. **[V]**
3. **What sets the practical time step** for immersed capsules in a viscoelastic
   matrix — elastic CFL on the polymer stress (1/Wi), advective CFL (relaxed at low
   Re), or membrane-elasticity stiffness — and which exponential/IMEX treatment of
   the relaxation term best avoids order reduction while preserving positivity. **[V]**
4. **Is HDG/EDG static-condensation worth its complexity** for low-Re microfluidics
   on GPU vs equal-order dual-splitting (which already circumvents LBB), and **how
   does either interact with non-conforming AMR meshes** (p-multigrid robustness
   under non-conformity + distortion)? **[V]**

---

## 7. Build order (maps to the milestone plan)

1. **Scalar elliptic GPU solve first.** Before any flow: a matrix-free DG Poisson
   solver with **CG + p-multigrid (Chebyshev-Schwarz) + LOR coarse correction** on
   quads. *This is the single most reused, most performance-critical kernel set* —
   get it right, instrument iteration counts, test vs p and mesh distortion.
2. **Unsteady Stokes** (no advection): add the dual-split velocity solve + the
   pressure-Poisson from step 1. Validate Kovasznay / Taylor–Green; check ‖∇·u‖ and
   observed order via MMS.
3. **Incompressible Navier–Stokes:** add explicit upwind-DG advection (low-Re →
   explicit is fine). Lid-driven cavity, flow past cylinder (drag).
4. **Viscoelastic:** add the conformation transport in **log/SRCR** form, coupled by
   dual-splitting. Validate 4:1 contraction (Wi-ramp), flow past cylinder at rising
   Wi; watch conformation positivity and solver iterations.
5. **(Upgrade, if needed)** swap the spatial discretization to **H(div)-HDG** for
   pressure-robustness / exact divergence-free, once step-2/3 divergence diagnostics
   justify the complexity.

### Reference solvers to study
- **nekRS** (p-multigrid + Chebyshev-Schwarz, GPU, poly-algorithmic) — the
  preconditioner template. arXiv:2110.07663.
- **libParanumal** incompressible NS (semi-implicit, GPU multigrid-CG). arXiv:1801.00246. **[V earlier pass]**
- **rheoTool** (OpenFOAM-based viscoelastic: log-conformation, SRCR, many
  constitutive models) — the viscoelastic constitutive/stabilization reference.
  github.com/fppimenta/rheoTool.
- **NGSolve/Netgen** (Lehrenfeld–Schöberl HDG H(div)) — the HDG-upgrade reference.

---

*Synthesized from a fact-checked pass (5 angles, 23 sources, 90 claims → 24 verified
3-0, 1 refuted). Building blocks are high-confidence; the **integrated** stack is
research frontier — instrument the §4 watchdogs and treat §6 as live risks.*
