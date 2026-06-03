# Immersed Boundary Strategy for gale

Verified-research-backed decision doc for how gale should immerse boundaries —
rigid and **deformable elastic/viscoelastic** particles — in a high-order DG-SEM
incompressible/viscoelastic solver on multi-GPU, with AMR first-class.

> **Confidence:** **[V]** = survived a 3-vote adversarial verification pass
> (24/25 claims confirmed 3-0; sources cited inline). **[E]** = engineering
> synthesis for gale. *Research pass run 2026-06-02 (107 agents, 25 sources).*
> **Framing caveat [V]:** the high-order-DG + immersed-interface evidence is almost
> entirely on *linear/smooth* problems; deformable-particle methods (IFEM,
> front-tracking) are validated on FV/FEM, **not** DG-SEM; no source demonstrates
> the full deformable-viscoelastic-particle + high-order-DG + multi-GPU + AMR stack
> end-to-end. This is an assembly of separately-validated components.

---

## 0. TL;DR — recommended staged path

1. **Volume penalization (Brinkman) for rigid bodies first.** Simplest, maps onto
   the nodal body-force we *already* feed `Stokes::step_ns_forced`. Establishes the
   fixed-mesh immersed infrastructure. **Low-order (1st–2nd) at the no-slip
   surface** — accepted for the PoC.
2. **Front-tracking / direct-forcing IBM with a Lagrangian elastic membrane** for
   the headline deformable particles. This is the path the entire
   deformable-suspension literature actually uses (incl. in *viscoelastic* media).
3. **Grow into cut-cell / Nitsche DG** only where high-order accuracy at a
   (rigid/slowly-moving) immersed surface is worth the implementation cost.

The central, verified tension: **regularized-delta and volume-penalization IBM are
1st–2nd order at the interface**, which caps near-body accuracy regardless of the
DG order `p`; **cut-cell/Nitsche recover high order but are hard with deformation +
AMR**. The deformable-particle goal makes the low-order-interface family the
pragmatic first target — the field accepts the interface-order hit to get the
membrane physics and many-particle flexibility.

---

## 1. Taxonomy & how the boundary condition is imposed

All the fixed-mesh families **unify under one constrained formulation** — the choice
between them is a *discretization* choice, not different physics; rigid and
deformable bodies differ only by the interfacial stress jump (Bhalla & Patankar
2024, arXiv:2402.15161). **[V]** Body-conformal (remeshing) approaches are
**impractical for many moving particles** (Haeri & Shrimpton 2012), so gale must be
fixed-mesh. **[V]**

| Family | BC imposition | Moving? | Deformable? | Interface order |
|---|---|---|---|---|
| **Continuous-forcing / Peskin IBM** (regularized δ) | Spread Lagrangian force to fluid, interpolate velocity back via regularized delta | ✅ natural | ✅ natural (elastic markers) | 1st–2nd **[V]** |
| **Direct/discrete forcing** (Uhlmann, Mohd-Yusof; MLS/RKPM variants) | Force set so interpolated velocity = body velocity | ✅ | ✅ (with Lagrangian struct) | ~1st–2nd |
| **Volume penalization / Brinkman** | Porous-drag term `−(χ/η)(u−u_s)` drives `u→u_s` inside solid | ✅ | partial (filled bodies) | √η Dirichlet, η Neumann **[V]** |
| **DLM / fictitious domain** (Glowinski); one-field / fully-Eulerian | Lagrange multiplier enforces rigid/elastic constraint over body volume | ✅ | ✅ | low |
| **Cut-cell / embedded boundary** | Sharp geometry cut into mesh; flux on true surface | ✅ (hard) | ✗ (very hard) | **p+1 (L1)** **[V]** |
| **Nitsche / CutFEM / shifted boundary** | Weak BC on unfitted mesh + ghost penalty | ✅ (hard) | ✗ (hard) | high-order (FEM) **[V]** |

## 2. The high-order tension (item 2)

- **Volume penalization does *not* guarantee high order.** A claim that VP + nodal
  DGSEM retains `p+1` for `p∈[1,6]` was **refuted 0-3** (arXiv:2512.05733). The
  modeling error scales like `√η` (Dirichlet/no-slip) and `η` (Neumann); full DG
  order is recovered *only* where the penalized solution is smooth — which it is
  **not** at a sharp body surface. **[V]** VP + DG-SEM has nonetheless been properly
  analyzed (first modified-equation analysis, optimal *derivative* penalization,
  `hp`-adaptivity to damp Gibbs oscillations — Llorente/Ferrer 2023
  arXiv:2212.09560; 2025 hp-DGSEM), **but on linear problems only**. **[V]**
- **Cut-cell DG recovers `p+1` (L1)** but suffers the **small-cell CFL** explosion;
  **state redistribution** (Berger) restores background-grid time steps while
  keeping L2 stability (Giuliani 2022 arXiv:2102.01857; Taylor & Chan 2024
  arXiv:2404.06630). **[V]**
- **CutFEM/Nitsche** preserves high-order accuracy on unfitted meshes; **ghost
  penalty** controls the conditioning blow-up from arbitrarily small cuts (Burman et
  al., Acta Numerica). It is **FEM, not DG-SEM** — porting the ideas is non-trivial.
  **[V]**

**Implication for gale [E]:** pairing high-order `p` with a low-order interface
buys little *near bodies* — the interface error dominates there. High `p` still pays
off in the bulk (matrix flow, stress transport). So: use low-order IBM near
particles now; reserve cut-cell/Nitsche for later, for cases where surface accuracy
is the bottleneck.

## 3. Deformable particles (item 3)

- **IFEM (Zhang–Liu / immersed FEM)** couples a Lagrangian elastic solid to an
  Eulerian incompressible solver with **no remeshing**, and is validated for
  capsules, RBCs and elastic solids — **including in viscoelastic media**
  (Saadat & Shaqfeh, PRE 98:063316, 2018). **[V]** (Implemented there on FV, not
  DG.)
- **Front-tracking + Peskin regularized-delta + octree/AMR** is a proven combination
  for deformable capsules (linear-FEM membrane + paraboloid-fit curvature; Huet &
  Wachs, JCP 2023), at 1st–2nd-order interface accuracy. **[V]**
- The particle's elastic/viscoelastic response lives on the Lagrangian
  representation (membrane constitutive law → forces spread to the fluid; fluid
  velocity interpolated back to advect markers). Watch: **volume conservation**,
  **membrane locking**, **area-incompressibility**, and time-stepping **stability**
  of stiff membranes. **[V/E]**

## 4. DG-specific evidence (item 4)
Verified DG couplings exist for **volume penalization** (DG-SEM, linear) and
**cut-cell DG** (high-order, with state redistribution for the CFL/quadrature
issues). No verified source shows IBM + DG-SEM for *nonlinear* Navier–Stokes, let
alone viscoelastic — a genuine gap gale would be exploring. **[V]**

## 5. Viscoelastic-specific concerns (item 5) — *under-covered, follow up*
A dedicated immersed-boundary-smooth-extension method for **Oldroyd-B** exists
(Stein, flatironinstitute IBSE-Oldroyd-B) and several papers treat viscoelastic
suspensions, but the verification pass did **not** pin down conformation-tensor
boundary conditions on immersed surfaces or the high-Weissenberg pathologies right
at the body. **Open risk** — recommend a focused follow-up before Stage 2.

## 6. GPU / AMR (item 6) — *under-covered, follow up*
General signal: Lagrangian-marker spreading is scatter/gather-heavy (load-balance
when particles cluster); cut-cell + AMR couples but adds quadrature/bookkeeping;
penalization + AMR is the simplest (just a spatially-varying mask field). Specifics
for 2× Titan V were not nailed down — follow-up needed.

## 7. Recommendation for gale

**Stage 1 — Volume penalization, rigid bodies (immediate PoC). [E]**
Add a solid mask `χ(x)` and the body force `−(χ/η_b)(u−u_s)`; feed it through the
existing nodal `step_ns_forced`. Validate: **cylinder drag**, **Jeffery orbits** of
an ellipse. Cheap, GPU-trivial (a mask + forcing), reuses current infrastructure.
Accept low-order interface.

**Stage 2 — Front-tracking / direct-forcing IBM, deformable membranes (the
headline). [E]** Lagrangian capsule with an elastic (then viscoelastic) membrane
law; regularized-delta spread/interpolate. Validate: **single capsule in shear**
(Taylor deformation parameter `D` vs capillary number), then **two-particle
interactions**. Precede with the §5 viscoelastic-BC follow-up research.

**Stage 3 — Cut-cell or Nitsche DG (accuracy, optional/later). [E]** Only where
high-order surface accuracy matters (e.g. rigid fixed obstacles); use **state
redistribution** for the small-cell CFL.

**Key risks to watch:** (1) interface order caps near-body accuracy — don't expect
spectral convergence at particles; (2) stiff-membrane time-step limits; (3)
high-Weissenberg stress build-up at immersed surfaces (interacts with our
log-conformation transport); (4) GPU load-balancing when particles cluster;
(5) no prior art for IBM + DG-SEM + viscoelastic — we are partly off-map.

**Validation ladder:** rigid cylinder drag → Jeffery orbit → single elastic capsule
Taylor deformation → two-capsule interaction → (stretch) viscoelastic-matrix case.

---

# Follow-up: viscoelastic IB specifics + GPU/AMR

*Second verified-research pass, 2026-06-02 (106 agents, 24 sources, 23/25 claims
confirmed 3-0; 2 killed). Topic A (viscoelastic IB) is well-covered; Topic B
(GPU/multi-GPU/AMR) sources were fetched but did not reach the verified top-25 — it
**remains open** and should get its own pass when we GPU-port the IB layer.*

## A. Conformation-tensor BCs at a non-conforming surface

**The reassuring headline (a claim that "IB can't reliably do viscoelastic FSI" was
refuted 0-3):** standard regularized-delta IB *does* work for viscoelastic FSI. **[V]**

- **Standard practice imposes NO special interface treatment for `C`.** The
  conformation tensor lives at Eulerian cell centers, is advected by its normal
  transport equation **across the whole domain including the fictitious solid**, and
  rigidity is enforced *only* by the penalty/Lagrange-multiplier force — **no**
  no-flux/extrapolation BC, **no** masking, **no** SUPG/artificial diffusion.
  (Benchmark: arXiv:2309.00548, 2024.) **[V]**
- **The price:** pointwise **1st-order velocity**, **1st-/half-order (L1/L2) stress**,
  and **stress that does not converge pointwise at the surface** — yet **net forces
  converge at 1st order**. The cause is smearing the pressure / normal-velocity-
  gradient jumps across the regularized interface. **[V]** → For gale's PoC this is
  acceptable: integrated quantities (drag, particle force/torque) are reliable even
  though the near-surface stress field is only ~half-order.
- **High-order remedy = IBSE** (Immersed Boundary Smooth Extension, Stein–Guy–
  Thomases): smoothly `C^k`-extend the unknown field into the fictitious solid →
  **3rd-order velocity, 2nd-order stress** for Stokes/NS (4th Dirichlet / 3rd Neumann
  for scalar BVPs). **[V]** Caveat: demonstrated on **Fourier-spectral Cartesian**
  grids, **not DG-SEM** — porting the idea is open research, not a Stage-1 task.

## B. High-Wi pathologies near surfaces (motivates AMR + log-conf)

- **Stagnation points** (ubiquitous at particle near-contact and fore/aft of bodies)
  grow a **birefringent strand** — an *essential* stress singularity where the
  convective term degenerates with no diffusion to regularize it. Strand width
  shrinks as **1/De²** for UCM (FENE-P saturates at ~`2/L²`); finite-difference
  accuracy there degrades like `1/x^(2n)`. **[V]** → **strong, quantified case for
  AMR around stagnation/contact regions.**
- **Solid walls / curved surfaces / corners** develop sharp **elastic stress
  boundary layers** (for UCM exactly 3 dominant-balance structures) — Wi-driven, at
  order-one Re, independent of inertial layers; **corner singularities** at particle
  contacts. **[V]**
- **Root cause & remedy:** exponential stress gradients that polynomials can't
  represent → catastrophic instability; the field-standard fix is the
  **log-conformation representation (Fattal–Kupferman)** — *which gale already has* —
  optionally with **EVSS** (SPH reached Wi≈85 with log-conf+EVSS). **[V]** This is a
  concrete advantage: our existing `LogConfOldroydB` is exactly the recommended
  stabilizer.

## C. Published viscoelastic suspensions that were stable (recipes to copy)

- **Rigid:** OpenFOAM FV+IBM with **log-conformation**, fully-resolved spheres
  (Fernandes 2019); **Smoothed-Profile-Method** DNS of *many* rigid spheres in a
  **multi-mode Oldroyd-B** fluid, quantitatively matching Boger-fluid experiments
  (Matsuoka 2021) — but only validated at **φ≤0.1, Wi~1**. **[V]**
- **Deformable:** neo-Hookean particles in a **Giesekus** matrix up to **Ca≈0.3**;
  **IB-LBM** capsules in Oldroyd-B at high Wi stabilized by **artificial damping**
  (Ma et al., JCP 2020). **[V]**
- **Takeaway [E]:** the validated regimes are modest (dilute, Wi~1, moderate Ca);
  gale's dense/high-Wi/deformable target is genuinely at/beyond the state of the art,
  and *no* prior code is high-order DG-SEM on GPU — so stability recipes (log-conf,
  EVSS, artificial damping, smooth profile) transfer by **inference**, not proof.

## D. Topic B (GPU / multi-GPU / AMR) — still open

The pass did not verify GPU data-layout, load-balancing, marker migration, or AMR-
coupling claims (sources fetched: a Georgia Tech IBM-GPU kernels paper, multi-GPU
AMR refs — but none reached the verified set). **Open questions to resolve before
GPU-porting the IB layer:** scatter/gather (atomic-add vs marker-binning, SoA) for
spread/interpolate on sm_70; load balancing when particles cluster; marker halo
exchange / migration across the 2× Titan V split (NCCL/P2P); and which IB family
couples best with GPU AMR. **Recommend a third focused pass at GPU-port time** —
not blocking, since gale builds CPU-first.

## As-built status (Stage 1 — 2026-06-02)

`src/dg/immersed.rs` implements volume penalization for rigid bodies:
- `ImmersedSolid` trait + `Disk` (sharp or `tanh`-smoothed indicator, optional rigid velocity).
- `VolumePenalization` — nodal mask `χ` + implicit Brinkman relaxation
  `u ← (u + β u_s)/(1+β)`, `β = χ dt/η_b` (unconditionally stable).
- `force()` — penalization drag `∫(χ/η_b)(u−u_s)dV`, consistent with the implicit
  update.

Validated: exact relaxation algebra + η_b convergence; penalized disk in a driven
`channel_x` suppresses in-disk speed to ~2%·U_max; **drag** points downstream with
machine-zero lift by symmetry and is **exactly linear in the drive (ratio 2.000)**
in the Stokes regime. **Capstone:** a penalized rigid disk in a **log-conformation
Oldroyd-B** channel flow runs stably — flow suppressed inside, drag downstream, and
`C = exp(Ψ)` SPD *everywhere including at the immersed surface* (the §B risk did not
bite at moderate Wi). The full DG + incompressible + log-conf-viscoelastic +
immersed-rigid-particle stack is demonstrated end-to-end.

Freely-suspended **rigid-body (Jeffery)** dynamics are in via an L2 rigid-motion
projection (`VolumePenalization::project_rigid`): a disk in shear gives ω = −γ̇/2
exactly, an ellipse reproduces the 2D Jeffery ω(φ), and the integrated **tumbling
period matches `(π/γ̇)(r+1/r)` to <0.1%**.

**Stage 2 (deformable membranes) foundation — `src/dg/membrane.rs`:**
- DG-basis Eulerian↔Lagrangian coupling: `interpolate` (exact on polynomials) and
  `spread` (its mass-consistent adjoint — the consistent weak point load). Uses the
  DG nodal basis directly, not Peskin's uniform-grid delta.
- `Membrane`: closed marker ring with stretching-spring elasticity (force, strain
  energy, area, advection). Zero force at rest; purely internal (net force/torque 0).
- **Coupled front-tracking loop** (force → spread → Stokes solve → interpolate →
  advect) validated: a perturbed capsule in quiescent fluid relaxes (strain energy
  strictly decreases) with machine-zero centroid drift and no blow-up.

Lesson recorded: stretching-only springs restore local segment *lengths*, not global
shape — shape/area control (Skalak / neo-Hookean / bending) is the next refinement.

Not yet built: shape/area membrane elasticity + Taylor-deformation-in-shear
quantitative benchmark; high-order interface (IBSE/cut-cell); and all of Topic B
(GPU/AMR).

## Net effect on the plan
The staged path stands and is now **de-risked on physics**: Stage 1 (volume
penalization, rigid, CPU) is safe; integrated forces are trustworthy at low order;
our **log-conformation already is the high-Wi stabilizer** the literature prescribes;
**AMR around stagnation/contact is quantitatively justified**; high-order near-surface
stress (IBSE-on-DG) and all of Topic B are deferred research, not Stage-1 blockers.
