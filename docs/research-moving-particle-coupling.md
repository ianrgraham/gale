# Two-way coupling for freely-moving rigid particles in viscoelastic flow — research reference

**Status:** verified deep-research pass (2026-06-12). 5 search angles → 24 primary sources →
109 candidate claims → 25 adversarially verified (3-vote, need 2/3 to kill) → **25 confirmed,
0 killed** → 8 synthesized findings. Scope chosen with the user: **method open** (do not assume
the current penalization IBM must be kept), **full suspension machinery** in scope, **2D first**.

This is the implementation reference for extending gale's *fixed*-body immersed boundary
(`VolumePenalization` / `GpuPenalizationHook`) to *freely-moving* rigid particles and ultimately
dense particle-laden viscoelastic suspensions — the project's headline goal.

---

## TL;DR — the staged recommendation

1. **Move particles first with what we have.** Extend the existing volume-penalization (Brinkman)
   IBM with a rigid-body **Newton–Euler** update: recover hydrodynamic force/torque from the
   penalization term, integrate translation/rotation, move the mask. Smallest diff, reuses the
   validated fixed-body path. This is **explicit (weak) coupling** — correct and stable for
   *heavy* particles (solid/fluid density ratio above the critical value).
2. **Validate on the canonical physics.** Target **elasto-inertial single-particle lateral
   migration in Oldroyd-B channel flow** — well-established equilibrium-position-vs-elasticity-
   number phenomenology, and the literature benchmark itself was produced with a fictitious-domain
   coupling, so it is a true validation target (not just a CPU self-check).
3. **Plan the jump to strong coupling early.** Explicit coupling is *provably unstable below a
   critical density ratio* (added-mass effect) — so neutrally-buoyant / light particles (exactly
   the microfluidic-suspension regime) will need **implicit / strong coupling**. The
   architecturally elegant route for gale's dual-splitting projection is the **Added-Mass
   Partitioned (AMP)** idea: a generalized **Robin pressure boundary condition embedding the
   body's linear/angular acceleration**, inserted into the pressure-Poisson — no sub-iterations,
   stable even for zero-mass bodies, and it reuses the existing p-MG-PCG. **DLM/fictitious-domain**
   is the proven viscoelastic-validated alternative.
4. **Then go many-body.** Add short-range **repulsive/contact forces** (roughness-element model)
   to prevent interpenetration, layered with **subgrid lubrication corrections** for
   under-resolved gaps. These compose with whichever coupling is chosen.

A unifying review (Bhalla & Patankar) frames penalization, DLM/FD, direct-forcing, immersed-FEM,
and immersed-interface as **one constraint formulation** differing only in time-stepping and
discretization — so gale can start with penalization and *incrementally* swap in stronger
constraint enforcement without changing the conceptual core. That is what makes the staged path
low-risk.

---

## 1. The central stability fact: added-mass and the density-ratio limit

**Explicit / weak (partitioned, loosely-coupled) FSI for rigid bodies in incompressible flow is
unstable below a critical solid-to-fluid density ratio** due to the added-mass (and viscous
added-damping) effect. This is the dominant stability concern for moving particles, and it is
**geometry-sensitive**, not purely density-driven: a simple cylinder may be stable where a body
with a protrusion is not, at the *same* low density ratio (Lacis/Taira/Bagheri, JCP 2016). For
any loosely-coupled scheme there exists a density ratio below which it is unstable, driven by the
added-mass of the fictitious fluid inside the body (Causin–Gerbeau–Nobile / Förster–Wall–Ramm
canon). *[confidence: high; vote 11-2]*

Consequence for gale: an explicit penalization+Newton-Euler scheme is fine for a **heavy**
particle demo, but the microfluidic-suspension target (neutrally-buoyant particles, density ratio
≈ 1 and below) sits squarely in the regime where explicit coupling fails. Budget for strong
coupling.

## 2. Strong / implicit coupling removes the limit — at a cost

**Strong coupling is the established remedy and removes the density-ratio limit.** Implicitly-
coupled IB projection methods (Lacis et al., JCP 2016) are stable to density ratios as low as
**1e-4**. A related IB projection method (Lee & Lee, JCP 2022) keeps Navier–Stokes in a decoupled
fractional step while solving Newton–Euler *simultaneously* with the IB-force-density constraint,
preserving incompressibility and the no-slip kinematic constraint at the discrete level, stable to
density ratio of unity and below. Strongly-coupled IB methods solve the nonlinear fluid+structure
system simultaneously to enforce no-slip, giving favorable stability (Goza & Nair). *[high; 16-2]*

**The cost is the bottleneck.** Enforcing no-slip in strong coupling requires solving several
**flow-domain-sized** systems per step even though the constraint unknowns scale only with the
small number of interface points — a Schur-complement saddle-point structure (Nair & Goza, JCP
2021). *[high; 3-0]* **Direct implication for gale: a strong coupling must be expressed so it
reuses the existing matrix-free p-MG-PCG, not by forming/solving the saddle-point system
directly.**

## 3. AMP — the route that fits gale's dual-splitting projection

The **Added-Mass Partitioned (AMP)** algorithm (Banks, Henshaw et al., JCP 2017) achieves
strong-coupling stability **without sub-iterations**, even for **light / zero-mass** bodies, by
imposing a **generalized Robin interface condition on the fluid pressure** that embeds the body's
linear and angular acceleration (plus a boundary-integral added-mass term) — i.e. the added-mass
coupling is moved into the *pressure boundary condition*. *[high; 9-0]*

This is the most architecturally promising lead: gale already solves a pressure-Poisson in its
dual-splitting projection, and **the pressure BC is the natural insertion point**. An AMP-style
Robin term on the singular pure-Neumann pressure-Poisson would couple the body acceleration
without a separate saddle-point solve — reusing the p-MG-PCG. *Caveat: AMP was demonstrated on
finite-difference overset (body-fitted) grids, Newtonian, with the stability proof on a linearized
model problem — not DG-SIPG, not immersed-boundary, not viscoelastic.*

## 4. DLM / fictitious-domain — the proven viscoelastic-validated coupling

**DLM/FD is the validation-grade coupling for viscoelastic particle migration.** It imposes
rigid-body motion inside each particle via a distributed Lagrange multiplier (a body force per
unit volume, analogous to pressure enforcing incompressibility); the fluid–particle motion is
treated implicitly in a combined weak formulation in which the **mutual forces cancel**, so
explicit force/torque integration is **not required** for the Newton–Euler update
(Glowinski/Pan/Hesla/Joseph, 1999). The canonical Oldroyd-B elasto-inertial channel-migration
benchmark — equilibrium positions moving midline → diagonal → corner → centreline as elasticity
rises, governed strongly by elasticity number (Wi/Re) — was produced with a fictitious-domain
method (Yu/Wang/Lin/Hu, JFM 2019). *[high; 12-0]*

So DLM/FD is both the **proven physics-validation route** and the **strongest candidate for
gale's second-generation coupling** if AMP-on-SIPG proves too research-heavy.

## 5. Sharp-interface cut-cell DG — the native high-order alternative

Cut-cell DG (Krause & Kummer, Computers & Fluids 2017; BoSSS) represents bodies with
sharp-interface cut cells (not a diffuse mask), couples fluid/rigid-body by splitting (explicit),
and recovers force/torque by **hierarchical moment fitting (HMF)** quadrature over cut cells. It
is the most **architecturally native high-order** alternative to penalization. *[high; 8-1]*
But: Newtonian, explicit (added-mass-limited), and the **heaviest** path for a matrix-free GPU
codebase (small-cut-cell conditioning needs cell agglomeration). Not recommended as the first
step; revisit only if diffuse-interface accuracy near the particle proves limiting.

## 6. The unified picture (why the staged path is safe)

Bhalla & Patankar (arXiv:2402.15161, 2024) show **penalization/Brinkman, DLM/FD, direct-forcing,
immersed-FEM, and immersed-interface are one continuous constraint formulation**: treat the whole
domain as fluid, then constrain the fluid inside the solid to move rigidly via Lagrange
multipliers — a surface force per unit area Λ_s on the interface and a volume force per unit volume
λ_b in the solid, added to the fluid momentum equation. The methods differ **only** in
time-stepping and spatial discretization. Hydrodynamic force/torque for Newton–Euler can be
recovered directly from the constraint multipliers via a stress-jump relation Δ(σ)·n = F_s − Λ_s,
with three documented recovery options. *[high; 8-1; only weakness: arXiv preprint, peer-review
status unconfirmed]*

This is the theoretical license for the incremental plan: start with penalization, swap in
stronger constraint enforcement later, same framework throughout.

## 7. Dense / many-body suspensions

Dense suspensions require **short-range repulsive/contact forces** to prevent interpenetration,
modeling surface roughness elements and activated on close approach (Glowinski et al., 1999) —
the established complement to *any* immersed-boundary coupling, paired with **subgrid lubrication
corrections** for gaps too small to resolve on the mesh. *[high; 3-0]* These are an additive stage
on top of the single-particle coupling, not a coupling choice in themselves.

---

## Concrete build order for gale

- **M1 — Moving single particle, explicit, heavy.** Add rigid-body (Newton–Euler) state + force/
  torque recovery from the existing penalization term; move the mask each step. Validate vs CPU
  oracle (extend `ve-ibm-check` to a free body) on a *heavy* particle (above critical density
  ratio, so explicit is stable). Reuses the whole existing stack.
- **M2 — Migration physics demo.** Run elasto-inertial lateral migration in 2D Oldroyd-B channel
  flow; reproduce the equilibrium-position-vs-elasticity-number trend qualitatively, then
  quantitatively against a sourced 2D benchmark. First real scientific result.
- **M3 — Strong coupling for light/neutrally-buoyant particles.** Prototype the **AMP Robin
  pressure-BC** inserted into the dual-splitting pressure-Poisson (reusing p-MG-PCG). If the
  SIPG/viscoelastic transfer proves too uncertain, fall back to **DLM/FD** (proven for VE
  migration). This is the step that unlocks the actual suspension regime.
- **M4 — Many-body.** Short-range repulsion (roughness model) + lubrication corrections; multi-GPU
  partition for particle count (note gale's measured PCIe P2P crossover ≈128²/op — collisions are
  cheap relative to the elliptic solve until high areal fraction).

Each milestone is independently validatable and reuses the prior one. M1+M2 are achievable on the
current architecture with no new numerics research; M3 is the genuine research step.

---

## Caveats (read before committing to M3)

- **Architecture transfer is the central uncertainty.** ALL strong-coupling stability results
  (Lacis 1e-4; Lee/Lee at unity; AMP zero-mass) were shown in finite-difference / vorticity /
  overset / staggered-Cartesian-FFT solvers — **not** DG-SIPG matrix-free GPU. The discrete
  stability/consistency proofs do **not** automatically carry to gale's primitive-variable SIPG
  operators + p-MG-PCG. Transferability is an inference.
- **None of the coupling-stability sources are viscoelastic** — all Newtonian. The interaction of
  strong coupling with the IMEX log-conformation substep, the SPD/positivity limiters, and the
  conformation-tensor BC at a *moving* surface (esp. the near-wall stress boundary layer at
  moderate-to-high Wi) is essentially **uncovered**. The only VE-specific surviving result is the
  DLM/FD Oldroyd-B migration benchmark, which validates *physics*, not gale's numerics.
- The unified-framework source (Bhalla & Patankar) is an arXiv preprint.
- The migration benchmark (Yu et al.) is **3D** rectangular-channel; the **2D** analogue and exact
  elasticity-number transition values for a 2D demo need separate sourcing (see open questions).
- AMP stability is proven only on a *linearized model problem* in the cited Part I.

## Open questions (the M3 research agenda)

1. How does AMP-style (Robin pressure BC) or Lacis/Lee-style (simultaneous Newton-Euler + IB-force)
   strong coupling interact with gale's IMEX log-conformation substep and the SPD-cone limiter at
   a **moving** boundary? Does the limiter need a moving-mask-aware variant, and what conformation
   BC belongs at the moving rigid surface?
2. Can the strong-coupling constraint solve be expressed to **reuse gale's matrix-free p-MG-PCG**
   (avoiding the Schur large-solve bottleneck), e.g. via the AMP Robin term in the singular
   pure-Neumann pressure-Poisson of the dual-splitting projection?
3. What is the validated **2D** analogue of the Oldroyd-B lateral-migration benchmark (Re, Wi,
   viscosity ratio, blockage) for direct quantitative validation? Which other 2D VE benchmarks
   (sedimenting particle, drafting-kissing-tumbling, particle-in-shear) have published reference
   data?
4. For dense suspensions, the right composition of subgrid lubrication + short-range repulsion +
   contact with the chosen coupling on multi-GPU, and at what particle count / areal fraction the
   collision/lubrication stage (not the elliptic solve) becomes the scaling bottleneck.

---

## Primary sources

**Coupling methods / FSI:**
- Krause & Kummer, *An incompressible immersed boundary solver for moving body flows using a cut
  cell discontinuous Galerkin method*, Computers & Fluids 153:118 (2017). [cut-cell DG]
- Glowinski, Pan, Hesla, Joseph, *A distributed Lagrange multiplier/fictitious domain method for
  particulate flows*, Int. J. Multiphase Flow 25:755 (1999). [DLM/FD + repulsion]
- Bhalla & Patankar, *A unified constraint formulation of immersed body techniques*,
  arXiv:2402.15161 (2024). [unified framework + force/torque recovery]

**Added-mass / strong coupling:**
- Lācis, Taira, Bagheri, *A stable fluid–structure-interaction solver for low-density rigid bodies
  using the immersed boundary projection method*, JCP (2016), S0021999115007184. [implicit, ρ-ratio 1e-4]
- Banks, Henshaw et al., *Added-Mass Partitioned (AMP) algorithm for rigid bodies*, JCP 343:432
  (2017), arXiv:1611.05711. [Robin pressure BC, zero-mass, no sub-iterations]
- Nair & Goza, JCP (2021), arXiv:2103.06415. [strong-coupling cost / Schur bottleneck]
- Lee & Lee, *IB projection method, simultaneous Newton-Euler + IB-force constraint*, JCP (2022),
  S0021999122004296. [stable to ρ-ratio unity and below]
- arXiv:2001.01576, arXiv:1909.07423. [loosely-coupled instability canon]

**Viscoelastic migration / particulate benchmarks:**
- Yu, Wang, Lin, Hu, *Equilibrium positions of the elasto-inertial particle migration in
  rectangular channel flow of Oldroyd-B viscoelastic fluids*, JFM 868 (2019). [migration benchmark]
- D'Avino et al., PhysRevFluids 4:053301; Goyal & Derksen S0045793010003142; Fortin/Fortin JNNFM
  63:63 (1996); DLM/FD for viscoelastic particulate flows (Academia). [further VE particle refs]

**Conformation-tensor BCs / stress at surfaces:**
- S0021999124001372 (JCP 2024); S0377025705000455 (JNNFM); arXiv:2606.04005; Springer
  10.1007/978-3-319-93891-2_4.

**Dense suspensions / lubrication / GPU:**
- arXiv:2109.08300; arXiv:2307.13802; S0032591020305064; S003259102300387X.

*Verification: 25 claims verified by 3-vote adversarial check, 25 confirmed / 0 refuted.
Full machine record: workflow wf_1af7aa7f-72a.*
