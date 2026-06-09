# Research: Entropy-Stable & Structure-Preserving High-Order Methods — relevance to gale

**Status: COMPLETE — two passes (2026-06-08).** Pass 1 (`wf_2987f8a5-17e`): compressible
foundations + incompressible analogy (§1–§5), 25/25 verified 3-0. Pass 2, focused
follow-up (`wf_2cb6c679-8c5`): knapsack limiting + viscoelastic SPD (§6–§7), 20/25
verified. Together they answer all of RQ1–RQ5. **Headline:** gale's existing choices
(Chandrashekar EC, KEP incompressible flux, log-conformation) are all squarely inside
verified established theory; the **non-conforming AMR mortar needs the modified
(entropy-stable) procedure**; and **SPD-cone/bound-preserving limiting for viscoelastic
DGSEM is genuinely NOVEL territory** — all ingredients exist separately, the
combination has no prior art (§7 verdict, §9 direction + risk).

**Confidence tags:** ✅ verified this pass (3-0, source) · ⚠️ domain-knowledge lead
(verify before relying) · ❓ open / not covered by verified evidence.

Context: gale = GPU matrix-free nodal DGSEM (GL collocation, SBP-SAT), targeting
incompressible + viscoelastic (Oldroyd-B/Giesekus/FENE-P) flows, IBM + 2:1
non-conforming AMR. Already has Chandrashekar EC flux (compressible Euler) + a KEP
convection flux (incompressible) on CPU.

---

## 1. Verified foundations (compressible) — and they validate gale's design

- ✅ **The foundational mechanism is the Fisher–Carpenter equivalence**: diagonal-norm
  high-order SBP operators are equivalent to a *subcell finite-volume* formulation,
  which is what lets nodal DGSEM be rewritten as **flux differencing with two-point
  volume fluxes**. Everything else hangs off this. (Gassner/Winters/Kopriva 2016, JCP
  327:39-66, arXiv:1604.06618.)
- ✅ **The two-point flux choice *is* the scheme**: picking the subcell two-point flux
  systematically generates all common split forms (Ducros, Kennedy–Gruber) as one
  unified DG formulation. (Same.)
- ✅ **KEP ≠ entropy stability.** There's a systematic proof technique (Jameson's
  condition) for which split forms are kinetic-energy preserving — but **those KEP
  split forms are *not* entropy stable.** (Same.) Directly relevant to gale's KEP
  incompressible flux.
- ✅ **KEP and EC are structurally distinct families**: KEP depends only on the
  continuity + momentum discretization; **EC additionally depends on the energy
  equation and hence the equation of state.** Chandrashekar's notability is achieving
  *both at once*. (arXiv:2507.08115.)
- ✅ **gale's GL-DGSEM matches the canonical entropy-stable construction** (LGL nodes,
  derivative/mass pair = SBP operator, mimics the continuous entropy proof), including
  the constructive **3D curvilinear** compressible-NS formulation. **Aliasing** from
  under-resolved nonlinear advection is the identified root cause of classical-DGSEM
  blow-up, cured by the SBP differentiation matrix + flux-differencing of advective
  terms. (Friedrich et al. 2018; Winters/Kopriva/Gassner/Hindenlang 2021.)
- ✅ **Relaxation Runge–Kutta** makes the *fully-discrete* scheme entropy-stable via a
  per-step scalar relaxation parameter (one extra scalar solve), and **generalizes to
  enforce conservation/dissipation of *any convex functional*** — not just entropy.
  (Ranocha et al., SISC 2020, arXiv:1905.09129.) This is the temporal piece gale needs,
  and its convex-functional generality is a hook for non-entropy invariants (kinetic
  energy; viscoelastic free energy).

## 2. KEEP vs entropy-stable DGSEM (RQ1) — VERIFIED

- ✅ **Distinct but related families with shared machinery.** KEEP lineage =
  Tadmor → Chandrashekar → Coppola et al. → Kuya/Totani/Kawai, built on EC two-point
  fluxes. **Chandrashekar (2013)** gave the first scheme simultaneously KEP **and** EC
  (Ismail–Roe log-mean fluxes); **Ranocha** later gave a distinct KEP+EC formulation
  with more physically consistent pressure/KE treatment. (Jain & Moin 2022;
  arXiv:2507.08115.)
- ✅ **Kawai-lineage KEEP (Kuya/Totani/Kawai 2018)**: recasts mass + momentum
  convective terms into split convective forms; the energy-equation fluxes are then
  *determined* by requiring the discrete fluxes to satisfy the analytical relations
  among the governing equations. Extended to **unstructured FV** (Kuya et al. 2023).
- ✅ **Why KEEP adds entropy:** for *compressible* flow, discrete KE conservation alone
  is **not** sufficient for stability — additional entropy constraints are needed.

## 3. Incompressible bridge (RQ3) — VERIFIED, and decisive for gale

- ✅ **For incompressible Euler, kinetic energy *is* the mathematical entropy** — a
  bounded L2 measure of velocity that provides the nonlinear-stability bound. In
  compressible flow KE is no longer conserved and can't bound the solution, so
  thermodynamic entropy takes that role. (arXiv:2507.08115; Jain & Moin 2022.)
- **Verdict:** gale's **KEP incompressible-convection flux is the *correct and
  sufficient* structure-preserving choice** for the incompressible target — it sits
  squarely inside established theory, not a compromise. The compressible machinery
  (EC fluxes, thermodynamic entropy) is *not* needed for incompressible; KE is the
  whole story there.

## 4. AMR / non-conforming interfaces (part of RQ5) — VERIFIED, gale-critical

- ✅ **Standard mortar coupling does NOT guarantee entropy stability** for nonlinear
  problems on h/p non-conforming meshes and can cause instabilities; a **modified
  mortar procedure** recovers provable entropy stability. (Friedrich et al. 2018, JSC;
  corroborated Chan et al. 2021.) **gale does 2:1 non-conforming AMR — this is a real
  design constraint, now confirmed:** the entropy-stable interface needs the modified
  mortar, not the naïve one.

## 5. Limiting substrate (part of RQ2/RQ5) — partially verified

- ✅ **Provably entropy-stable subcell shock-capturing** convexly blends the
  high-order ES split-form DGSEM with a **low-order subcell FV scheme built on the SAME
  LGL nodes**, entropy-stable across the whole blend. (Hennemann/Rueda-Ramírez/
  Hindenlang/Gassner, JCP 2021, arXiv:2008.12044.) **This shared-node subcell
  construction is the structural prerequisite** for *all* subcell/convex/Zhang–Shu —
  and, by extension, knapsack — bound-preserving limiting in DGSEM.
- ❓ **Knapsack limiting itself is NOT covered by verified evidence** (see below).

---

## 6. Knapsack limiting (RQ2) — VERIFIED (focused follow-up, run `wf_2cb6c679-8c5`)

20/25 claims confirmed; the 5 killed were over-specific QP-algebra variants, not the
core results.

- ✅ **Origin = "limiting-as-optimization" (Lin & Chan, arXiv:2306.12663, JCP 2023).**
  Standard subcell limiting does *not* provably satisfy a semi-discrete cell entropy
  inequality; they fix this by formulating the limiting factors as the solution to a
  **continuous knapsack linear program** (`max Σxᵢ s.t. aᵀx ≤ b, 0 ≤ x ≤ U`), solved
  **exactly** by a deterministic **greedy algorithm** (proven optimal), cost
  `O(m log m)`, `m = N(N+1)`, **per element**.
- ✅ **Quadratic-knapsack variant (Christner & Chan, arXiv:2507.14488, 2025).** Blends a
  low-order entropy-stable positivity-preserving update with the high-order DGSEM update
  via per-interface coefficients `θ ∈ [0,1]`, an **FCT-type convex combination**
  `f_KL(θ) = (1−θ)f_L + θ f_H`. Decision variables are the **blending coefficients**
  (not antidiffusive flux magnitudes). Result: entropy stable, arbitrarily high-order in
  smooth regions, relative-positivity, and **hyperparameter-free**.
- ✅ **It reduces to per-element *scalar root-finding*** (quasi-Newton, ≤ L+1
  iterations, rarely > 4), **linear-time, element-local, embarrassingly parallel →
  strongly GPU-amenable**. Being *continuous* in the solution (vs the linear knapsack)
  gives higher temporal order at shocks (O(dt²) vs O(dt)) and far fewer adaptive steps
  (Table 2: linear-knapsack 44540 vs quadratic 2459 steps; LK did-not-finish where QK
  completes).
- ✅ **Independent monolithic line (Vilar, arXiv:2407.16815):** subcell DG/FV where each
  subcell face blends a 1st-order FV flux and a high-order flux convexly; the
  Gauss-Lobatto DGSEM variant uses a continuous knapsack. Treats entropy stability as an
  open question within the monolithic frame.
- ✅ **QP formulation (read directly from 2507.14488, "Entropy Stable Nodal DG Methods
  via Quadratic Knapsack Limiting"):** *minimize the total added diffusion* (i.e. stay
  as close as possible to the high-order solution) subject to (i) the discrete **entropy
  inequality**, (ii) **positivity** (density/pressure) at nodes, with θ_ij ∈ [0,1] and a
  symmetrization `θ_ij = max(θ_ij, θ_ji)` for conservation. **The reduction to 1-D
  root-finding works via a single scalar Lagrange multiplier μ**: the entropy inequality
  is *one scalar constraint per element*, so its multiplier μ is the only unknown — solve
  `Σ min(θ_ij(μ),1) = target` by quasi-Newton. **This single-scalar-constraint structure
  is exactly why the solve is cheap — and the crux of whether it extends to SPD (§9).**
- ⚠️ **Caveats:** "closed-form" overstates — it's iterative scalar root-finding (genuinely
  linear-time/local). **GPU speed is inferred, not benchmarked** — no source reports GPU
  numbers; the per-element-local property is proven, GPU-amenability is the corollary.

## 7. Viscoelastic SPD preservation (RQ4) — VERIFIED, with a sharp verdict

- ✅ **SPD loss is the core HWNP failure mode** (Yerasi/Picardo/Gupta/Vincenzi,
  arXiv:2312.09165): the chaotically advected conformation tensor develops huge
  gradients and loses positive-definiteness → instability. SPD preservation is
  **necessary but not sufficient** for large-scale accuracy.
- ✅ **Representation choice matters *physically*, not just for stability**: SSR vs
  **Cholesky-log** — only Cholesky-log preserves the large-scale forcing pattern;
  accuracy is attributed to the **log transform**. This validates gale's existing
  **log-conformation** choice.
- ✅ **SPD-preserving viscoelastic schemes exist — but only as continuous-FEM /
  representation-based, NOT limiters and NOT DG.** Lee–Xu / Lozinski–Owens–Phillips
  (2005–06): positivity-preserving FEM via a rate-type ↔ matrix-Riccati equivalence,
  SPD-preserving at any resolution (semi-Lagrangian FEM, 2nd-order). Two recent
  energy/entropy-stable confirmations of the *same pattern* (read directly):
  **Zhu/Pan/He 2025 (arXiv:2509.01278)** — energy-stable + positive-definiteness-
  preserving Oldroyd-B via log transform, but **first-order FEM**; and **Peng 2026
  (arXiv:2606.04005, "Entropy-Compatible Reconstruction for High-Weissenberg Flow")** —
  see next point. *All representation/FEM based; none is a DG limiter.*
- ✅ **Important nuance — log-conformation is SPD-safe but not automatically
  free-energy/entropy-compatible** (Peng 2026, arXiv:2606.04005): even a positive-definite
  reconstructed tensor can produce nonphysical behavior ("polymeric-work defects,"
  "entropy-budget errors"); the paper proposes a *corrected* logarithmic reconstruction
  satisfying discrete free-energy balance. **Directly relevant to gale: our existing
  log-conformation guarantees SPD but may carry these thermodynamic defects** — a
  separate axis from limiting (see §10, Direction 3).
- ✅ **THE KEY FIND — a general eigenvalue-cone-preserving *limiter* now exists
  (Amiri/Barrenechea/Pryer, arXiv:2601.04839, Jan 2026):** a variational inequality on
  the closed convex set of tensor fields whose DOF eigenvalues lie in `[ε,κ]`, realized
  by a **per-node projection that diagonalizes each tensor and truncates eigenvalues**
  into the bounds (ε=0 → positive-semidefinite). The paper **explicitly cites
  conformation tensors / FENE-P / extra-stress positive-definiteness as motivation** —
  *but it is continuous CIP-stabilized FEM, NOT DGSEM, NOT knapsack-based.*

### Verdict (synthesized, medium confidence — absence-of-prior-art argument)

**SPD-cone-preserving knapsack/convex limiting for viscoelastic high-order DG/DGSEM is
NOVEL / UNEXPLORED.** Every ingredient is established *in adjacent settings* —
knapsack limiting in DGSEM (scalar/system bounds + entropy), SPD-cone *projection*
limiting in FEM (tensors), log-conformation SPD representation — but **no prior art
combines them** into a GPU DGSEM viscoelastic SPD-cone/bound-preserving limiter.

**The sharp practical refinement** (from the caveats): gale *already* uses
log-conformation, which guarantees SPD **by construction** — so the truly novel,
useful gap is **not** re-enforcing SPD. It is **bound-preserving limiting**: the FENE-P
trace/eigenvalue bound `tr(C) < b`, and/or SPD-cone limiting **in the primitive `C`
variable at the exact loci where SPD is actually at risk — the 2:1 non-conforming AMR
interfaces and IBM interpolation/forcing**, where `C` is reconstructed in primitive
form and can leave the cone. That's where a limiter earns its keep in gale's pipeline.

---

## 8. gale verdict (final)

| Layer | Status |
|---|---|
| Compressible ES-DGSEM foundations | ✅ verified; already in gale, design validated |
| Incompressible (KE = entropy) | ✅ verified; gale's KEP flux is correct & sufficient |
| AMR non-conforming mortar | ✅ verified; needs **modified mortar** for ES |
| Relaxation-RK fully-discrete ES | ✅ verified; convex-functional generality is a hook |
| Subcell-limiting substrate | ✅ verified (shared-LGL-node low-order FV) |
| Knapsack limiting | ✅ verified; element-local, hyperparameter-free, GPU-amenable |
| SPD-preserving representations | ✅ verified; log-conformation (gale has it) is the good choice |
| **SPD-cone / bound-preserving limiting for viscoelastic DGSEM** | 🟡 **NOVEL** — all parts exist, combination unproven |

## 9. The pivotal technical question — now answerable

**Does the cheap quadratic-knapsack solve survive an SPD-cone constraint?** With the QP
formulation now read (§6), the mechanism is clear: the solve collapses to **1-D
root-finding because there is exactly *one scalar* constraint (the entropy inequality)
per element**, so a single Lagrange multiplier μ parameterizes everything.

- **Full SPD cone breaks this.** "All eigenvalues ≥ 0" is *not* one scalar inequality —
  it's a coupled multi-eigenvalue constraint. Enforcing the whole cone inside the blend
  needs a multidimensional (SDP-type) projection per node and **forfeits the cheap
  univariate structure**. High risk.
- **A single *scalar surrogate* keeps it cheap.** Constrain *one* scalar function of `C`
  per node and the 1-D Lagrange-multiplier reduction carries over:
  - **FENE-P trace bound `tr(C) < b`** — a *single linear* scalar constraint → an almost
    exact fit for the knapsack form.
  - **SPD via `λ_min(C) ≥ ε` or `det(C) ≥ ε`** — a *single nonlinear* scalar constraint;
    one multiplier, the root-find just becomes nonlinear in μ.

  Key realization: **don't enforce the full cone — enforce a scalar surrogate (trace /
  min-eigenvalue / determinant) and the cheap, element-local, hyperparameter-free,
  GPU-parallel structure is preserved.** That is the viable path.

---

## 10. Novel research directions (notes)

Ranked by promise; all 🟡 **novel** (no located prior art) unless noted.

1. **Scalar-surrogate knapsack limiting for viscoelastic bounds — the viable, cheap path.**
   Extend Christner–Chan quadratic-knapsack with a *single scalar* admissibility constraint
   per node: FENE-P `tr(C) < b` (linear → near-perfect fit), or `λ_min(C) ≥ ε` / `det(C) ≥ ε`
   for SPD. Keeps the 1-D root-find, element-local, GPU-parallel, hyperparameter-free.
   **Highest promise** — exploits gale's GPU + the verified knapsack machinery, sidesteps
   the SDP-projection cost.

2. **Locus-targeted limiting at AMR/IBM interfaces.** log-conformation preserves SPD in the
   bulk *by construction*, so apply the Direction-1 limiter **only where primitive `C` is
   reconstructed and can leave the admissible set — the 2:1 non-conforming mortar and IBM
   interpolation/forcing**. Cheaper than global limiting; targets the actual failure locus.
   Pairs with Direction 1.

3. **Entropy/free-energy-compatible log-conformation (orthogonal to limiting).** Per Peng
   2026, log-conformation guarantees SPD but may carry free-energy/entropy-budget defects.
   Investigate whether gale's log-conf *transport* is discretely free-energy-consistent, and
   whether an entropy-compatible reconstruction correction is warranted. A thermodynamic-
   fidelity axis, distinct from bounds.

4. **Full SPD-cone projection blend — the general, expensive fallback.** Amiri et al.'s
   per-node diagonalize-and-truncate eigenvalue projection as the "low-order admissible"
   state in an FCT/knapsack blend. More general than Direction 1 but reintroduces the
   multidimensional solve from §9. Pursue only if scalar surrogates prove insufficient.
   Unbenchmarked: per-node 2×2/3×3 symmetric eigendecomposition cost per RHS on GPU.

5. **Entropy-stable mortar + IBM compatibility (engineering-research, not in literature).**
   Establish whether the verified modified entropy-stable mortar (§4) and volume-penalization
   IBM forcing compose with the entropy-stable framework on GPU. IBM-entropy-stability is
   uncharted; no paper covers it.

**Cross-cutting verdict:** the verified ingredients (knapsack limiting in DGSEM, the
scalar-constraint 1-D solve, log-conformation, SPD-cone projection) have *never* been
assembled for a GPU DGSEM viscoelastic solver. Directions 1+2 are the concrete, low-risk,
high-leverage first target; 3 and 5 are independent correctness/fidelity tracks; 4 is the
fallback if scalar surrogates don't suffice.

---

## 11. Applicability across gale's simulation types (scope)

**These are two different tools, and they apply to different parts of the stack.** Easy to
over-generalize "limiting" to places it has no job — so be precise:

- **Tool A — entropy / kinetic-energy *stability*** (split-form two-point fluxes +
  relaxation-RK). About preventing nonlinear blow-up from aliasing, *not* bounds.
  **Broadly applicable** across the advective operators.
- **Tool B — *bound-preserving limiting*** (knapsack scalar-surrogate, Directions 1–4).
  **The governing rule: a limiter only has a job when the solution is required to live in a
  constrained admissible set** (positivity, a bound, a convex cone). No constraint → nothing
  to limit. Hence **selective.**

| gale simulation type | Admissible-set constraint? | Tool B (bound limiting) | Tool A (stability) |
|---|---|---|---|
| **Viscoelastic conformation `C`** (Oldroyd-B/log-conf; Giesekus/FENE-P later) | **Yes** — SPD cone; FENE-P adds `tr(C) < b` | ✅ **prime target** (the novel direction) | ✅ free-energy Lyapunov (less established) |
| **Compressible Euler** | Yes — `ρ>0`, `p>0` | ✅ if run with shocks/strong gradients | ✅ thermodynamic entropy (gale has Chandrashekar) |
| **Incompressible NS** (Stokes, dual-splitting) | **No** — velocity unconstrained; pressure is elliptic, not transported | ❌ **nothing to limit** | ✅ **kinetic energy = entropy** (gale has KEP) — the one that helps the flow solve |
| **Bounded scalar transport** — concentration `c≥0`, temperature (max principle), volume fraction `φ∈[0,1]` | Yes | ✅ applies — **but gale has no continuum scalar field today** | ✅ if advective |
| **IBM (volume penalization)** | No — forcing, not a transported bounded quantity; mask `χ∈[0,1]` precomputed | ❌ not a limiting target | ◻️ open: does penalization preserve the energy/entropy estimate? (Direction 5) |
| **Linear advection** | No (sign-unconstrained) unless a max principle is imposed | ◻️ optional | ✅ trivially |

### Reading of the table

- **Bound-preserving limiting is fundamentally a viscoelastic-(and-scalar-transport) tool
  for gale — not a flow-solver tool.** The incompressible velocity solve (most of gale's
  "flow" work) gets nothing from it: there is no bound on velocity to enforce.
- **Stability tools are the broadly-applicable ones** — they cover flow *and* viscoelastic
  advection. gale already carries the relevant fluxes (Chandrashekar for Euler, KEP for
  incompressible).
- **This is reassuring for prioritization:** the novel, effortful limiting work (Directions
  1–2) is precisely targeted at the viscoelastic part — the headline application — and does
  *not* need to be retrofitted across the whole solver. The flow side is already covered.

### The fork to watch — particle-laden modeling

gale's target is *particle-laden* suspensions. **If particles stay geometric** (rigid bodies
via IBM, no transported bounded field) → no new limiting target. **If gale ever introduces a
continuum suspension concentration / volume-fraction field** (`φ ∈ [0, φ_max]`) → that pulls
bound-preserving limiting into the suspension model as a **second first-class target**
(`φ`-positivity / max-packing bound), on top of the conformation tensor. Decide this
explicitly when the suspension model is designed — it determines whether Tool B has one
target or two.

---

> **Resolved loose ends (2026-06-08, direct reads):** QP formulation confirmed
> (2507.14488 — minimize diffusion s.t. one scalar entropy constraint + positivity,
> 1-D root-find on multiplier μ); arXiv:2509.01278 = energy-stable SPD-preserving
> Oldroyd-B but *first-order FEM*; arXiv:2606.04005 = entropy-compatible log
> reconstruction (representation, not DG limiter, + the free-energy-defect nuance §7) →
> both reinforce the NOVEL verdict. ResearchGate 347539614 (high-order DG viscoelastic
> solver) returned HTTP 403 — bound-preservation status unconfirmed, but every other
> source shows DG viscoelastic solvers do *not* do bound-preserving limiting.

> **Verified sources:** arXiv:2306.12663 (Lin–Chan), 2507.14488 (Christner–Chan),
> 2407.16815 (Vilar), 2508.21226, 2312.09165 (HWNP/SPD), 2601.04839 (Amiri et al.,
> eigenvalue-cone limiter), 2606.04005 (Peng, entropy-compatible recon), 2509.01278
> (Zhu/Pan/He), CMAME S004578250500157X (Lee–Xu/Riccati FEM). Prior pass:
> arXiv:1604.06618, 2507.08115, 1712.10234, 2008.12044, 1905.09129; Springer
> 978-3-030-60610-7_3; Jain & Moin 2022.
