# IMEX / structure-preserving time integration for stiff high-Wi viscoelastic DG-SEM

> Status: research note, 2026-06-08. Verified deep-research pass (23 sources, 25 claims
> adversarially verified, 19 confirmed, **6 refuted** — good skepticism discipline). Confidence
> tags: ✅ verified (≥2/3) · ⚠️ inference/analogy · ❓ open. Foundational area (many sources
> 2002–2009); none are GPU/DG-SEM/IBM/AMR-specific, so cross-domain transfer is the main caveat.

## 0. Why this matters for gale

Explicit Runge-Kutta hits a stability wall at high Weissenberg number Wi — exactly the
elastic-turbulence / strongly-elastic regime that is gale's scientific frontier. The fix is
implicit-explicit (IMEX) time integration: treat the stiff pieces implicitly, the cheap pieces
explicitly. The good news from this pass: **the stiffness sources decouple cleanly by cost**,
and the stiffest one (polymer relaxation) is also the cheapest to make implicit.

## 1. The three stiffness sources, separated by cost ✅

| Source | Nature | Stiffness | Implicit cost |
|---|---|---|---|
| **Polymer relaxation** `1/λ` (Oldroyd-B/Giesekus/FENE-P) | **Local, pointwise per quadrature node** (an ODE) | grows with Wi | **Cheap — element-local / analytic, NO global solve** |
| **Incompressible pressure / divergence constraint** | **Global elliptic** | always | Needs a global GPU linear solver (pressure-Poisson) |
| **DG-SEM CFL penalty** | per-element, ~quadratic in polynomial degree N (`~1/(2N+1)` canonical; quadratic is the conservative bound) | grows with N | Relieved by locally-implicit/globally-explicit ADER-DG |

**The key structural fact:** the *stiffest* term at high Wi — relaxation — is **local and pointwise**,
so making it implicit costs no global solve. Only the pressure constraint is genuinely global, and
gale already pays for that elliptic solve regardless of time integrator. **This is what makes IMEX
cheap for gale specifically.** ✅ (3-0, arXiv:2507.07304 for the N-CFL; Springer
10.1007/s00366-022-01707-5 for the pressure split; SINTEF/Aursand for the local-relaxation-ODE class)

## 2. Recommended scheme family: additive/IMEX Runge-Kutta ✅

**Kennedy–Carpenter ARK** is the canonical fit: an **L-stable, stiffly-accurate ESDIRK** integrates
the stiff part (relaxation, optionally viscous/pressure); a **traditional ERK** integrates the
nonstiff transport (convection + upper-convected derivative). The ARK2 stability function
**vanishes as the stiff eigenvalue z→∞**, so large `1/λ` is absorbed without hurting the nonstiff
convection. Reference schemes: ARK3(2)4L[2]SA / ARK5(4)8L[2]SA (Kennedy & Carpenter, Appl. Numer.
Math. 44, 2003). ✅ 3-0

Equivalent operator split for hyperbolic-with-stiff-relaxation: **Pareschi–Russo IMEX-SSP** — SSP
explicit for transport + **L-stable DIRK** for relaxation (J. Sci. Comput. 2005). Maps directly onto
"explicit upper-convected transport + implicit polymer relaxation." ✅ 3-0

**Bonus option — stiff-independent CFL:** SSP IMEX *multiderivative* RK (Gottlieb–Grant–Hu–Shu, SIAM
J. Numer. Anal. 2022) gives **a time-step restriction independent of the stiff term** (explicit part
sets CFL, implicit stiff part is unconditionally SSP) and is asymptotic-preserving. ⚠️ Validated on
Broadwell/BGK kinetic relaxation, not conformation tensors — transfer by analogy. ✅ 3-0 (with that caveat)

**What to treat how (recommendation):**
- **Implicit:** polymer relaxation (local/analytic per node) — and keep the existing
  BDF-implicit + extrapolation (dual-splitting) **pressure-Poisson projection** for incompressibility
  (decouples velocity/pressure, sidesteps the LBB/inf-sup condition).
- **Explicit (ERK/SSP):** convection + the upper-convected transport of C.

## 3. Does structure preservation survive the implicit treatment? ✅ (with a real gap)

| Property | Survives implicit? | Mechanism | Conf. |
|---|---|---|---|
| **SPD of conformation tensor C** | Yes | square-root (SRCR) **or** log-conformation representation — gale already uses log-conf | ✅ 3-0 |
| **Free-energy / entropy dissipation (Oldroyd-B)** | Yes, discretely | log-formulation's benchmark stability *explained* by a discrete free-energy analysis (Boyaval–Lelièvre–Mangoubi, M2AN 2009) — theoretical basis for keeping log-conf under a stiff integrator | ✅ 3-0 |
| **Entropy stability under IMEX** | Yes | **relaxation IMEX-RK** (Kang–Constantinescu, JSC 2022) extends explicit relaxation-RK to IMEX, entropy-preserving at discrete level; costs **one scalar nonlinear solve per step** (closed-form for quadratic entropy) | ✅ 3-0 |
| **FENE-P trace bound / positivity** | Plausibly | ETD/AP integrators can be "monotonically asymptotically stable" — solution unconditionally bounded by equilibrium, no overshoot, ∀Δt — for the *local monotonic-relaxation ODE class* gale's relaxation belongs to | ✅ 3-0 ⚠️ |

**The honest gap (❓):** **no source demonstrates SPD-cone + FENE-P bound + entropy *simultaneously*
preserved for a conformation-tensor DG-SEM solver.** Every structure result transfers by analogy:
Kang–Constantinescu is on ODEs/Burgers; AP/positivity is on BGK/kinetic; "entropy" there is a scalar
functional, not the full SPD/bound structure. This is the same novel-territory finding as the
structure-preserving research note — implicit-in-time *and* SPD/bound/entropy *together* for
viscoelastic DG-SEM is unproven ground. It also has to compose with gale's scalar-surrogate knapsack
limiter and entropy-stable AMR mortar (untested interaction).

## 4. ETD / exponential integrators — strong alternative, niche fit ✅

When the stiff relaxation operator is **linear/diagonal — which it becomes in log-conformation
space** — ETD solves the linear part *exactly* via the matrix exponential `e^{Lh}` (variation of
constants: `u_{n+1}=e^{Lh}u_n + e^{Lh}∫e^{-Lτ}N dτ`), removing the stiff time-step restriction
(Kassam–Trefethen 2005; SINTEF/Aursand). ✅ 3-0. ETD can be AP **and** positivity/bound-preserving
(Hu–Shu, SIAM MMS). ✅ 3-0

**The tradeoff that decides it for gale:** ETD beats IMEX on linear-dominated transients and
fixed-step stiff 1D PDEs, **but does not extend cheaply to adaptive/variable time-stepping — where
IMEX is the natural choice** (Kassam–Trefethen, explicit). ✅ 3-0. Since gale has **AMR-driven dt
variation and adaptive stepping**, this points to **IMEX as the primary path, ETD as a possible
fast-path for the relaxation substep** (it's exactly linear there in log-conf). ⚠️ The
"linear-in-log-conf" applicability is an inference, and ETD-beats-IMEX is a Cox–Matthews heuristic,
not a theorem.

## 5. GPU cost implication ✅ — the decisive practical point

- **Implicit relaxation = cheap and fully local.** It's a pointwise ODE per quadrature node →
  element-local, no global communication, possibly closed-form. ❓ Open: is the Giesekus/FENE-P
  *nonlinearity* closed-form per node in log-conf space, or does it force a **per-node Newton
  iteration** (with warp-divergence cost)? Oldroyd-B is likely analytic; Giesekus/FENE-P unclear.
- **Only the pressure-Poisson needs a global GPU linear solver** — which gale pays anyway.
- Net: IMEX adds essentially **local** work on GPU, not new global solves. This is the cleanest
  possible cost profile for going implicit.

## 6. Recommended first implementation target ⚠️ (synthesis, medium confidence)

**IMEX/ARK with implicit element-local (analytic-where-possible) relaxation + explicit transport,
reusing the existing dual-splitting pressure-Poisson projection.** Concretely:
1. Start with Oldroyd-B (likely analytic per-node implicit relaxation update in log-conf space).
2. ESDIRK/DIRK stiff stage on relaxation only; ERK on convection + upper-convected transport.
3. Keep the pressure projection as-is.
4. Measure realistic high-Wi speedup vs explicit RK (❓ no source quantified this for viscoelastic).

This is a **cross-source synthesis**, not a single validated result for a GPU DG-SEM viscoelastic
IBM+AMR solver — hence medium confidence.

## 7. What the adversarial pass *refuted* (don't believe these)

Six claims were killed 0-3 — useful guardrails against overclaiming:
- ❌ "Implicit BDF2 enables Courant numbers as large as 64 for viscoelastic flow" — NOT supported.
- ❌ "Log-conformation *guarantees* SPD and that guarantee is retained under BDF2" — NOT supported
  (log-conf helps SPD but the blanket guarantee-under-implicit claim failed).
- ❌ "Viscoelastic sims become unstable as Wi→critical and mesh refinement can't fix it" — NOT supported.
- ❌ "Log-conf is inherently fully-implicit / steady Newton solve is tractable" — NOT supported.
- ❌ Two exponential-RK overclaims (that ETD is *the* mechanism for positivity; that fully-discrete
  exp-RK satisfies entropy-decay under coupled spatial discretization) — NOT supported.

## 8. Open questions (verify before building)

1. ❓ Does structure-preserving IMEX preserve **SPD cone + FENE-P bound + entropy simultaneously**
   for a conformation-tensor DG-SEM system, and how does the implicit relaxation update interact with
   the scalar-surrogate knapsack limiter and entropy-stable AMR mortar?
2. ❓ For gale's actual Wi range and target N, **which stiffness dominates** the explicit dt
   (relaxation `1/λ` vs pressure elliptic vs `1/N²` CFL) — and thus the realistic IMEX speedup?
3. ❓ Is the per-node implicit relaxation update **closed-form** for Giesekus/FENE-P in log-conf, or
   does it need a per-node Newton solve (warp-divergence cost on GPU)?
4. ❓ How does locally-implicit/globally-explicit ADER-DG (relieves only the N-CFL) compose with the
   IMEX relaxation split and the pressure projection — and does either fight 2:1 AMR mortars or IBM forcing?

## 9. Key sources

- Kennedy & Carpenter, **"Additive Runge–Kutta schemes for convection–diffusion–reaction equations,"**
  Appl. Numer. Math. 44 (2003) — the canonical ARK reference (ESDIRK+ERK).
- Pareschi & Russo, **"Implicit–Explicit RK schemes for hyperbolic systems with stiff relaxation,"**
  J. Sci. Comput. 2005 (arXiv:1009.2757) — SSP-explicit + L-stable DIRK split.
- Gottlieb, Grant, Hu, Shu, **SSP IMEX multiderivative RK,** SIAM J. Numer. Anal. 2022 (arXiv:2102.11939)
  — stiff-independent CFL, asymptotic-preserving.
- Kang & Constantinescu, **entropy-preserving/-stable partitioned (relaxation) IMEX-RK,** JSC 2022
  (arXiv:2108.08908) — entropy survives IMEX via one scalar solve/step.
- Boyaval, Lelièvre, Mangoubi, **free-energy-dissipative log-Oldroyd-B,** ESAIM:M2AN 2009 (arXiv:0801.2248).
- Ma, Ouyang, Wang, **SRCR + dual-splitting pressure-Poisson at high Wi,** Eng. with Computers 2023
  (Springer 10.1007/s00366-022-01707-5).
- Kassam & Trefethen, **ETD for stiff systems,** SIAM J. Sci. Comput. 26 (2005) — ETD vs IMEX tradeoff.
- SINTEF/Aursand et al., **ETD for monotonic relaxation ODEs** — monotonic asymptotic stability.
- Hu & Shu, **AP + positivity-preserving exponential scheme,** SIAM MMS (10.1137/18M1226774).
- arXiv:2507.07304 — DG-SEM `~1/N²` CFL + locally-implicit/globally-explicit ADER-DG.

## 10. RESOLVED (2026-06-09): open questions §8.1 and §8.3 — with a framing correction

Resolved by codebase inspection + analysis. **Two findings, one of which corrects §1.**

### 10.1 Q (§8.3): is the implicit relaxation update closed-form per node? — YES, for all three models ✅ (analysis)
The IMEX implicit stage solves, per node, `C* − γ·R(C*) = B` (γ=Δt·a_ii, B = explicit accumulated
state, R = relaxation source). **The relaxation source of every standard model is an isotropic
polynomial in C, so it commutes with C and shares its eigenframe** — the matrix-implicit solve
decouples into independent **scalar equations on the eigenvalues `c_i`** (which gale's log-conf scheme
already computes each step). No full-tensor Newton.

| Model | Relaxation source R(C) | Per-eigenvalue implicit eqn | Solve |
|---|---|---|---|
| **Oldroyd-B** | `−(C−I)/λ` | `c_i*(1+γ/λ) = b_i + γ/λ` | **closed-form, affine** — `c_i* = (b_i+γ/λ)/(1+γ/λ)`; SPD-preserving (convex combo of SPD B, I) |
| **Giesekus** | `−(C−I)/λ − (α/λ)(C−I)²` | `c_i* + (γ/λ)(c_i*−1) + (γα/λ)(c_i*−1)² = b_i` | **closed-form quadratic** per eigenvalue (take positive root) |
| **FENE-P** | `−(f(trC)·C − I)/λ`, Peterlin `f=1/(1−trC/b)` | eigenvalues coupled **only through scalar trace** `T*=Σc_i*` | **one scalar 1-D solve on T\*** (closed-form quadratic in T* for standard Peterlin), then back-substitute `c_i*=(b_i+γ/λ)/(1+(γ/λ)f(T*))` |

Consequences:
- **Implicit relaxation is essentially free on GPU.** Marginal cost over the existing eigendecomposition
  is a scalar root per eigenvalue (Oldroyd-B/Giesekus) or a single 1-D solve per node (FENE-P).
- **Structure preservation is automatic, not bolted-on:** solving per-eigenvalue and keeping `c_i*>0`
  preserves SPD; **FENE-P's Peterlin barrier `f→∞ as T*→b` enforces the trace bound `tr C < b` by
  construction** — the implicit relaxation solve is itself bound-preserving (no separate limiter needed
  for the relaxation substep).
- ⚠️ **Warp divergence is minimal** — closed-form roots are branch-light; FENE-P's scalar solve is a
  fixed low-degree polynomial / few uniform Newton iters on one variable. (gale currently implements
  **only Oldroyd-B** — the simplest, fully-affine case; Giesekus/FENE-P are not yet in the code.)
- ⚠️ Eigenframe-commutes holds for the **relaxation substep in isolation** (operator-split / additive
  IMEX with relaxation in the implicit table). It is exact for these isotropic sources; the explicit
  transport/stretching (which *does* rotate the eigenframe) stays in the explicit table.

### 10.2 Q (§8.2) + CORRECTION to §1: which stiffness actually binds gale's explicit step? ✅ (codebase)
Inspection of gale: flow is **already semi-implicit** — pressure-Poisson (deflated CG) and the viscous
Helmholtz are **implicit** (dual-splitting BDF1, `src/dg/operators/stokes.rs`); only the **conformation
transport** is explicit (**SSP-RK3**). Tested regime: **Oldroyd-B**, **λ∈[0.5,1.0]**, **Wi≤10**,
**N=3–4**, **dt∈[0.001,0.02]**, β=0.5. No CFL is auto-computed; dt is set by hand.

Plugging in: SSP-RK3's relaxation stability limit is `dt ≲ 2.5·λ`. With λ≈0.5–1.0, that's `dt ≲ 1.25–2.5`
— but gale runs `dt≈0.001–0.02`, i.e. **60–2500× below the relaxation limit.** So **relaxation is NOT
the binding constraint** at gale's current settings; the explicit step is bound by the **advective CFL of
the conformation transport** (`dt ≲ C·h/(|u|·(2N+1))`, with the mild 1/(2N+1)≈0.11–0.14 order penalty at
N=3–4) and BDF1 splitting accuracy.

**Framing correction to §1/§0:** "relaxation `1/λ` stiff as Wi grows" is imprecise. The relaxation
Jacobian eigenvalue is `−1/λ`, so the source is stiff for **small λ (low Deborah / fast relaxation)** —
which is **low** Wi at fixed shear, not high. **High Wi does not make relaxation stiff**; the high-Wi
wall is the **High Weissenberg Number Problem (HWNP)** — loss of SPD positivity + unresolved steep
stress layers — which IMEX-of-relaxation does **not** fix. gale already handles HWNP with
log-conformation; the remaining high-Wi levers are **bound-preserving limiting** (see
[[structure-preserving-dg-research]]) + **resolution/AMR**, plus possibly **semi-implicit polymer-stress
coupling** if the explicit elastic body force (dual-splitting step 1) is what limits dt.

### 10.3 Revised recommendation (supersedes §6's blanket framing)
IMEX-relaxation is **cheap insurance, regime-targeted — not a high-Wi lever:**
- **Build it** (it's nearly free given §10.1, and Oldroyd-B is already in code): add the implicit local
  relaxation substep as an option. It pays off concretely when gale pushes to **small-λ / fast-relaxation
  / concentrated-polymer (high elastic modulus G=η_p/λ)** regimes, where `dt ≲ 2.5λ` binds.
- **Don't expect speedup at current settings** (λ~0.5–1): relaxation isn't binding there, so an
  IMEX scheme that only implicits relaxation buys little — the advective CFL still rules.
- **For the high-Wi elastic-turbulence frontier:** prioritize bound-preserving limiting + AMR resolution
  (HWNP), and investigate whether the **explicit polymer-stress→momentum coupling** is the true dt
  limiter (→ a *coupled/stress-implicit* IMEX, a bigger change than implicit relaxation).
- ❓ Still open (§8.2 quantitative part): a measured advective-CFL-vs-relaxation crossover and realistic
  IMEX speedup for a concrete gale case — needs a numerical experiment, not literature.
