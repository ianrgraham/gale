# Viscoelastic Solver — Implementation Notes

What is actually built in `src/dg/viscoelastic.rs`, the equations behind it, and how
each piece is validated. Companion to
[`implicit-solver-strategy.md`](./implicit-solver-strategy.md) (which planned the
formulation) — this doc records the *as-built* state.

All of it is CPU host code, test-driven against analytic solutions. The headline
application is **viscoelastic particle-laden microfluidic suspensions**, which needs
the high-Weissenberg-number regime — hence the log-conformation form is first-class,
not an afterthought.

---

## 1. Model: Oldroyd-B

The polymer is described by the **conformation tensor** `C` (symmetric
positive-definite, 2×2 in 2D, stored as `[Cxx, Cxy, Cyy]`). It obeys the
upper-convected Maxwell equation with `L = ∇u` (`Lᵢⱼ = ∂uᵢ/∂xⱼ`):

```
∂C/∂t + (u·∇)C = L·C + C·Lᵀ − (1/λ)(C − I)
```

and contributes the polymer stress `τ_p = (η_p/λ)(C − I)` to the momentum balance.
`λ` is the relaxation time, `η_p` the polymer viscosity, `η_s` the solvent
viscosity; the zero-shear viscosity is `η₀ = η_s + η_p`.

### 1a. Direct form — `OldroydB`
Advection by nodal collocation (element-local, reusing `grad_x`/`grad_y`); the
stretching `L·C + C·Lᵀ` and relaxation are algebraic/pointwise. Time-stepped with
SSP-RK3. Adequate at low–moderate Weissenberg number `Wi = λγ̇`.

**Validated against:**
- **Steady simple shear** `u = (γ̇y, 0)`: reaches the analytic
  `Cxx = 1 + 2Wi², Cxy = Wi, Cyy = 1` (tested at Wi = 2). Shear stress `τxy = η_p γ̇`.
- **Stress relaxation** (no flow): exponential decay `C(t) = I + (C₀−I)e^{−t/λ}`.

### 1b. Log-conformation form — `LogConfOldroydB` (high-Wi)
The direct form loses robustness at high `Wi` because numerical errors can push `C`
out of the SPD cone, triggering blow-up (the **high-Weissenberg-number problem**).
Fattal–Kupferman evolve `Ψ = log C` instead, so `C = exp(Ψ)` is **SPD by
construction**. Derived (2D) from the upper-convected equation:

```
∂Ψ/∂t + (u·∇)Ψ = ΩΨ − ΨΩ + 2B + (1/λ)(e^{−Ψ} − I)
```

where, with the eigendecomposition `C = R Λ Rᵀ` and `M = RᵀLR`:
- `B = R diag(m₁₁, m₂₂) Rᵀ` — pure stretching along eigendirections,
- `Ω = [[0, ω],[−ω, 0]]` (rotation is invariant in 2D), with
  `ω = (m₁₂λ₂ + m₂₁λ₁)/(λ₂ − λ₁)`.

**The isotropic-point subtlety.** When `λ₁ = λ₂` (e.g. starting from `C = I`) the
eigenframe is indeterminate and `ω` is singular; a naive `ω = 0` guard drops the
extensional driving and the state is *stuck at equilibrium* (`Ψ̇ = 0`) — it never
deforms. The fix: near-degenerate eigenvalues, **align the eigenframe with the
rate-of-strain tensor** `D = (L + Lᵀ)/2` instead of with `C`. Then `2B → L + Lᵀ`,
recovering the correct small-deformation limit `Ψ̇ ≈ 2D − (1/λ)Ψ`, and `ω → 0`. With
this, startup from rest works and the eigenvalues separate under shear.

**Validated against:**
- `exp(log C) = C` round-trip on an SPD field (to 1e−12).
- **Wi = 10** simple shear → analytic `Cxx = 201, Cxy = 10, Cyy = 1` (to <1%), with
  `C` SPD (`det > 0`) at *every* step — the regime where the direct form is fragile.

---

## 2. Coupling to incompressible momentum

The polymer feeds back through the stress divergence `∇·τ_p`, which enters the
momentum predictor of the existing dual-splitting solver (`Stokes`, solvent
viscosity `η_s`) as a body force:

```
ρ(∂u/∂t + (u·∇)u) = −∇p + η_s∇²u + ∇·τ_p + f,    ∇·u = 0
```

`∇·τ_p = (∂ₓτxx + ∂_yτxy, ∂ₓτxy + ∂_yτyy)` is computed by nodal collocation.
`Stokes::step_ns_forced` was added to accept a **nodal** body force (the closure-
based `step_ns` now delegates to it).

### `ViscoelasticFlow<M: ConstitutiveModel>`
Generic over the constitutive model via the `ConstitutiveModel` trait
(`equilibrium`, `advance`, `stress_div`, `recover_c`), implemented by both
`OldroydB` and `LogConfOldroydB`. One step is the standard decoupled split:
**momentum first** with the current stress, **then** the constitutive update with
the new velocity.

**Validated against — body-force-driven planar channel:** the polymer enters
momentum *only* through `∇·τ_p`, so the steady velocity must be the parabola with
the **total** viscosity `U(y) = (G/2η₀)y(1−y)`. Recovering `η₀` (not `η_s`) is the
proof the coupling is wired correctly.
- Direct form: velocity matches to **1.5e−3**; interior first normal-stress
  difference `N₁ = 2η_pλγ̇²` to **2.9e−3**.
- Log-conformation form (generic solver): recovers the `η₀` parabola; `C = exp(Ψ)`
  stays SPD throughout.

**Known limitation.** The channel is run on the all-Dirichlet box with the
fully-developed profile imposed on every boundary. The inlet/outlet–wall corners
carry a **stress singularity** the box can't represent (`N₁` is wrong there, fine in
the interior). The clean fix is **directional (channel) periodicity** in the mesh —
a small feature, not yet built.

---

## 3. Status & next steps

| Piece | Status |
|---|---|
| Oldroyd-B direct constitutive transport | ✅ validated (shear, relaxation) |
| Log-conformation (high-Wi, SPD-safe) | ✅ validated (Wi = 10) |
| `∇·τ_p` coupling into momentum | ✅ validated (channel → η₀, N₁) |
| Generic `ViscoelasticFlow` over both models | ✅ |
| Directional/channel periodicity (kill corner artifact) | ⬜ next |
| Immersed boundary method (elastic particles) | ⬜ headline feature |
| GPU port of the viscoelastic transport | ⬜ |

### Numerical caveats to revisit
- **Advection of `C`/`Ψ` is nodal-collocation** (element-local, no interface flux).
  Fine for the smooth/decoupled validations here; sharp stress layers at higher `Wi`
  will want a proper DG advection with an interface flux (the `Hyperbolic` operator
  already provides the machinery) and likely the SVV `ModalFilter` for
  stabilization.
- **Decoupled (first-order) splitting** between momentum and constitutive update;
  a higher-order or more strongly-coupled scheme may be needed for stiff
  high-`Wi` transients.
