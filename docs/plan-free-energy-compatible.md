# Plan: free-energy / entropy-compatible viscoelastic schemes

> Status: research-grounded plan, 2026-06-09. Verified deep-research pass (`wdvsy7ycz`, 24/25 confirmed;
> findings banked here). First pieces (diagnostic + relaxation-γ primitive) implemented; full enforcement
> gated on a dependency (below). Companion to `docs/research-entropy-stable-methods.md`.

## 0. Why

Prior research (the tensor-knapsack pass, research §12) surfaced Peng (arXiv:2606.04005, 2026): with
log-conformation, SPD positivity is **free by construction** but does **not** guarantee compatibility with
the discrete **free-energy (relative-entropy) balance** — a distinct, *necessary* structure-preservation
property. This pass pinned the functional, the enforcement mechanisms, and the honest caveats.

## 1. The free-energy functional (✅ verified)

Oldroyd-B elastic / relative-entropy (Helmholtz) free energy — the Boyaval–Lelièvre–Mangoubi functional:

```
F(C) = (η_p / 2λ) · [ tr C − ln det C − d ]          (d = 2 in 2D)
F'(C) = (η_p / 2λ) · (I − C⁻¹)                        (entropy variable)
```

`F` is convex, `F ≥ 0`, `F(I) = 0`; the relative entropy (Bregman divergence) is
`Φ(A|B) = Φ(A) − Φ(B) − (I − B⁻¹):(A−B) ≥ 0`. **The finiteness of `F` (via `−ln det C`) is the structural
property that precludes `det C → 0`** — i.e. free-energy boundedness *controls* SPD positivity, not the
reverse (Barrett–Boyaval: the log-`det` free-energy bound removes the time-step restriction needed to keep
`C` SPD). gale's `Ψ = log C` is exactly the discretization basis whose high-Wi stability BLM explain via
this free-energy analysis.

**Compatibility condition:** a scheme is free-energy-stable when it satisfies a discrete `dF/dt ≤ production`
inequality. This is **non-automatic** — it requires using *one* accepted tensor consistently across stress
work, stretching, entropy variables, and entropy quadrature, else an uncancelled `(1−β)/Wi (A_m−A_e):∇u`
coupling defect appears (Peng Prop 2.1).

## 2. Enforcement mechanisms (✅ both proven, complementary)

- **(A) Relaxation Runge–Kutta** (Ranocha et al. SISC 2020; Kang–Constantinescu JSC 2022 for IMEX-RK).
  One scalar `γ` per step: `u_{n+γ} = γ·u_{n+1} + (1−γ)·u_n` chosen so the time-adjusted state satisfies the
  discrete entropy/free-energy balance for **any convex functional**. `γ` = root of **one scalar nonlinear
  equation** per step (good guess `γ≈1`); closed-form only for *quadratic* entropy → gale's `ln det` `F`
  needs a general root-find. **Composes with IMEX/ARK** (Kang–Constantinescu carry both implicit-`f` and
  explicit-`g`). Cheap post-stage add-on, not a spatial-scheme rewrite.
- **(B) Peng's per-cell free-energy-correcting limiter.** A single per-cell `θ` along a *log path*
  `A(θ) = exp(Ψ̂ + θ(Ψ̃ − Ψ̂))` between the physical predictor (`θ=0`) and the raw high-order reconstruction
  (`θ=1`); accept raw if `J(1) ≤ J(0) + τ_K`, else bisect for the **largest** admissible `θ*` (least damping)
  under a per-cell entropy budget `τ_K = C_τ h_K^{2k+2}|K|`. `J(θ)` convex ⇒ bisection well-posed. Local,
  positive, spectrally controlled, asymptotically inactive under refinement. Layers on log-conf + the
  existing det/trace limiter (reconstruction-side, doesn't need an entropy-stable operator).

## 3. The honest caveats (what bounds the scope)

1. ⚠️ **"Free-energy is THE binding constraint" was REFUTED (1-2).** It is a *distinct necessary* property
   positivity doesn't provide — but whether it dominates over e.g. gradient resolution is Peng's editorial
   emphasis, not a theorem. So: worth *having*, not proven to be the single bottleneck.
2. ⚠️ **Relaxation-RK presupposes an entropy-stable SPATIAL operator** (each stage `⟨F',k_i⟩ ≤ 0`); `γ`
   cannot create entropy stability from a non-compatible spatial discretization. **gale does not have an
   entropy-stable two-point-flux / split-form conformation-transport operator today — this is the real
   missing piece** for *full* discrete free-energy stability (the largest open dependency).
3. ⚠️ **Jensen bias is real** (Peng Prop 3.1; Hameduddin–Zaki JFM 2019): a zero-mean log fluctuation
   preserves the Ψ-mean but raises the physical-`C` mean and `F` (bias ∝ ⟨η²⟩). Free-energy enforcement and
   physical-`C`-mean conservation **cannot both** be exactly held by a single log-path `θ` — *measure* it.
4. ⚠️ Foundational papers (BLM, Barrett–Boyaval) are low-order mixed FEM + backward Euler, not high-order
   DG. Peng is a ~2-week-old single-author preprint (MMS/diagnostic validation, no hard HWNP benchmark).
   Relaxation-RK lit treats only scalar/quadratic entropy — zero conformation-tensor / `ln det` treatment.

## 4. What's implemented (first prototype)

- **Free-energy diagnostic** (`src/dg/operators/viscoelastic.rs`): `free_energy_density(c, η_p, λ)` (per
  node) and `free_energy_total(mesh, c, η_p, λ)` (`∫F`, quadrature-weighted). Tracks the discrete `F` over a
  run — needed by *every* enforcement mechanism, and lets us **measure** whether free-energy actually
  misbehaves rather than assume it. Tests: `F(I)=0`, `F≥0`, `F` decays to 0 under pure relaxation, `F` grows
  under shear, and the **Jensen bias** (physical-`C`-mean `F` > log-mean `F`).
- **Relaxation-γ primitive** (`src/sim/integrate.rs`): `relaxation_gamma(eval, eta_old, prod)` — the
  Ranocha scalar root-find returning `γ` with `eval(γ) = eta_old + γ·prod` near `γ=1`, for any convex
  functional. Reusable; tested against the quadratic closed form and a `γ≈1` no-op case.

## 5. Roadmap (gated on the dependency)

1. **Measure first.** Run the diagnostic on a high-Wi shear/transport case and the IMEX path — is `F`
   spuriously increasing? How big is the Jensen bias? *Don't enforce what isn't broken* (caveat 1).
2. **If time-integration leakage is the issue:** wire `relaxation_gamma` into the conformation stepper
   (production estimate `e = dt Σ b_i ⟨F'(C_i), k_i⟩` from the RK stages) — enforces `F` non-increase at the
   time level. Cheap, composes with ARK.
3. **The real prize / dependency:** an **entropy-stable spatial conformation-transport operator**
   (entropy-conserving two-point flux / split-form for the `ln det` free energy) — without it, relaxation
   only fixes time leakage. This is a genuine research build.
4. **Reconstruction side:** Peng's per-cell log-path `θ`-limiter (mechanism B) as a `StageHook` — doesn't
   need the entropy-stable operator; complements the diagnostic.
5. FENE-P / Giesekus free-energy analogues (their nonlinear spring/destruction terms change the functional).

## 6. Verdict

Free-energy compatibility is a **real, necessary** structure property gale should at least *track*. The
**diagnostic is unambiguously worth building** (done). **Full enforcement is gated** on an entropy-stable
spatial operator gale lacks — so the honest path is *measure → relaxation-RK for time leakage → Peng limiter
→ (research) entropy-stable operator*, not a single drop-in limiter. Sources: Peng 2606.04005; BLM
0801.2248; Barrett–Boyaval 0907.4066; Ranocha 1905.09129; Kang–Constantinescu 2108.08908; Hameduddin–Zaki
1902.07790.
