# Plan: bound-preserving limiting for the conformation (HWNP frontier)

> Status: **first piece IMPLEMENTED + tested (2026-06-09)**; roadmap below. Grounds the verified research
> in `docs/research-entropy-stable-methods.md` (§6 knapsack limiting, §9 scalar-surrogate verdict) against
> the real code. This is the **transport-side** complement to the IMEX implicit relaxation solve
> (`docs/plan-imex-relaxation-substep.md`): that keeps *relaxation* in-bounds; this keeps the *high-order
> transport* in-bounds — the High-Weissenberg-Number Problem (HWNP) positivity failure the research
> flagged as the real high-Wi wall (distinct from the temporal-stiffness IMEX work).

## 1. The problem

High-order DG transport of the conformation can produce nodal values that violate the admissible set
(`C` not SPD; or `tr C ≥ b` for FENE-P) even when the cell mean is fine — Gibbs overshoot at sharp stress
gradients. The classic HWNP. A **bound-preserving limiter** scales the high-order part toward the
(admissible) cell mean just enough to restore the bound, preserving conservation and high-order accuracy
where the bound is slack.

## 2. The approach (research §9): scalar surrogates, not the full SPD cone

The full SPD-cone projection is an expensive SDP. The research verdict: enforce **scalar surrogates** that
are cheap (a per-element `θ ∈ [0,1]` from closed-form roots) — `det C ≥ ε` and `tr C ≥ 2√ε` (together ⇒
SPD for a symmetric 2×2), plus `tr C ≤ b` (FENE). Zhang–Shu / Christner–Chan style.

## 3. What's implemented (`src/dg/operators/viscoelastic.rs`)

- **`limit_conformation_bounds(mesh, c, eps, b_max)`** — the core. Per element: quadrature-weighted cell
  mean `C̄`; for each node, the largest `θ` keeping `C̄ + θ(C_k−C̄)` admissible — `det` is quadratic in `θ`
  (closed-form root via `theta_first_root`), `tr` bounds are linear; take `θ_elem = min` over nodes and
  constraints; apply. Conservative (mean unchanged exactly), high-order where slack (`θ=1`), and it
  skips elements whose mean is itself inadmissible.
- **`ConformationBoundLimiter { eps, b_max }`** — a `StageHook` (the `sim::integrate::StageHook` seam
  designed for "limiters, positivity/entropy enforcement") applying the limiter after each RK stage.
- Tests: `limiter_restores_spd_and_conserves_mean` (non-SPD node → all `det≥ε`, mean preserved to 1e-12),
  `limiter_enforces_fene_trace_bound`, `limiter_leaves_admissible_field_unchanged` (θ=1, high-order intact).

## 4. Roadmap

1. ✅ **Wired into the conformation time-stepping (2026-06-09).** `OldroydB::step_ssp_rk3_bounded`
   applies the limiter after each SSP-RK3 stage (Zhang–Shu stage limiting). End-to-end test
   `bounded_stepper_keeps_spd_where_plain_fails`: advecting a steep `Cxy` front (under-resolved high-order
   transport overshoots `|Cxy|>1` ⇒ `det<0`) loses SPD in `step_ssp_rk3` but stays `det≥ε` in the bounded
   stepper. (Still TODO: the same for the GPU direct-form advance, and the log-conf path.)
2. **Knapsack-optimal θ (research §6, Christner–Chan).** ✅ **Primitive + scalar application done
   (2026-06-09).** `knapsack_theta(weights, devs, caps)` solves the quadratic knapsack
   `min Σ½w(1−θ)² s.t. Σ w·δ·θ = 0, 0 ≤ θ ≤ cap` via the single-multiplier `θ_i(μ)=clamp(1−μδ_i,0,cap_i)`
   + a monotone 1-D bisection on `μ` — never more dissipative than uniform `min θ`, strictly less when
   non-capped deviations vary (tested). `limit_scalar_bounds(mesh, u, lo, hi)` applies it to a **scalar**
   field (conservative, less smearing) — directly usable for a bounded transported scalar (concentration
   `φ`). **Caveat / open frontier:** single-multiplier is exact only for *one* conserved scalar; the
   conformation *tensor* has 3 conservation constraints (3 multipliers) — the genuinely-hard multi-constraint
   case the research §9 flags as open. So `limit_conformation_bounds` stays on uniform `θ` (correct, fully
   tensor-conservative); the knapsack primitive is the building block for the eventual subcell/multi-constraint
   tensor limiter.
3. ✅ **Log-conformation variant done (2026-06-09).** `limit_logconf_trace_bound(mesh, psi, b_max)` +
   `LogConfTraceLimiter` StageHook. In `Ψ`-space `C = exp(Ψ)` is SPD by construction (SPD limiter moot);
   only `tr exp(Ψ) ≤ b` can be violated. Per element it scales `Ψ` toward its mean by the largest common
   `θ` keeping every node's `tr exp(Ψ) ≤ b` — and since `tr exp(·)` is **convex** (blend affine in `θ`),
   `g(θ)=tr exp(Ψ̄+θΔ)` is convex with `g(0)≤b<g(1)`, so the crossing is a robust **bisection**. Conserves
   the **mean of `Ψ`** (not of `C` — the standard log-conf trade-off, documented); no-op where slack /
   `b=∞`. Tests: `logconf_limiter_enforces_trace_bound_and_keeps_spd` (tr C ≤ b, SPD, Ψ-mean conserved),
   `logconf_limiter_leaves_admissible_unchanged`. Note: in the FENE-P *IMEX* path the implicit solve already
   bounds the step's *final* `tr C`; this limiter covers the *transport/explicit* overshoot (and non-IMEX paths).
4. ✅ **GPU port (2026-06-09) — the log-conf trace limiter.** `gale-gpu` kernel `limit_logconf_trace`
   (one block/element): each thread reduces the shared arrays for the quadrature-weighted cell mean
   (redundant per-thread sum — no power-of-2 dependence, `nn≤81`), computes its node's `θ` by the convex
   bisection, then a per-thread min-reduction gives the element `θ`, and applies. Host wrapper
   `gale_gpu::logconf_limit_trace`; bin `limiter-check`: **max|gpu−cpu|/|Ψ| = 2.5e-17**, `tr C ≤ b` on the
   Titan V (sm_70). (Architecture note: the limiter is cheap element-local work, so in the *host-orchestrated*
   GPU driver — where the field is already host-side between kernel stages — calling the **CPU** limiter adds
   no transfer and is equally valid; the device kernel is for an eventual GPU-resident stepper. The
   direct-form `limit_conformation_bounds` and the scalar knapsack are not yet ported — same pattern.)
5. **AMR / mortar interaction.** Verify the limiter composes with 2:1 non-conforming interfaces (the
   research's modified entropy-stable mortar) and IBM forcing — the cell-mean admissibility assumption
   needs checking across hanging-node faces.

## 5. Scope note

This is a **viscoelastic-(and-bounded-scalar-transport) tool**, not a flow-solver tool — the incompressible
velocity has no bound to limit (research §11). It pairs with the IMEX implicit solve (relaxation bound) and
log-conformation (SPD-by-construction) to give end-to-end bound preservation for high-Wi viscoelastic DG.
