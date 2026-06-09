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
2. **Knapsack-optimal θ (research §6, Christner–Chan).** The current `min`-over-constraints `θ` is the
   simple robust Zhang–Shu choice; the **quadratic-knapsack** limiter finds the least-dissipative `θ`
   satisfying all constraints jointly (a 1-D root-find on one Lagrange multiplier — cheap, GPU-amenable).
   Upgrade for less smearing.
3. **Log-conformation variant.** In `Ψ`-space, `C = exp(Ψ)` is SPD *by construction* — so the SPD limiter
   is moot there; only the FENE bound `tr exp(Ψ) ≤ b` can be violated (nonlinear in `Ψ`). A log-conf
   limiter would scale `Ψ` toward its mean to satisfy the scalar trace surrogate.
4. **GPU port.** The limiter is element-local (one block per element, a reduction for the mean + per-node
   `θ` + a min-reduction) — fits the established `gale-gpu` kernel pattern.
5. **AMR / mortar interaction.** Verify the limiter composes with 2:1 non-conforming interfaces (the
   research's modified entropy-stable mortar) and IBM forcing — the cell-mean admissibility assumption
   needs checking across hanging-node faces.

## 5. Scope note

This is a **viscoelastic-(and-bounded-scalar-transport) tool**, not a flow-solver tool — the incompressible
velocity has no bound to limit (research §11). It pairs with the IMEX implicit solve (relaxation bound) and
log-conformation (SPD-by-construction) to give end-to-end bound preservation for high-Wi viscoelastic DG.
