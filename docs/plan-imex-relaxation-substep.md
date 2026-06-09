# Plan: IMEX/ARK relaxation substep for viscoelastic conformation transport

> Status: **COMPLETE for Oldroyd-B + Giesekus + FENE-P, CPU & GPU (2026-06-09).** Phases 1, 2, 3-GPU and
> all three constitutive models implemented + tested. Grounds the verified research in
> `docs/research-imex-viscoelastic.md` (esp. §10) against the real code. Scope is deliberately narrow:
> the **conformation substep only** — momentum/pressure are already semi-implicit (dual-splitting BDF1,
> `src/dg/operators/stokes.rs`). **187/187 CPU lib tests green; GPU validated on the Titan V** (all three
> models, ~1e-12 vs CPU oracle, all SPD, all `tr C < b`).
>
> **FENE-P** (extensibility `b`): added `LogConfOldroydB::extensibility` + `.with_extensibility(b)`
> (default ∞). The Peterlin relaxation `−(1/λ)[f·C−I]`, `f=(1−2/b)/(1−tr C/b)`, couples eigenvalues only
> through the trace. The implicit solve (`implicit_relax_solve_fenep`, CPU; the FENE-P branch in the GPU
> `implicit_relax` kernel) is a **bisection on `T∈(0,b)`** (monotone `G(T)=Σe^{ψ_i(T)}−T`) wrapping a
> per-eigenvalue Oldroyd-B-style inner Newton — the bracket keeps `tr C < b` ⇒ **bound-preserving by
> construction**. `relax_exact`/Strang is Oldroyd-B/Giesekus only (FENE-P's trace coupling has no
> per-eigenvalue closed form — use the ARK stepper). Tests: `fenep_implicit_solve_is_consistent_and_bounded`
> (stage residual <1e-9 + `tr C<b`), `fenep_imex_steady_shear_respects_trace_bound` (steady residual,
> SPD, bound). GPU `imex-ark-check` covers all three models (FENE-P 9.1e-12 vs CPU, `tr C<b`). The
> nested-loop (bisection+Newton) device kernel compiles via cuda-oxide and runs on sm_70.
>
> **Giesekus** (`α` mobility): added `LogConfOldroydB::mobility` + `.with_mobility(α)` (default 0 =
> Oldroyd-B). Because Giesekus differs from Oldroyd-B *only* in relaxation, and the IMEX split puts
> relaxation entirely in the implicit solve, the physics localizes to two CPU methods — `relax_exact`
> (now the Bernoulli flow `c(τ)=1+R/(1−αR)`) and `implicit_relax_solve` (Giesekus Newton
> `g=ψ+(γ/λ)(1−e^{−ψ})+(γα/λ)(e^ψ−1)²e^{−ψ}−b`, still strictly monotone) — plus the GPU `implicit_relax`
> kernel (same Newton, `alpha` arg). `psi_rhs`/`relax_source` stay Oldroyd-B (they cancel in the transport
> split, so transport is model-independent); the bare explicit `step_ssp_rk3` is therefore Oldroyd-B —
> **use the IMEX steppers for Giesekus** (which is the point). All α=0 paths are byte-identical (adding
> `0.0`/dividing by `1.0`). Tests: `giesekus_relax_exact_matches_ode_integration` (vs RK4),
> `giesekus_imex_satisfies_steady_shear_equation` (steady residual <1e-5, bounded extension), and GPU
> `imex-ark-check` now covers α=0.4 (~7.7e-12 vs CPU; Cxx 2.21 < Oldroyd-B 2.53 — shear-thinning).
>
> **Phase 3 (GPU)** — fully wired and validated on the Titan V (sm_70):
> - `gale-gpu/src/operators/logconf.rs`: the `implicit_relax` `#[kernel]` (per node: eigendecompose `B`
>   atan2-free, Newton-solve `ψ−(γ/λ)(e^{−ψ}−1)=μ` per eigenvalue, recompose in `B`'s eigenframe — purely
>   pointwise, one thread/node, reusing the `psi_rhs` eigendecomposition) + host `logconf_implicit_relax`.
>   Bin `imex-relax-check`: **max|gpu−cpu|/|Ψ| = 3.4e-16**, stage residual **4.4e-16**.
> - `gale-gpu/src/flow.rs`: `logconf_ark2_advance_gpu` — the full GPU ARK2/ARS(2,2,2) conformation step,
>   chaining explicit transport (`logconf_psi_rhs` + host upwind lift − `relax_source`) with the device
>   implicit solve per the §2.3 stage recursion. Bin `imex-ark-check`: after 20 steps at Wi=2,
>   **max|gpu−cpu|/|C| = 8.8e-12**, conformation SPD. Both re-exported from `lib.rs`; kernel export name
>   `implicit_relax` is crate-unique. (Build/run: `cargo oxide run --bin NAME` — `build` has no `--bin`;
>   run from the `gale-gpu/` dir or it resolves against the root `gale` package.)
>
> **Phase 1** (`viscoelastic.rs`, `LogConfOldroydB`): `relax_source`, `psi_transport_rhs`
> (= `psi_rhs − relax_source`, so `psi_rhs` is untouched/bit-identical), `relax_exact` (exact eigenvalue
> flow `c_i(τ)=1+(c_i−1)e^{−τ/λ}` via `sym_apply`), `step_transport_ssp_rk3`, `step_strang_imex`. Tests:
> `relax_exact_matches_analytic_for_any_dt` (1e-12 for dt/λ≤100), `imex_unlocks_large_timestep_where_explicit_fails`
> (stable+SPD at dt=10λ where explicit→NaN), `strang_imex_recovers_high_wi_steady_shear`.
>
> **Phase 2** — the production, no-splitting-error path:
> - `viscoelastic.rs`: `implicit_relax_solve` (per-node Newton on the eigenvalues, `ψ−(γ/λ)(e^{−ψ}−1)=b`,
>   monotone → globally convergent), `step_ark2_imex` (bespoke **ARS(2,2,2)**: L-stable, 2nd-order,
>   stiffly accurate), and `LogConfImex` (the `ImexSemi` adapter).
> - `sim/integrate.rs`: the generic seam — `ImexSemi` trait, `ArkTableau` (+`ars222()`), `ArkImex` driver
>   (tableau-general stage recursion; recovers `S_i=(Y_i−B_i)/γ` for free at implicit stages).
> - Tests: `ark2_imex_recovers_high_wi_steady_shear`, `ark2_imex_is_second_order_in_time` (observed order
>   ∈ (1.8, 2.3) — **no splitting error, confirmed**), `ark2_imex_stable_and_spd_at_large_dt`,
>   `ark_imex_generic_matches_bespoke_ark2` (generic driver == bespoke to 1e-12, Validation 6).
>
> **Next: Phase 3** — port `implicit_relax_solve` to a GPU device kernel (fuse into the existing per-node
> eigendecomposition); add Giesekus (per-eigenvalue quadratic) / FENE-P (scalar trace solve) to
> `implicit_relax_solve` per §7. **Design note:** `ArkImex` is its own integrator (not an `Integrator`
> impl) because it consumes an `ImexSemi`, not a `Semi` — the operand types differ (§4.3 realized).

## 1. Goal, scope, and the honest value statement

**Goal.** Add an implicit-explicit time advance for the conformation transport that treats the
**polymer relaxation** term implicitly and the **transport** (advection + rotation + stretching)
explicitly, removing the `dt ≲ 2.5λ` explicit-relaxation stability limit.

**Why it's cheap (the research payoff).** The relaxation source is local, pointwise, and isotropic in
`C` (commutes with `C`), so the implicit solve decouples into **scalar equations on the eigenvalues
gale already computes** (`sym_eig`, `viscoelastic.rs:361`). No global solve, no tensor Newton — see §3.

**Honest scope caveat (from research §10.2).** At gale's *current* settings (λ∈[0.5,1.0],
dt∈[0.001,0.02], N=3–4) relaxation is **60–2500× from binding** — the advective CFL of the transport
rules, so this buys ~no speedup *there*. The payoff is **regime-targeted**: it unlocks **small-λ /
fast-relaxation / concentrated-polymer (high elastic modulus G=η_p/λ)** runs. It is **not** the lever
for the high-Wi elastic-turbulence frontier (that's the HWNP → log-conf ✓ + bound-preserving limiting +
AMR). The value demonstration must therefore be a **small-λ** test (§6), not a current-regime one.

**Non-goals (this plan):** Giesekus/FENE-P (only Oldroyd-B is in code today — §7), stress-implicit
coupling of `∇·τ_p` into momentum, AMR/IBM interaction with the implicit stage, GPU port (sketched §8,
deferred).

## 2. Mathematical formulation

### 2.1 The additive split (log-conformation, the SPD-safe variable)
gale evolves `Ψ = log C` (`LogConfOldroydB`, `viscoelastic.rs:391`). Split its RHS
(`psi_rhs`, `viscoelastic.rs:448`) into the non-stiff transport `E` and the stiff relaxation `S`:

```
∂Ψ/∂t = E(Ψ, u)            + S(Ψ)
E(Ψ,u) = −(u·∇)Ψ + (ΩΨ−ΨΩ) + 2B      ← explicit  (advection + rotation + stretching)
S(Ψ)   = (1/λ)(e^{−Ψ} − I)            ← implicit  (relaxation)
```

`E` is exactly `psi_rhs` **minus** the `relax_*` lines (`viscoelastic.rs:511–515, 522–524`); `S` is
exactly those `relax_*` terms. The split is a one-function refactor (§4.2), no new physics.

### 2.2 The eigenvalue ODE is exactly solvable — two equivalent stiff treatments
In the shared eigenframe of `Ψ`/`C`, the relaxation acts independently on each eigenvalue `c_i` of `C`
(`c_i = e^{ψ_i}`):

```
dc_i/dt = −(c_i − 1)/λ          ⇒  EXACT flow:  c_i(τ) = 1 + (c_i(0) − 1) e^{−τ/λ}
```

This gives two clean, SPD-preserving options for the stiff part — both reuse the existing per-node
`sym_eig`:

- **(EXACT / exponential)** integrate the stiff substep *exactly* via the eigenvalue flow above
  (an ETD-style treatment — **zero stiff discretization error, unconditionally stable**). Closed form,
  no iteration.
- **(IMPLICIT / backward map)** the ARK implicit stage on `Ψ` solves `ψ*_i − γ S_i(ψ*_i) = b_i` with
  `S_i(ψ) = (1/λ)(e^{−ψ}−1)`, i.e. `g(ψ)=ψ − (γ/λ)(e^{−ψ}−1) − b_i = 0`. Since
  `g'(ψ)=1+(γ/λ)e^{−ψ} > 0` (strictly monotone), Newton from `ψ=b_i` converges globally in ~3 iters.

SPD is preserved in both: `Ψ` stays real-symmetric ⇒ `C = exp(Ψ)` SPD by construction (gale's existing
guarantee). The exact option additionally keeps each `c_i > 0` literally (`1 + (c_i−1)e^{−τ/λ} > 0`).

### 2.3 The ARK stage structure (production target)
For an `s`-stage additive RK with explicit pair `(A^E, b^E)` and L-stable, stiffly-accurate implicit
pair `(A^I, b^I)` sharing nodes `c` (Kennedy–Carpenter ARK; ARS/Giraldo IMEX for low order):

```
stage i:  Ψ_i − γ S(Ψ_i) = Bᵢ,   γ = dt·a^I_{ii},
          Bᵢ = Ψⁿ + dt Σ_{j<i} a^E_{ij} E(Ψ_j) + dt Σ_{j<i} a^I_{ij} S(Ψ_j)   (known)
update:   Ψⁿ⁺¹ = Ψⁿ + dt Σ_i b^E_i E(Ψ_i) + dt Σ_i b^I_i S(Ψ_i)
```

Each stage's only implicit work is the **local per-node solve** of §2.2 on `Bᵢ`. The diagonal
`a^I_{ii}=γ` is constant across nodes, so the eigenvalue solve is uniform (GPU-friendly).

## 3. The per-node implicit-relaxation kernel (the crux)

Input: an accumulated state `B` (in `Ψ`) and scalar `γ`. Output: `Ψ*` solving `Ψ* − γ S(Ψ*) = B`.

```
1. (μ1, μ2, c, s) = sym_eig(Bxx, Bxy, Byy)          // existing, viscoelastic.rs:361
2. for each eigenvalue b ∈ {μ1, μ2}:
     EXACT option:     ψ* = log( 1 + (e^b − 1) · e^{−γ/λ_eff} )   // see note on γ↔τ below
     IMPLICIT option:  solve ψ − (γ/λ)(e^{−ψ}−1) = b  by Newton (ψ0=b, ~3 iters)
3. Ψ* = R diag(ψ*_1, ψ*_2) Rᵀ                        // recompose, cf. sym_apply viscoelastic.rs:381
```

- **Cost:** one `sym_eig` (already paid by the transport stage) + 2 scalar evals. Marginal cost ≈ a
  handful of `exp`/`log` per node. Branch-light; the Newton has a fixed iteration count (no data-
  dependent loop) → negligible warp divergence on GPU.
- **Bound preservation:** automatic per §2.2. For the future FENE-P, the same kernel gains a single
  outer 1-D solve on the trace whose Peterlin barrier enforces `tr C < b` (research §10.1) — the kernel
  shape is forward-compatible.
- **γ↔τ note:** in the pure-split (§5 Phase 1) the exact stiff step uses `τ = dt`; inside an ARK stage
  the "exact" interpretation is per-stage and the **implicit backward map is the consistent choice** —
  use the EXACT form only for operator-splitting, the IMPLICIT Newton for true ARK.

## 4. Code architecture & touch-list

### 4.1 Reuse (no change)
`sym_eig`/`sym_apply` (`viscoelastic.rs:361,381`), `conformation`/`from_conformation` (`:435,:422`),
`upwind_advection_lift` (`:66`), the `Semi`/`Integrator`/`StageHook` seam (`sim/integrate.rs`) — whose
header already reserves "dual-splitting, IMEX … as further `Integrator` implementors" (`integrate.rs:11`).

### 4.2 `src/dg/operators/viscoelastic.rs`
- **Refactor** `psi_rhs` → extract `psi_transport_rhs(psi, ux, uy)` = current body **without** the
  `relax_*` contributions; keep `psi_rhs = psi_transport_rhs + relax` for back-compat / SSP-RK3.
- **Add** `relax_source(psi) -> [Vec<f64>;3]` (= the `S(Ψ)` terms, lines 511–515) for diagnostics/ARK.
- **Add** `implicit_relax_solve(b: &[Vec<f64>;3], gamma: f64) -> [Vec<f64>;3]` — the §3 per-node kernel
  (IMPLICIT/Newton variant).
- **Add** `relax_exact(psi: &[Vec<f64>;3], tau: f64) -> [Vec<f64>;3]` — the §2.2 EXACT eigenvalue flow.
- Mirror the direct-form split on `OldroydB` (`conformation_rhs`, `:144`) — there the C-space implicit
  relaxation is the trivially-closed-form affine map `C* = (B + (γ/λ)I)/(1+γ/λ)` (no eigendecomp at all),
  useful as an independent cross-check of the log-conf path.

### 4.3 `src/sim/integrate.rs`
- **Add trait** `ImexSemi { fn rhs_explicit(&self,…); fn implicit_solve(&self, state, gamma, t) -> …; }`
  (the additive analogue of `Semi`). Keep `Semi` untouched.
- **Add integrator** `ArkImex { tableau: ArkTableau, dt }` implementing `Integrator`, looping §2.3 over
  stages, calling `rhs_explicit` and `implicit_solve`. Ship one vetted low-order tableau first
  (ARS(2,2,2) or Giraldo ARK2) as a constant; Kennedy–Carpenter ARK3(2)4L[2]SA later.
- `ArkTableau`: small `struct` of `a_e, a_i, b_e, b_i, c` arrays (literal coefficients, unit-tested
  against published values).

### 4.4 Tests (`viscoelastic.rs` test module, cf. existing `:634+`)
New module per §6.

### 4.5 First-cut shortcut (recommended ordering)
Mirror the SSP-RK3 migration pattern (`integrate.rs:8` values bit-for-bit refactors): land the math as a
**bespoke method first** — `LogConfOldroydB::step_strang_imex(psi, ux, uy, dt)` (§5 Phase 1) — get the
validation green, *then* lift it into the generic `ImexSemi`/`ArkImex` seam (§4.3) and assert the generic
path reproduces the bespoke one. Lower risk, debuggable in isolation.

## 5. Scheme sequencing (what to build, in order)

- **Phase 1 — Strang split, exact relaxation** (smallest unlock demo). `transport(dt/2)` [explicit
  SSP-RK3 on `psi_transport_rhs`] → `relax_exact(dt)` [§2.2 exact, closed-form] → `transport(dt/2)`.
  2nd-order, **exact & unconditionally-stable stiff part**, SPD-safe, ~30 lines reusing existing pieces.
  Demonstrates `dt ≫ λ` stability immediately. Splitting error O(dt²) is the only approximation.
- **Phase 2 — ARK2 (production)**. `ArkImex` with an L-stable ESDIRK + ERK pair, per-node implicit
  Newton (§3). **No splitting error**, adaptive-dt-friendly (research §4: IMEX > ETD under AMR-driven dt
  variation). The real target.
- **Phase 3 — GPU + higher order** (§8): port the per-node solve to a device kernel; optional
  Kennedy–Carpenter ARK3 for 3rd order.

## 6. Validation plan (acceptance criteria)

1. **Pure-relaxation decay at large dt — THE unlock test.** Velocity = 0, `C(0)≠I`, **small λ** (e.g.
   λ=0.01). Analytic: `C(t) = I + (C₀−I)e^{−t/λ}`. Take `dt = 5λ … 50λ` (where SSP-RK3 *diverges*) and
   require the IMEX/exact path to (a) stay stable, (b) match the analytic decay to tolerance, (c) keep
   `C` SPD. This is the headline demonstration.
2. **Steady simple-shear unchanged.** The existing `Cxx=1+2Wi², Cxy=Wi, Cyy=1` check (`docs/
   viscoelastic-implementation.md`; tests at `viscoelastic.rs:634+`) must still pass at the same
   tolerance — IMEX must not perturb the converged state.
3. **Consistency vs SSP-RK3 at small dt.** For `dt ≪ λ` (both stable), IMEX and SSP-RK3 agree to the
   expected order — confirms the split/coefficients are correct.
4. **Temporal order of accuracy.** dt-refinement on a manufactured/coupled case → observed order
   matches the scheme (2 for Phase 1/ARK2).
5. **SPD preserved every step** at high Wi (reuse the Wi=10 log-conf case) — no regression.
6. **Generic == bespoke** (per §4.5): `ArkImex` path reproduces `step_strang_imex`/its analytic target.

## 7. Forward-compatibility: Giesekus / FENE-P (not in code yet)

The §3 kernel is shaped to extend (research §10.1): Giesekus adds a **quadratic per eigenvalue**
(closed-form positive root); FENE-P couples eigenvalues only through the **scalar trace** → one extra
1-D solve on `tr C` whose Peterlin barrier enforces `tr C < b`. Both slot into `implicit_relax_solve`
without changing the ARK driver. (Implementing the *models* themselves is separate prerequisite work —
`gale-gpu/.../probe_fp64_math.rs:6` already flags "FENE-P needs `powf`".)

## 8. GPU path (deferred sketch)

The per-node implicit solve is embarrassingly local → a device kernel that **fuses into the existing
per-node eigendecomposition** in the GPU log-conf advance (`gale-gpu/src/operators/oldroyd.rs`).
Transport stays as today (device collocation volume term + host `upwind_advection_lift`, the established
"solves on device, cheap assembly on host" split, `viscoelastic.rs:66`). The fixed-iteration Newton (or
closed-form exact map) keeps warps convergent. No global communication added.

## 9. Risks & open questions

- ❓ **Quantified payoff (research §10.3).** No measured advective-CFL-vs-relaxation crossover or
  speedup exists for a concrete gale case — Validation test 1 produces the first data point; a coupled
  small-λ channel run would quantify real wall-clock gain. Until then, "regime-targeted insurance,"
  not a promised speedup.
- ⚠️ **Splitting error (Phase 1).** Strang error grows when transport and relaxation strongly don't
  commute (high shear × fast relaxation). Phase 2 ARK removes it — don't ship Phase 1 as the endpoint.
- ⚠️ **Order reduction.** Stiff IMEX can drop below formal order at very large `dt/λ` (research caveat).
  Validation test 4 guards against silently shipping a degraded scheme.
- ⚠️ **The real high-Wi limiter may be elsewhere.** If the explicit polymer-stress→momentum coupling
  (dual-splitting step 1) is what actually caps `dt` at high Wi, implicit relaxation won't help — that
  needs a *stress-implicit* IMEX (a larger change, flagged research §10.3, out of this plan's scope).

## 10. Milestones

1. Refactor `psi_rhs` → `psi_transport_rhs` + `relax_source`; assert `psi_rhs` unchanged bit-for-bit.
2. `relax_exact` + `step_strang_imex`; Validation tests 1–3, 5 green (the unlock demo).
3. `implicit_relax_solve` (Newton) + `ImexSemi`/`ArkImex` + one ARK2 tableau; Validation 4, 6.
4. (Later) GPU kernel; (later) Giesekus/FENE-P kernels; (separate) assess stress-implicit coupling.
