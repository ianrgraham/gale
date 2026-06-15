# Elastic / elasto-inertial turbulence with AMR — status & notes (2026-06-15)

Working toward a 2D AMR-resolved elastic-turbulence example (resolve the thin birefringent stress
strands with adaptive refinement). The automated literature pass was rate-limited; the background
below is from domain knowledge (verify against the cited papers when web access returns).

## Regimes
- **Elastic turbulence (ET):** inertialess (Re≪1), high Wi (≳ a few), needs **curved streamlines**
  (Pakdel–McKinley hoop-stress instability). Groisman & Steinberg 2000.
- **Elasto-inertial turbulence (EIT):** finite Re (~10²–10³), high Wi; occurs in **straight**
  channels. Key group **El = Wi/Re** (material+geometry, flow-rate independent). Dubief/Terrapon/Sid.
- Groups: Wi=λγ̇, Re=UL/ν, El=Wi/Re, β=η_s/η₀, FENE-P L². For our Kolmogorov box,
  El = λ·2πn·ν₀ (set by λ, ν₀, forcing wavenumber n — NOT by the forcing amplitude).

## Geometry choice
Constraint: AMR (`Mesh2d::cartesian_refined`) is **walled-box only** (no periodic/channel refined
builder), and **SBM is not integrated with AMR**. So the AMR-resolved demo must be a walled-box flow.
Chosen: **viscoelastic Kolmogorov flow** — body force `fx = F·sin(2πn y)` (zero at the no-slip
walls) → counter-flowing shear bands whose elastic instability sheds thin strands. The canonical 2D
ET model. Implemented in `traj-amr-kolmogorov` (env: KO_N/NU0/BETA/LAMBDA/F/NK/DT/BMAX/REFINE/
COARSEN/AMR, TRAJ_STEPS/EVERY).

## Stability findings (empirical, this solver)
- **Development time is ~λ.** The conformation stretches over the relaxation time; high El (truly
  elastic) ⇒ large λ ⇒ long runs (many thousands of steps). EIT (smaller λ, finite Re) develops
  faster and is the cheaper first target. The transport here is **convection-CFL-limited, not
  relaxation-limited** (λ large ⇒ relaxation not stiff), so dt can be raised well above the probe's
  5e-4 (use ~2e-3+).
- **Unbounded Oldroyd-B blows up at high Wi** (extensional growth Cxx~2Wi²). Added a FENE-P-like
  **trace cap** to the framework log-conf advance: `GpuViscoelasticDualSplitting::with_trace_bound(b)`
  (applies `limit_logconf_trace_bound` after each RK stage). With it, the AMR-OFF high-Wi run is
  healthy (Wi≈16, El≈4: tr C grows smoothly 2.0→3.6 over 600 steps, no blow-up).
- **OPEN ISSUE — AMR conformation remap breaks SPD.** With AMR ON at the same high Wi, the
  conformation collapses (tr C → 0) shortly after a remesh. Root cause: `remap_component_flat`
  prolongs/restricts the conformation tensor **componentwise**, which does NOT preserve positive-
  definiteness; high-Wi C sits near the SPD boundary, so interpolation pushes it non-SPD → `log C`
  collapses it. Velocity has no SPD constraint, which is why the Newtonian AMR demos were fine.
  Confirmed by isolation: AMR-off healthy, AMR-on collapses (same params).

## Fix path (next)
1. **SPD-preserving conformation remap** (the blocker). Options: (a) remap in log-space (Ψ = log C
   is unconstrained; C = exp(Ψ) is always SPD) — cleanest but the state field stores C; (b)
   project C back to SPD after each remap via the existing `limit_conformation_bounds` (det ≥ ε) —
   a defensive clamp, easiest to bolt on (apply post-remap, or in the integrator before `log C`).
2. **Then** longer runs + parameter tuning to reach developed ET/EIT (likely Re~1–10, Wi~10–50,
   El~1–10, FENE-P L²~few hundred), with AMR refining the strands.
3. **Possibly** a proper polymer **stress-diffusion** term `κ∇²Ψ` (Sc=ν/κ) if the limiter alone
   doesn't tame the steep stress fronts — the literature standard. AMR's payoff: resolve the strands
   so a smaller κ (higher Sc, more faithful) suffices vs a uniform mesh.

## What works now
`traj-amr-kolmogorov` runs a stable viscoelastic Kolmogorov flow with the trace-bound limiter; AMR
is wired (refines on conformation smoothness) and works for the velocity field but corrupts the
conformation at high Wi until the SPD-preserving remap (fix #1) lands. Without AMR it's a clean
moderate/high-Wi VE flow.
