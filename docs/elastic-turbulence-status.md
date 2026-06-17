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
- **AMR + viscoelastic flow works at high Wi — no SPD issue.** The LogConf model stores **Ψ = log C**
  (equilibrium Ψ = 0), which is SPD-by-construction (C = exp Ψ), so the componentwise AMR remap of Ψ
  is fine. With the correct equilibrium IC, AMR-on at Wi≈16/El≈4 is healthy: tr C grows smoothly
  2 → 10 → 66 → 236 → 500 (the trace cap acting as FENE-P L²), no collapse, no blow-up, and thin
  birefringent **strands form** (visible in tr C). [Earlier I reported a "tr C → 0 collapse / SPD-loss
  via remap" — that was WRONG: an artifact of a buggy demo IC (Ψ set to [1,0,1] instead of 0) plus a
  mislabeled diagnostic (it printed tr Ψ, not tr C). Corrected here.]
- **CONSTITUTIVE NOTE:** with the LogConf field = Ψ, initialize the conformation field to **0**
  (equilibrium), not [1,0,1]; and to report tr C you must convert Ψ → C via
  `LogConfOldroydB::conformation` (the field is the log, not C).

## The AMR blow-up — ROOT CAUSE FOUND & FIXED
With a refine threshold low enough to actually track the strands, the AMR-on run blew up to NaN
**at a remesh event** (velocity → inf in one step; the implicit Helmholtz/pressure solve returned a
NaN residual), while the *identical* AMR-off run was perfectly healthy (tr C → 500 cap). The
blow-up was **not** physics, **not** a remap overshoot, and **not** an SPD loss. It was a stale
**persistent GPU solver handle**.

- The flow integrators hold persistent elliptic-solver handles (`GpuPoisson`/`GpuPoissonNc`/
  `GpuPoissonMg`, the P4 perf win) and rebuilt them only when the **dof count changed**
  (`h.ndof() != ndof`). But an AMR remesh can refine one region and coarsen another, leaving the
  element count unchanged while the operator (connectivity + per-element metrics) is completely
  different. (Confirmed in the trace: two consecutive frames both had 594 elements, then it blew
  up.) The stale handle then applied the **old mesh's operator** to the **new mesh's field** ⇒
  garbage solve ⇒ one-step blow-up of the coupled flow.
- **Fix** (`gale-gpu/src/flow.rs`): a cheap **topology fingerprint** (`mesh_fingerprint`, hashing
  element count + two opposite corner coords per element) stored per integrator; at the top of every
  `step`, `invalidate_handles_on_remesh` clears all five handle slots when the fingerprint changes,
  forcing a rebuild for the new operator. Static (non-AMR) meshes compute the fingerprint once and
  never rebuild ⇒ zero regression (confirmed: `ns-check`, `ve-check` still PASS; `gpu-amr-flow-check`
  matches CPU to 1.6e-10).
- **Defense-in-depth** (not the root cause, but a legitimate safety net kept in): a **pointwise**
  spectral clamp `clamp_logconf_spectrum(Ψ, b)` (cap each eigenvalue of Ψ to ±ln b ⇒ every C
  eigenvalue in [1/b, b]; reset non-finite nodes to equilibrium). Applied to the conformation as the
  VE integrator reads it (post-AMR-remap), since the cell-mean-preserving `limit_logconf_trace_bound`
  *bails out* when the element mean itself violates the bound — exactly what a bad remap produces.

Also fixed alongside: **trajectory durability** — `TrajectoryWriter::write_frame` now flushes after
each frame, and the demo checks finiteness *before* writing, so a blow-up leaves a valid, openable
`.h5` with only good frames (previously a crash left the whole file unreadable).

**And the SAME ne-detection mistake in the dump** — the AMR trajectory re-emitted topology only on
`ne != last_ne`, so a constant-ndof remesh paired the new field with a STALE topology and the viewer
drew each element at the wrong cell: a blocky horizontal-band scramble. I first misread that scramble
as high-Wi under-resolution; it was overwhelmingly the topology-pairing bug (proven by re-pairing the
identical field data with correct vs stale topology — stale=blocky, correct=smooth). FIX:
`TrajectoryWriter::topology_for(mesh)` re-emits topology only when the geometry actually changes.
The field is much smoother than first reported. A *minor* genuine under-resolution effect remains at
late times once tr C saturates the cap (real strand-tip specks + `trC_min → 0.01`), which is the
honest motivation for stress diffusion below — but it is secondary, not the cause of the glitch.

## What works now
`traj-amr-kolmogorov` runs a **robust** viscoelastic Kolmogorov flow with AMR tracking the strands:
at Wi≈16/El≈4 it develops the elastic instability, the thin birefringent stress strands appear, the
mesh refines to follow them (576 → ~1200 elements over 1000 steps, no blow-up), and the FENE-P-like
trace cap holds tr C at L². The stale-handle bug that capped how aggressively we could refine is
gone.

## Fix path (next)
1. **Longer runs + parameter tuning** to reach developed ET/EIT (Re~1–10, Wi~10–50, El~1–10,
   FENE-P L² ~ few hundred), now that AMR can refine freely.
2. **Possibly** a proper polymer **stress-diffusion** term `κ∇²Ψ` (Sc=ν/κ) if the trace cap alone
   doesn't tame the steep stress fronts — the literature standard. AMR's payoff: resolve the strands
   so a smaller κ (higher Sc, more faithful) suffices vs a uniform mesh.
3. Indicate on a strand-sharp field if needed (Ψ = log C compresses strand contrast: tr C≈300 is
   only Ψ≈5.7); storing/indicating on tr C is an option.
