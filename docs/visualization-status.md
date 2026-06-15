# Visualization & trajectory I/O — status (2026-06-15)

Where the viewer/dump stack stands, so we can pick it back up later. The headline solver
work has moved on to the physics frontier; this is a checkpoint, not a finished product.

## What exists

### Trajectory format (`gale-traj`)
- HDF5 container (`hdf5-metno`, statically-linked bundled libhdf5 + zlib — needs `cmake`
  at build time). One file, or a size-capped split set `<stem>.NNNN.h5` (`create_split`,
  `TRAJ_MAX_MB`).
- Geometry/fields decoupled by a **topology id**: the mesh (nodes, connectivity, order)
  is written once per distinct topology; each frame references it. Fields are per-frame,
  `f32`, chunked + gzip. Rigid bodies: static radii + per-frame poses `[cx, cy, φ]`.
- Spirit of GSD (HOOMD) — fixed geometry amortized, only what changes per step is stored.
  AMR / moving-geometry would write a new topology when the mesh changes (not yet
  exercised — all current dumps are fixed-mesh).
- API: `TrajectoryWriter::{create, create_split, write_mesh2d, write_body_radii,
  write_frame, n_frames, n_files}`.

### Dumps (`gale-gpu/src/bin/traj_*.rs`, behind the `traj` feature)
- `traj-cylinder` — Newtonian SBM cylinder (Schäfer–Turek, Re≈20), device-resident.
- `traj-ve-cylinder` — **Oldroyd-B viscoelastic** SBM cylinder (same geometry). Flow stays
  device-resident on `GpuPoissonMg`; the conformation tensor is host-orchestrated each
  step (download C → `∇·τ_p` body force → upload; advance C with the new velocity via the
  bound-preserving SSP-RK3). Dumps `u` and the polymer-stretch diagnostic `tr C`. Knobs:
  `VE_BETA`, `VE_LAMBDA` (De), `VE_NU0` (Re), `VE_EPS`/`VE_BMAX` (SPD limiter).
- `traj-lid`, `traj-suspension` — lid-driven cavity, many-body suspension.
- **`traj-amr`** — dynamically adaptive 2D flow (`GpuDualSplitting` + `AmrUpdater`, refine+coarsen
  on a diffusing vortex): re-emits a NEW topology whenever the mesh changes, so the trajectory
  carries a per-frame mesh.
- **`traj-amr-dipole`** — the AMR showcase: a self-advecting counter-rotating vortex pair whose
  refined region TRACKS the moving cores and coarsens the wake (env: `AMR_N/NU/DT/AMP/S2/SEP/
  REFINE/COARSEN`). View with `gale-view … --grid`.
- Common env knobs: `SBM_NY`, `SBM_DT`, `TRAJ_STEPS`, `TRAJ_EVERY`, `TRAJ_MAX_MB`.

### Viewers
- **`gale-view`** (Rust, wgpu/winit) — the primary viewer.
  - Offscreen → PNG (headless Vulkan) or `--window` (interactive: arrow scrub, Space
    play/pause, Home/End, Esc). `--watch` polls a split set and appends frames live (watch
    a running sim, e.g. over SSHFS from the compute box).
  - Per-element CPU tessellation (inter-element jumps show — honest DG), Turbo colormap on
    the GPU. Embedded bodies masked by a **fragment-shader circle discard at the true
    radius** (resolution-independent, no staircase) + a thin outline.
  - Flags: `--field`, `--comp 0|1|mag`, `--frame N|--all`, `--out`, `--height`, `--vmin`,
    `--vmax` (fixed colour window), `--window`, `--watch`, `--fps`.
  - Robustness: clamps the surface to the adapter's max texture size; clean process exit
    (no teardown segfault); tolerant of missing/incomplete files under `--watch`.
- **`gale-traj/python/view_traj.py`** (matplotlib) — montage + GIF, multi-file, body
  overlays, NaN-aware range. Kept for quick scripting / GIFs.

### Honest near-wall rendering (SBM) — `gale::dg::sbm_reconstruct`
The embedded boundary is the load-bearing correctness detail. At the **dump phase** (never
in the viewer), `sbm_reconstruct` fills the gap between the staircased SBM surrogate edge
and the true boundary with the field the method itself implies: a 2nd-order Taylor
extrapolation (value + ∇ + Hessian) from the nearest surrogate node — the same high-order
reconstruction `sbm_force_torque` uses for drag.
- An inactive element with **any** fluid node straddles the surface ⇒ all its nodes are
  reconstructed (so the tessellator drops no sub-cells); fully-solid elements stay NaN
  (invisible behind the true-circle mask, and out of the colour range).
- The extrapolation is **envelope-limited** to within one data-span of the real surrogate
  values, so a sharp field (e.g. the `tr C` stress concentration) can't blow up to ±100s.
- Geometry is always drawn at the **true radius** — see the honest-visualization rule: no
  fudging size/data to hide artifacts.

## Known limitations / TODO (when we return)
- **No colorbar / axes / scale** in `gale-view` — values are printed to stdout only. A
  legend + physical axes would make stills publication-ready.
- **`tr C` near-wall speckle** — the reconstruction is bounded but still noisy right at the
  surface where `tr C` varies sharply; a local (not global) envelope or a gentler
  reconstruction order there would clean it up.
- **AMR / moving-mesh trajectories** — the format supports per-topology re-emit, but no
  dump exercises it yet; the viewer assumes one topology for the whole run. Needed once
  GPU AMR lands.
- **Conformation tensor viz** — only `tr C` (stretch) is dumped; principal-stretch
  direction / full tensor glyphs would show orientation.
- **Side-by-side / difference views** (Newtonian vs viscoelastic) are manual today.
- `--vmin/--vmax` are global; no percentile auto-range or per-frame normalization toggle.
