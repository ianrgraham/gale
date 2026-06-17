# Stage 3: GPU-resident AMR — design & staged plan (from the research gate)

Research gate run 2026-06 (5-angle Sonnet deep-research workflow), then a **separate adversarial
verification pass** (12 Sonnet fact-checkers, one per claim/citation). **Verification result: the
core design VERDICT HOLDS independently of all citation issues** — masked fixed-level rests on a hard
CUDA constraint (not a paper) and Ψ-not-C rests on basic interpolation theory + matrix-exp. But the
research synthesis's bibliography had real errors (corrected/removed below); this doc cites concepts,
not those IDs, so the design is unaffected. Citations to AVOID / correct:
- **WRONG — do not cite:** "Chan, Fernandez, Carpenter, arXiv:2005.03237, standard mortars
  energy-stable for *incompressible* NS." Actual paper (Chan, Bencomo, Del Rey Fernandez) is
  *compressible* and argues the OPPOSITE (an entropy correction IS needed). Not used here.
- **Attribution fixes:** arXiv:2509.19701 = Poptani et al. (FV/WENO5 Parthenon-VIBE, ~22× serial-CPU
  AMR overhead), NOT "Trotta / DG p=3". arXiv:2604.21600 = Yang & Fu, negative-entry result is
  **Remark 3.1** (not Pan/Bohm/Winters; that's the 2017 arXiv:1712.10234). GCL metric-refresh is
  classical (Thomas–Lombard 1979; Farhat et al. JCP 174:669, 2001), not novel to "MARUT".
- AGAL (arXiv:2308.08085) and the Wang/Witherden/Jameson GPU FR/DG result are the real support for
  on-device masked AMR — but framed as **GPU-native research codes**, not "production" (AMReX/Parthenon
  still keep CPU-side mesh management).

## Verdict: masked fixed-max-level (NOT dynamic device-side connectivity rebuild)

Reason it's structural, not just perf: gale's device-resident step runs the time loop as a CUDA
while-graph, and **`cudaMalloc`/topology change is illegal inside a conditional graph node body**. A
true dynamic connectivity rebuild needs allocation per adapt ⇒ breaks the graph invariant (or forces
host orchestration). The masked approach makes "remesh" = flip activation bits + refresh metrics in
PRE-ALLOCATED fixed-capacity arrays — no allocation, no topology change, graph-legal.

Memory check (Titan V 12 GB): 64² base, L_max=2, p=3 ⇒ N_max=65 536 blocks, ~83 MB total. 256²/L_max=2
⇒ ~5.4 GB (fits). Dynamic rebuild only wins when the finest-level pre-allocation exceeds ~4–6 GB
(e.g. 256²/L_max=3 ≈ 86 GB) — not our near-term target. We commit to masked; revisit only if we need
L_max=3 on a large base.

NOTE on tooling: the research says "Thrust" (sort/scan/copy_if/custom-allocator). gale is
cuda-oxide/Rust, NO Thrust — we write scan/compaction/sort as our own `#[kernel]`s (the
dot_partial/reduce_scalar pattern already shows device reductions). This actually SIDESTEPS the #1
risk (Thrust's internal cudaMalloc inside a graph) — our kernels allocate nothing at launch.

## Device data layout (all pre-allocated, resident)

- Block metadata, one slot per max-capacity block (`N_max = base·4^L_max`): `block_level`,
  `block_parent`, `block_children[4]`, `block_nbr[4]` (face neighbours), `block_active`, `block_morton`.
- Active-set + free-list (AGAL pattern): `id_setL[L+1][...]` (compacted active IDs per level) +
  `id_setL_count`, `gap_set` (recycled IDs) + `gap_count`.
- Fields sized for finest level: `u,v,p` `[N_max][16]`, `psi[N_max][16][3]` (or 3 separate arrays for
  coalescing) — p=3 ⇒ 16 DOF/elem.
- Metrics (refresh after mask flip): `J`, contravariant components `[N_max][16]`. Cartesian quadtree ⇒
  child metric = parent/scale (trivial; satisfies discrete GCL).
- Mortar face list (extends GpuPoissonNc): `mortar_faces[M_max]` + `mortar_count`; constant 2:1
  prolong/restrict matrices `P_C2F`/`P_F2C` in constant memory (same matrices GpuPoissonNc uses).
- Indicator/flag: `indicator[N_max]`, `refine_flag[N_max] (+1/0/-1)`, device `adapt_pending`.

## On-device adapt pipeline (each a kernel, runs inside a conditional graph node)

0. **Indicator** (every N_adapt steps): Persson–Peraire smoothness `Se` on `tr(Ψ)` per block (local
   modal transform + 2 inner products; no halo). Set `adapt_pending` via a device any_of.
1. **Flag**: `Se>refine_thr & level<L_max ⇒ +1`; `Se<coarsen_thr & level>0 & all 4 siblings agree ⇒ -1`.
2. **2:1 balance** (2–3 passes over `block_nbr`): revert a coarsen flag if any neighbour is finer.
   (2–3 passes suffices at L_max=2 — imbalance propagates ≤2 hops — but is NOT a general bound;
   deep trees need O(L_max) passes, p4est-class. Add a device any_of "no flag changed" early-out.)
3. **Slot alloc**: prefix-scan over flags → claim/return child IDs from `gap_set` (scan, NOT atomics,
   to avoid contention); update children/level/parent/active.
4. **Metric write** for newly-activated blocks — MUST land in the device metric arrays before any
   operator kernel reads them. Cartesian quadtree = closed-form arithmetic, NO interpolation:
   `det J_child = det J_parent / 4` (2D, per level). (This is just metric consistency / discrete SCL
   free-stream preservation — the classical GCL is for time-varying/curvilinear meshes, not this
   static affine case; the no-interpolation shortcut does NOT generalize to curved elements.)
5. **Mortar rebuild**: compact non-conforming face pairs (level mismatch across `block_nbr`) into
   `mortar_faces` — the array GpuPoissonNc already consumes.
6. **Remap**: prolong (refine) = apply 2:1 restriction matrix parent→4 children; restrict (coarsen) =
   conservative L2 projection 4 children→parent (precomputed 16×16 LU in shared mem).

Fits the device-driven loop: `OUTER while{ for N_adapt{ step-graph } ; IF(adapt_pending){ kernels 0–6;
clear } }`. Solver kernels guard `if tid>=active_count return;` (active_count a device scalar) ⇒ no host
grid-resize, zero host involvement.

## Conservation / stability (directly addresses our past exp-overshoot bug)

- **Remap Ψ = log C, never C.** Lagrange interpolation matrices have negative entries for p≥2 (a
  generic property of *any* node set — GLL gives positive *quadrature* weights, not a non-negative
  *interpolation* operator; see Yang & Fu arXiv:2604.21600 Remark 3.1 for the DG-AMR context). So a
  component-wise 2:1 prolongation of the SPD tensor C can produce a non-SPD result (Zhang et al.,
  Phys. Fluids 2023). In log space any linear op stays symmetric and exp(Ψ) is SPD by construction.
  gale already stores/transports Ψ ⇒ no conversion. (Exactly the fix we found empirically for the
  AMR-remap collapse earlier this session — verification confirms the theory.)
- **Zhang–Shu pre-limiter before prolong** (reuse gale's validated bound-preserving limiter): scale
  nodal deviation toward the cell mean so child reconstructions stay admissible.
- **L2 restrict is conservative** (parent constant mode = Jacobian-weighted child-average).
- **Metric consistency**: write child metrics (step 4, `det J_child = det J_parent/4^Δlevel`, pure
  arithmetic) before the next DG eval; affine elements ⇒ discrete SCL / free-stream exact. (Informal
  "GCL" shorthand; true GCL is for moving/curvilinear meshes.)
- **Mortar conservation**: fine-side ownership rule (fine block applies P_C2F, evaluates flux, scatters
  the 2×-scaled projected flux to the coarse side) — GpuPoissonNc's existing pattern.

## Staged implementation plan (validation gate per stage)

- **3a** Device indicator + flag kernels (no remesh). **DONE (indicator):** `operators/amr.rs`
  `smoothness_se` kernel + `smoothness_se_gpu`; `amr-indicator-check` matches host
  `SmoothnessIndicator` to 3.9e-18 (bit-exact). Flag kernel folds in once the masked block metadata
  (level/siblings) exists in 3c. **← next: 3b/3d.**
- **3b** Device 2:1 balance pass over flat neighbour arrays. Gate: matches host balance on random flag
  sets. Medium.
- **3c** Masked pre-allocation + gap_set; run the EXISTING GpuPoissonNc + resident step off the device
  block arrays on a STATIC mesh. Gate: cylinder C_D identical to pre-Stage-3. High (most invasive).
  - **3c.1 DONE:** `gale-gpu/src/amr_mesh.rs` `GpuAmrMesh` — masked fixed-capacity quadtree block pool
    (level/ix/iy/active/parent/children flat arrays + `gap` free-list), `refine`/`coarsen` by slot
    activation. `amr-mesh-check`: active-set + levels correct over 2 levels, refine→coarsen round-trips
    exactly (free pool restored, no leaks). The AGAL core. (Host arrays now, laid out for device upload.)
  - **3c.2 DONE:** `GpuAmrMesh::build_neighbors` fills `block_nbr/block_nbr2/block_nbr_kind` (face
    order −x,+x,−y,+y; kind 0/1/2/3 = boundary/same/coarser/finer) via 2:1 quadtree coord adjacency.
    `amr-connectivity-check`: vs host `cartesian_refined`, 76 active = 76 elems, all 304 faces match
    classification, 0 mismatches.
  - **3c.2.5 DONE:** `GpuAmrMesh::refined_base_cells()` — recovers the refine-set from the block
    structure (the host-AMR bridge: derive R → `cartesian_refined` drives the existing solver for the
    single-level static case). Validated in `amr-connectivity-check`.
  - **3c.3 NEXT — BIGGER than first scoped.** SCOPE FINDING: the resident integrators (`GpuResidentNs`/
    `GpuResidentVe`) are **uniform-rectangular ONLY** (`PMultigrid::from_mesh` returns `None` on a
    refined mesh; no `GpuPoissonNc` in `resident.rs` — Stage 1–2 was fixed-UNIFORM scope). So driving
    the resident step off the masked structure requires FIRST a **device-resident non-conforming
    solve** (port `GpuPoissonNc` into the resident step / `solve_dev`-style, masked), THEN the
    block→operator wiring. Two sub-pieces:
    - **3c.3a DONE:** `GpuPoissonNc::solve_dev(rhs_dev, x0, out, reaction, neumann_tags, deflate, tol,
      maxit)` — device-native NC CG (field resident, mortar matvec on device, only the CG scalar reads
      back), + `stream()`/`upload`/`download`/`alloc` helpers. `amr-nc-resident-check`: bit-identical to
      host `solve` on a refined mesh — Helmholtz rel 0.0 (92 it), deflated pressure rel 0.0 (556 it).
    - **3c.3b DONE:** added the device-resident NC assembly primitives to `GpuPoissonNc`
      (`gradient_dev`/`fma2_dev`/`scal_dev`/`copy_dev`/`rhs_madd_dev`/`axpy_dev` + kernels
      `scal_nc`/`fma2_nc`/`rhs_madd_nc`). `device-resident-flow-nc-check`: a full dual-splitting NS step
      on a 2:1 refined mesh, entirely device-resident (assembly + both solves via GpuPoissonNc), matches
      the host trajectory to 1.5e-12 (lid cavity, 8 steps). **Bug fixed:** the NC deflated-CG broke down
      (0/0→NaN) on a zero RHS (first projection, div=0) — added the convergence-before-α + `pap>0`
      guards to `solve`/`solve_dev` (the conforming path already had them). No regression
      (amr-nc-resident-check, gpu-amr-flow-check unchanged).
    - **3c.3c NEXT:** drive `GpuPoissonNc`'s metrics/mortar from the masked block arrays directly (vs
      the current Mesh2d build); + a unified resident integrator dispatching conforming↔NC. The bridge
      (refined_base_cells→cartesian_refined→GpuPoissonNc) already lets the masked set drive the solver.
- **3d** Remap kernels (prolong/restrict) in isolation. **DONE:** `operators/amr.rs` `prolong2d`
  (refine, exact) + `restrict2d` (coarsen, conservative L2) + `prolong_gpu`/`restrict_gpu`;
  `amr-remap-check` matches host `RefineQuad` to 1.3e-15/1.8e-15 and cell-average conservation 6.7e-16.
  (Added host accessors `RefineQuad::axis_matrix()/weights()`.) TODO when wired to Ψ: apply the
  Zhang–Shu log-conf pre-limiter before prolong (reuse `limit_logconf_trace`) so exp(Ψ) stays SPD.
- **3e DONE:** end-to-end device-resident ADAPTIVE NS. `device-resident-adapt-check`: device-resident
  steps (assembly+solves via GpuPoissonNc, no per-step transfer) between host-orchestrated remeshes
  (4-stage refine+coarsen schedule), fields remapped on adapt (`remap_component_flat` — the only
  host↔device round-trip, every N steps not per step). Matches the host-orchestrated adaptive reference
  to 1.1e-11 (lid cavity, 70-elem final mesh). **The GPU-resident-per-step adaptive loop works end to
  end** — remesh is host-orchestrated (cheap, infrequent); all per-step compute is on the GPU.
- **3f** Embed the adapt sequence in a CUDA conditional graph node. Gate: graph-legal, bit-identical to
  3e, adapt overhead <5%. High.
- **3g** Solver kernels read the dynamic `mortar_faces` directly. Gate: matches static-mesh reference.
  Medium. (Audit GpuPoissonNc face-iteration order — prefer order-agnostic atomic accumulation.)

## Top risks to probe first

1. Device scan/compaction + free-list (our own kernels, no Thrust) — prototype prefix-scan slot
   allocation, confirm no atomic contention. 2. 2:1 balance convergence in ≤3 passes on adversarial
   (thin-strip) patterns — stress-test on host first. 3. GCL incl. FACE metrics refreshed after mask
   flip (run a div-free field through one adapt, check conservation residual). 4. GpuPoissonNc face
   ordering vs compacted mortar list — make it order-agnostic (atomics) before 3g.

## Note vs current code
gale already has the validated host AMR (indicator/flag/balance/remap in `dg::amr`) and the
`GpuPoissonNc` 2:1 mortar operator — the host versions are the ORACLES to validate each device stage
against. This is a port + restructure to device-resident masked arrays, not a from-scratch method.
