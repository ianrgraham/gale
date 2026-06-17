# Plan: 100% device-resident, GPU-self-driving simulations (production standard)

**Mandate (user, 2026-06).** A "production" gale GPU sim — one ready to run parameter sweeps — must be
**100% device-resident from dispatch to conclusion**: the host does NO per-step work and issues NO
per-step synchronization. The simulation drives itself on the GPU. Host↔device traffic is limited to
uploading ICs once and an async trajectory dump every N steps (off the critical path). A sim with
per-step host orchestration is a PROTOTYPE, not production.

This is the architecture and the staged plan to get the viscoelastic AMR Kolmogorov sim (and the
flow stack generally) there. It is a major effort; it is staged with a validation gate per stage.

## Why (the cost being removed)

Profiling the current VE step: GPU sustained ~82% util after the `GpuLogConf` persistent-handle fix,
but the remaining ~18% (and all the per-step PCIe traffic) is **host work between kernels** —
upwind lift, trace limiter (eigendecomp), spectral clamp, RK axpy/combine, assembly — each forcing a
`to_host_vec`/upload round-trip and a stream sync. [[gpu-sync-host-device-copies]] (RULE ZERO): these
serialize CPU↔GPU. Eliminating them ALL is the only way to reach the throughput a sweep needs.

## Target architecture

State (all `DeviceBuffer`, resident for the whole run, allocated once / on remesh):
- velocity `ux,uy`, pressure `p`, conformation `Ψxx,Ψxy,Ψyy`, plus solver scratch.
- mesh metrics (`rx,ry,sx,sy,jw`, diff matrices, face/mortar maps) — uploaded once, refreshed only
  inside the on-device AMR step.

Per-step (ONE captured CUDA graph, replayed — no host code between nodes):
1. predictor (explicit force) → 2. pressure Poisson solve (`solve_dev`, while-graph CG) →
3. gradient correction → 4. viscous Helmholtz solves (`solve_dev`) → 5. conformation SSP-RK3:
   `psi_rhs` + **device upwind lift** + **device axpy/combine** + **device trace limiter** +
   **device spectral clamp** → 6. implicit stress-diffusion solve (`solve_dev`).

Time loop (device-driven): a WHILE conditional graph (`set_conditional`/`capture`) advances N steps
(or until a device-computed stop criterion) with the whole-step graph as its body; inner solves are
nested conditional sub-graphs. Already proven for the PCG inner loop (`while_graph_pcg_check`).

I/O: every N steps, an async `cuMemcpyDtoHAsync` of the dump fields on a side stream; never blocking.

## What already exists (build on it)

- Device-resident solve + vector primitives: `GpuPoissonMg::{solve_dev, gradient_dev, axpy_dev,
  scal_dev, alloc_field, stream}`; foundation bins `device_resident_flow_check`,
  `sbm_cylinder_resident_check`. (memory [[device-resident-flow]])
- Conditional/while CUDA graphs in the cuda-oxide fork: `set_conditional`, capture, the while-graph
  PCG (`pcg_cond`, `capture_while`). (memory [[cuda-oxide-conditional-graphs]])
- Persistent kernel-module handles: `GpuPoisson`, `GpuLogConf`. (memory [[gpu-oneshot-kernel-antipattern]])

## Staged plan (each stage ends with a validation gate: GPU-vs-CPU match + profile showing no
## per-step host sync)

**Stage 1 — device-resident host-op kernels (no graph yet).** Port the per-step host math to device
kernels operating on resident buffers: RK3 axpy/combine, the trace limiter (per-node `sym_eig`+exp
bisection — a device kernel), the spectral clamp, the upwind advection lift (DG face flux; conforming
first, mortar after), stress-divergence/body-force assembly. Validate each bit-vs-CPU. This is the
bulk of the kernel work and removes the round-trips even before graphing.

**Stage 2 — capture the whole fixed-mesh step as one CUDA graph; device-driven loop.** Assemble the
resident step from Stage-1 kernels + `solve_dev`, capture it, wrap N steps in a while-graph. Result:
a **fixed-mesh** VE sim that is 100% device-resident — production for non-AMR sweeps. Gate: trajectory
matches the host-orchestrated path; nsys shows `cuMemcpy`/`cuStreamSynchronize` flat in step count.

**Stage 3 — AMR on the GPU (the crux; research-gated).** Run a deep-research + design pass FIRST
(per [[research-before-architecture]]) on GPU-resident adaptive meshing without host sync. Leading
approach: a **fixed-max-level masked representation** — allocate for the maximum refinement depth and
activate/deactivate cells via device masks, so "remeshing" is a mask + metric refresh (device
kernels) with NO host-side connectivity rebuild, and can run as a conditional sub-graph inside the
time loop. Port: smoothness indicator (per-element device kernel — easy), refine/coarsen flagging +
2:1 balance (device), mortar/metric refresh from masks (device), field prolong/restrict remap
(device kernels — the operators are local). Alternative (true dynamic device-side connectivity
rebuild) is far harder; evaluate in the research pass. Gate: AMR sim matches the host AMR trajectory;
adaptation triggers with zero host sync.

**Stage 4 — promote + sweep.** Wire the device-resident self-driving integrator behind the existing
`Simulation` API, mark it production, and run the elastic-turbulence parameter sweep on it.

## Sequencing recommendation

Stages 1–2 are tractable, high-value, and deliver production fixed-mesh sweeps; they also de-risk the
graph/loop machinery. Do them first. Stage 3 (GPU AMR) is the hard, research-grade part — gate it on
a verified deep-research pass before committing to the masked-level vs dynamic-rebuild design.
