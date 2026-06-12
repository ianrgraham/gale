# gale

**A GPU-native, high-order solver for incompressible and viscoelastic flow around immersed objects — written in native Rust with CUDA device kernels.**

gale discretizes space with the **discontinuous Galerkin spectral element method (DG-SEM)** and runs the expensive work — elliptic solves, polymer-stress transport, immersed-boundary penalization — entirely on the GPU, across multiple devices, validated bit-for-bit against a pure-CPU reference.

> The headline target is **viscoelastic, particle-laden, microfluidic suspensions**: polymer solutions carrying suspended particles through small channels. That is a genuinely hard corner of CFD — low Reynolds number (weak inertia) but high Weissenberg number (strong elasticity) — and it drives every design choice in the code.

---

## Why this project exists

This grew out of PhD-era research on viscoelastic fluid flows, where the available tooling (OpenFOAM + rheoTool) felt archaic and limiting. gale is an attempt to do it properly with modern tools:

- **GPU-native, not GPU-accelerated.** Device kernels are written in Rust via the [`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) toolchain and compiled to PTX — no calling out to a hand-written C++/CUDA library.
- **Rust end to end.** One language for host orchestration and device kernels, with a clean path to a Python entry point via pyo3.
- **Multi-GPU as a first-class concern**, not an afterthought.
- **First-class immersed boundary method** for embedding solids without meshing their surfaces.

## What's built today

Everything below runs **on the GPU** and is validated against a CPU oracle to ~10⁻¹⁴ relative error.

| Capability | Status |
|---|---|
| **DG-SEM spatial discretization** (hyperbolic + elliptic operators, 2D & 3D) | ✅ |
| **Incompressible Navier–Stokes** (dual-splitting projection) | ✅ |
| **Viscoelastic flow** — Oldroyd-B / Giesekus / FENE-P, log-conformation formulation | ✅ |
| **GPU linear solvers** — CG, preconditioned CG, p-/h-multigrid (mesh-independent) | ✅ |
| **IMEX time integration** for high-Wi stiffness (ARS/ARK, implicit local relaxation) | ✅ |
| **Bound-preserving & free-energy diagnostics** for the conformation tensor | ✅ |
| **Immersed boundaries** — penalization-based, rigid particles | ✅ |
| **Adaptive mesh refinement** (2D; non-conforming faces) | ✅ |
| **Multi-GPU** — distributed advection + distributed SIPG matvec, P2P halo exchange | ✅ |

**The capstone:** an immersed rigid particle in viscoelastic flow, end-to-end on the GPU in both 2D and 3D — exercising the elliptic solvers, dual-splitting flow, log-conformation polymer model, and immersed-boundary penalization together.

See [`docs/`](docs/) for design notes and verified research reports, and the roadmap chapter of the book for what is *not* yet done (3D adaptivity, true two-phase flow, hp-adaptivity).

## Design philosophy

1. **The CPU is the oracle.** The pure-host `gale` library implements every operator on the CPU; `gale-gpu` re-implements them as device kernels and is checked to match. The CPU library is a clean, `std`-only reference you can read to understand the math without GPU noise.
2. **Expensive work on the GPU, cheap work simple.** Flow solvers are *hybrid*: the costly elliptic solves run entirely on-device, while cheap element-local assembly reuses the validated host code.
3. **High order, because the application demands it.** Spectral elements give exponential accuracy on smooth flow — fewer, larger elements — which matters when you eventually want many particles in a domain.
4. **Validate everything, be honest about what isn't done.** Every kernel has a `*-check` binary comparing it to the oracle; planned-but-unbuilt features are tracked, not hidden.

## Repository layout

```
src/                  # `gale` — pure-host CPU core (the oracle): reference elements,
  dg/                 #   mesh, DG operators, solvers, AMR, immersed boundaries
  sim/                #   simulation framework (state, integrators, stage hooks)
gale-gpu/             # `gale-gpu` — CUDA device kernels + host launch wrappers
  src/operators/      #   per-operator device modules (advection, poisson, oldroyd, logconf, …)
  src/bin/            #   *-check validation binaries and micro-benchmarks
examples/             # standalone examples (incl. Enzyme autodiff probes)
docs/                 # design notes + verified deep-research reports
book/                 # "The Theory of gale" — an mdBook on the numerical methods
```

## Architecture

`gale` (the workspace root crate) is **pure host code** — no cuda-oxide — so it builds and unit-tests under ordinary `cargo test` and serves as the correctness oracle. All GPU device code lives in the `gale-gpu` member crate, which depends on `gale` (but not vice-versa). Kernels are compiled into a single crate-wide device bundle by the cuda-oxide codegen backend.

## Getting started

### Requirements

- An NVIDIA GPU with FP64 support (developed on **2× Titan V**, sm_70; native double precision)
- A pinned nightly Rust toolchain (see [`rust-toolchain.toml`](rust-toolchain.toml))
- The [`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) toolchain and the `cargo oxide` subcommand

> gale currently builds against a [local fork of cuda-oxide](https://github.com/NVlabs/cuda-oxide) (path dependencies in [`Cargo.toml`](Cargo.toml)) carrying fixes — typed-pointer bitcast, NVVM dialect default, `memcpy_peer_async` for P2P, and cross-crate artifact-anchor linking — that are upstream-contribution targets.

### Build & test

```bash
# CPU core: builds and unit-tests with the standard backend (the oracle)
cargo test

# GPU crate: requires the cuda-oxide codegen backend
cargo oxide build

# Run a validation binary (GPU kernel vs. CPU oracle)
cargo oxide run --bin ns_check          # incompressible Navier–Stokes
cargo oxide run --bin ve_ibm_check      # viscoelastic flow + immersed particle (2D)
cargo oxide run --bin ve3d_ibm_check    # viscoelastic flow + immersed particle (3D)
```

The `gale-gpu/src/bin/` directory holds dozens of `*-check` binaries (one per operator) plus profiling and multi-GPU micro-benchmarks (`pcie_crossover`, `multigpu_matvec_check`, `mg_profile`, `roofline_poisson`, …).

## The book

[`book/`](book/) contains *The Theory of gale* — a from-the-ground-up guide to the numerical methods, with the reasoning (especially numerical stability) behind each design choice. Build it with [mdBook](https://github.com/rust-lang/mdBook):

```bash
cd book && mdbook serve --open
```

## Roadmap

- Python entry point via pyo3
- 3D adaptive mesh refinement and hp-adaptivity
- Entropy-stable / structure-preserving spatial operators for guaranteed-SPD viscoelastic transport
- Differentiable solver (Enzyme-through-cuda-oxide) for inverse rheology / parameter inference
- Consumer-GPU precision path (FP32 + double-double emulation for cards without native FP64)

## License

See repository for license details.

---

*gale is a research code, built in collaboration with Claude (Anthropic).*
