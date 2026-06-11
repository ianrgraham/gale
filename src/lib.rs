//! `gale` — a GPU discontinuous-Galerkin solver for viscoelastic / incompressible flow.
//!
//! This crate's **library** is the CPU-side core: reference elements, mesh,
//! DG operators, and solvers. It is plain host code (no cuda-oxide) so it builds
//! and unit-tests under ordinary `cargo test`, and serves as the correctness
//! oracle (Milestone 0) against which the GPU kernels are validated. GPU device
//! code (via cuda-oxide `#[cuda_module]`) lives in the binaries / future device
//! modules.
//!
//! See `docs/dg-gpu-fluid-simulation.md` and `docs/implicit-solver-strategy.md`
//! for the design and build order.

// Process-wide thread-caching allocator (default feature `mimalloc`). The parallel
// p-MG build is allocation-heavy across all cores; profiling showed glibc malloc's
// per-call mmap/munmap dominated ~20% of build time and capped scaling at ~32 cores.
// Defined here in the lib root so it applies to every binary depending on `gale`
// (incl. all gale-gpu solver/bench bins) without per-binary boilerplate; opt out with
// `--no-default-features` (e.g. for the pyo3 module).
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod dg;
pub mod sim;
