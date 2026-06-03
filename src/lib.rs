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

pub mod dg;
pub mod sim;
