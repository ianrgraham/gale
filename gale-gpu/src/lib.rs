//! `gale-gpu` — GPU device kernels + host launch wrappers for gale's DG operators,
//! as **reusable library components** (not one-off binaries).
//!
//! This crate carries the `#[cuda_module]` device code, so it must be built with
//! the cuda-oxide codegen backend (`cargo oxide`). It depends on the pure-host
//! `gale` lib for mesh/operator types; `gale` does not depend on it, so `gale`'s
//! `cargo test` remains a pure-host CPU oracle. Each GPU operator here is validated
//! bit-for-bit against that oracle.
//!
//! ## Device-bundle model (important for adding operators)
//!
//! cuda-oxide compiles **every** `#[kernel]` in the crate into a single device
//! bundle keyed by the crate name (`CARGO_PKG_NAME`), and each `#[cuda_module]`'s
//! generated `load()` loads that same bundle. So:
//!
//! - Multiple `#[cuda_module]`s per crate are fine (one per operator module here).
//! - **Kernel export names must be unique crate-wide** — a non-generic `#[kernel]`
//!   exports under its bare fn name (no module namespacing). Hence `advect2d_rhs`
//!   vs a future `advect3d_rhs`, and shared linear-algebra primitives are defined
//!   once rather than per-operator.
//!
//! Reusing a `#[cuda_module]` from a *dependency library crate* relies on the fork
//! fix that anchors the embedded artifact object so the linker retains it across
//! the rlib boundary (see `docs/cuda-oxide-codegen-notes.md` §3 / fork commit
//! `02129b8`).

pub mod distributed;
pub mod flow;
pub mod immersed;
pub mod operators;

// Backwards-compatible flat re-exports of the most-used host API.
pub use distributed::{multigpu_advection_2d, multigpu_advection_3d};
pub use flow::GpuStokes;
pub use immersed::penalize_apply;
pub use operators::advection::{advection_rhs, GpuAdvection};
pub use operators::advection3d::advection3d_rhs;
pub use operators::burgers::burgers_rhs;
pub use operators::euler::euler_rhs;
pub use operators::logconf::logconf_psi_rhs;
pub use operators::oldroyd::oldroyd_conf_rhs;
pub use operators::poisson::{
    helmholtz_cg_solve, poisson_apply, poisson_cg_solve, poisson_pcg_solve, pressure_cg_solve,
};
