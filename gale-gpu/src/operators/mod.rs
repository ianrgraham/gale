//! GPU DG operators as reusable library components, grouped by role.
//!
//! Each submodule carries its own `#[cuda_module]` device kernels plus host launch
//! wrappers. They share one crate-wide device bundle (see [`crate`] docs), so kernel
//! export names must be unique across all of gale-gpu.

#[cfg(feature = "autodiff")]
pub mod ad_probe; // minimal std::autodiff → cargo-oxide → cuda-host regression probe
pub mod advection;
pub mod advection3d;
pub mod burgers;
pub mod euler;
pub mod logconf;
pub mod logconf3d;
pub mod oldroyd;
pub mod oldroyd3d;
pub mod poisson;
pub mod poisson3d;
pub mod poisson_nc;
