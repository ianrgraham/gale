//! GPU DG operators as reusable library components, grouped by role.
//!
//! Each submodule carries its own `#[cuda_module]` device kernels plus host launch
//! wrappers. They share one crate-wide device bundle (see [`crate`] docs), so kernel
//! export names must be unique across all of gale-gpu.

pub mod advection;
pub mod advection3d;
pub mod burgers;
pub mod euler;
pub mod logconf;
pub mod oldroyd;
pub mod poisson;
