//! GPU immersed-boundary volume penalization — reusable library component.
//!
//! The implicit Brinkman relaxation `u ← (u + β·u_s)/(1+β)`, `β = χ·dt/η_b`, applied
//! per node. Embarrassingly parallel, libdevice-free. Validated bit-for-bit against
//! `gale::dg::VolumePenalization::apply`.
//!
//! This is a *second* `#[cuda_module]` in the gale-gpu crate (alongside
//! [`crate::operators::advection`]). cuda-oxide compiles every `#[kernel]` in the
//! crate into one device bundle keyed by the crate name, so multiple cuda_modules
//! coexist as long as their kernel export names are unique crate-wide.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use gale::dg::{Mesh2d, VolumePenalization};

#[cuda_module]
mod kernels {
    use super::*;

    /// In-place implicit penalization of both velocity components. One thread per dof.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn penalize(
        mut ux: DisjointSlice<f64>,
        mut uy: DisjointSlice<f64>,
        mask: &[f64],
        usx: &[f64],
        usy: &[f64],
        r: f64,
        nn: u32,
    ) {
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let g = e * (nn as usize) + m;
        let beta = r * mask[g];
        let denom = 1.0 + beta;
        let (bx, by) = (beta * usx[g], beta * usy[g]);
        if let Some(o) = ux.get_mut(thread::index_1d()) {
            *o = (*o + bx) / denom;
        }
        if let Some(o) = uy.get_mut(thread::index_1d()) {
            *o = (*o + by) / denom;
        }
    }
}

/// Apply implicit volume penalization to the velocity field `(ux, uy)` in place on
/// the GPU, using a precomputed [`VolumePenalization`] mask/solid-velocity field.
/// Bit-for-bit equal to `VolumePenalization::apply`.
pub fn penalize_apply(
    mesh: &Mesh2d,
    ux: &mut [f64],
    uy: &mut [f64],
    pen: &VolumePenalization,
    dt: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    assert_eq!(ux.len(), ndof, "ux length must be n_elements·n_nodes");
    assert_eq!(uy.len(), ndof, "uy length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let mut ux_dev = up(ux)?;
    let mut uy_dev = up(uy)?;
    let mask_dev = up(&pen.mask)?;
    let usx_dev = up(&pen.us_x)?;
    let usy_dev = up(&pen.us_y)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.penalize(
        &stream, cfg, &mut ux_dev, &mut uy_dev, &mask_dev, &usx_dev, &usy_dev,
        dt / pen.eta_b, nn as u32,
    )?;
    ux.copy_from_slice(&ux_dev.to_host_vec(&stream)?);
    uy.copy_from_slice(&uy_dev.to_host_vec(&stream)?);
    Ok(())
}
