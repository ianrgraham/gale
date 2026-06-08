//! Per-region boundary conditions for incompressible flow.
//!
//! Boundary faces carry an integer `tag` (the [`Mesh2d::rectangular`] builder assigns
//! `0=bottom, 1=right, 2=top, 3=left`). A [`BoundaryConditions`] maps each tag to a
//! [`FlowBc`]; tags without an explicit entry fall back to the `default`. This is the
//! user-facing way to say "west = parabolic inflow, east = outflow, walls = no-slip",
//! replacing the single global velocity closure of the legacy
//! [`Stokes::step`](crate::dg::stokes::Stokes::step) API.
//!
//! It is consumed by [`Stokes::with_bcs`](crate::dg::stokes::Stokes::with_bcs), which
//! translates each region into the right pair of elliptic-operator settings:
//!
//! | [`FlowBc`]        | velocity (Helmholtz) | pressure (Poisson)   |
//! |-------------------|----------------------|----------------------|
//! | `NoSlip`          | Dirichlet `u = 0`    | Neumann              |
//! | `Velocity(u_s)`   | Dirichlet `u = u_s`  | Neumann              |
//! | `Outflow`         | Neumann (natural)    | Dirichlet `p = 0`    |
//!
//! The `Outflow` row is the important coupling: pinning `p = 0` there removes the
//! pure-Neumann pressure null space, so when an outflow is present the pressure solve
//! needs no deflation. With no outflow (all walls/inflow) the system is singular and
//! the deflated CG is used, exactly as in the legacy all-Dirichlet path.

use super::mesh::Mesh2d;
use std::collections::HashMap;

/// The boundary condition imposed on one boundary region (tag) of an incompressible
/// flow.
pub enum FlowBc {
    /// No-slip wall: `u = 0` (velocity Dirichlet, pressure Neumann).
    NoSlip,
    /// Prescribed velocity `(u, v) = u_s(x, y, t)` — an inflow profile or a moving
    /// wall (velocity Dirichlet, pressure Neumann).
    Velocity(Box<dyn Fn(f64, f64, f64) -> (f64, f64)>),
    /// Traction-free outflow: natural (Neumann) velocity, pressure pinned to `0`.
    Outflow,
}

impl FlowBc {
    /// A prescribed-velocity BC from a closure `(x, y, t) -> (u, v)`.
    pub fn velocity(f: impl Fn(f64, f64, f64) -> (f64, f64) + 'static) -> Self {
        FlowBc::Velocity(Box::new(f))
    }

    fn is_outflow(&self) -> bool {
        matches!(self, FlowBc::Outflow)
    }

    /// The velocity Dirichlet data this region imposes at `(x, y, t)`. Zero where the
    /// region is not a velocity-Dirichlet one (`Outflow`), where the value is unused.
    fn dirichlet(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        match self {
            FlowBc::NoSlip | FlowBc::Outflow => (0.0, 0.0),
            FlowBc::Velocity(f) => f(x, y, t),
        }
    }
}

/// Per-tag boundary conditions for an incompressible flow, with a `default` used for
/// any boundary tag not given an explicit entry.
pub struct BoundaryConditions {
    default: FlowBc,
    by_tag: HashMap<u32, FlowBc>,
}

impl BoundaryConditions {
    /// All boundaries no-slip walls unless overridden with [`set`](Self::set).
    pub fn no_slip() -> Self {
        Self { default: FlowBc::NoSlip, by_tag: HashMap::new() }
    }

    /// All boundaries take `default` unless overridden.
    pub fn with_default(default: FlowBc) -> Self {
        Self { default, by_tag: HashMap::new() }
    }

    /// Assign `bc` to boundary `tag` (builder style).
    pub fn set(mut self, tag: u32, bc: FlowBc) -> Self {
        self.by_tag.insert(tag, bc);
        self
    }

    /// The BC for `tag` (falls back to the `default`).
    pub fn get(&self, tag: u32) -> &FlowBc {
        self.by_tag.get(&tag).unwrap_or(&self.default)
    }

    /// Velocity Dirichlet data `(u, v)` at `(x, y, t)` for boundary `tag`.
    pub fn dirichlet(&self, tag: u32, x: f64, y: f64, t: f64) -> (f64, f64) {
        self.get(tag).dirichlet(x, y, t)
    }

    /// Mesh boundary tags that are outflows (velocity-Neumann / pressure-Dirichlet),
    /// sorted.
    pub fn outflow_tags(&self, mesh: &Mesh2d) -> Vec<u32> {
        mesh.boundary_tags()
            .into_iter()
            .filter(|&t| self.get(t).is_outflow())
            .collect()
    }

    /// Whether any boundary tag present in `mesh` is an outflow.
    pub fn has_outflow(&self, mesh: &Mesh2d) -> bool {
        mesh.boundary_tags().iter().any(|&t| self.get(t).is_outflow())
    }
}
