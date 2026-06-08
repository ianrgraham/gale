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
use super::mesh3d::Mesh3d;
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
    /// Symmetry plane / free-slip wall: no penetration (`u·n = 0`, Dirichlet on the
    /// **normal** component) + traction-free tangentially (Neumann on the **tangential**
    /// components); pressure Neumann. Assumes an axis-aligned boundary (the per-tag
    /// normal is read from a representative face).
    Symmetry,
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
    /// region is not a velocity-Dirichlet one (`Outflow`/`Symmetry`), where the value is
    /// either unused or the no-penetration `0`.
    fn dirichlet(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        match self {
            FlowBc::NoSlip | FlowBc::Outflow | FlowBc::Symmetry => (0.0, 0.0),
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

    /// Neumann tags for the **pressure**-Poisson: everything except outflow (where the
    /// pressure is pinned Dirichlet `p=0`). Walls / inflow / symmetry are all Neumann.
    pub fn pressure_neumann_tags(&self, mesh: &Mesh2d) -> Vec<u32> {
        mesh.boundary_tags()
            .into_iter()
            .filter(|&t| !self.get(t).is_outflow())
            .collect()
    }

    /// Neumann tags for the **velocity** Helmholtz solve of component `comp` (0=x, 1=y,
    /// 2=z): outflow tags (always natural), plus symmetry tags where this component is
    /// *tangential* — i.e. the symmetry face's normal is along a different axis. At a
    /// symmetry face the normal component stays Dirichlet (`u·n = 0`); the others are
    /// Neumann. The per-tag normal axis is read from a representative face via
    /// [`Mesh2d::boundary_tag_normal_axis`].
    pub fn velocity_neumann_tags(&self, mesh: &Mesh2d, comp: usize) -> Vec<u32> {
        mesh.boundary_tags()
            .into_iter()
            .filter(|&t| match self.get(t) {
                FlowBc::Outflow => true,
                FlowBc::Symmetry => mesh.boundary_tag_normal_axis(t) != Some(comp),
                _ => false,
            })
            .collect()
    }
}

// ===== 3D =======================================================================

/// The boundary condition imposed on one boundary region of a 3D incompressible flow —
/// the 3D analogue of [`FlowBc`]. Faces are tagged by local face index
/// (`Face::ALL` order: 0=Bottom, 1=Top, 2=South, 3=North, 4=West, 5=East).
pub enum FlowBc3d {
    /// No-slip wall: `u = 0`.
    NoSlip,
    /// Prescribed velocity `(u, v, w) = u_s(x, y, z, t)` — inflow / moving wall.
    Velocity(Box<dyn Fn(f64, f64, f64, f64) -> (f64, f64, f64)>),
    /// Traction-free outflow: natural (Neumann) velocity, pressure pinned to `0`.
    Outflow,
    /// Symmetry plane / free-slip wall (`u·n = 0`, tangential traction-free); pressure
    /// Neumann. Assumes an axis-aligned boundary.
    Symmetry,
}

impl FlowBc3d {
    /// A prescribed-velocity BC from a closure `(x, y, z, t) -> (u, v, w)`.
    pub fn velocity(f: impl Fn(f64, f64, f64, f64) -> (f64, f64, f64) + 'static) -> Self {
        FlowBc3d::Velocity(Box::new(f))
    }

    fn is_outflow(&self) -> bool {
        matches!(self, FlowBc3d::Outflow)
    }

    fn dirichlet(&self, x: f64, y: f64, z: f64, t: f64) -> (f64, f64, f64) {
        match self {
            FlowBc3d::NoSlip | FlowBc3d::Outflow | FlowBc3d::Symmetry => (0.0, 0.0, 0.0),
            FlowBc3d::Velocity(f) => f(x, y, z, t),
        }
    }
}

/// Per-tag 3D boundary conditions — the 3D analogue of [`BoundaryConditions`].
pub struct BoundaryConditions3d {
    default: FlowBc3d,
    by_tag: HashMap<u32, FlowBc3d>,
}

impl BoundaryConditions3d {
    /// All boundaries no-slip walls unless overridden.
    pub fn no_slip() -> Self {
        Self { default: FlowBc3d::NoSlip, by_tag: HashMap::new() }
    }

    /// All boundaries take `default` unless overridden.
    pub fn with_default(default: FlowBc3d) -> Self {
        Self { default, by_tag: HashMap::new() }
    }

    /// Assign `bc` to boundary `tag` (builder style).
    pub fn set(mut self, tag: u32, bc: FlowBc3d) -> Self {
        self.by_tag.insert(tag, bc);
        self
    }

    /// The BC for `tag` (falls back to the `default`).
    pub fn get(&self, tag: u32) -> &FlowBc3d {
        self.by_tag.get(&tag).unwrap_or(&self.default)
    }

    /// Velocity Dirichlet data `(u, v, w)` at `(x, y, z, t)` for boundary `tag`.
    pub fn dirichlet(&self, tag: u32, x: f64, y: f64, z: f64, t: f64) -> (f64, f64, f64) {
        self.get(tag).dirichlet(x, y, z, t)
    }

    /// Whether any boundary tag present in `mesh` is an outflow.
    pub fn has_outflow(&self, mesh: &Mesh3d) -> bool {
        mesh.boundary_tags().iter().any(|&t| self.get(t).is_outflow())
    }

    /// Pressure-Poisson Neumann tags: everything except outflow (pinned `p=0`).
    pub fn pressure_neumann_tags(&self, mesh: &Mesh3d) -> Vec<u32> {
        mesh.boundary_tags()
            .into_iter()
            .filter(|&t| !self.get(t).is_outflow())
            .collect()
    }

    /// Velocity Helmholtz Neumann tags for component `comp` (0=x, 1=y, 2=z): outflow
    /// tags plus symmetry tags where `comp` is tangential (the face normal is along a
    /// different axis). At a symmetry face the normal component stays Dirichlet.
    pub fn velocity_neumann_tags(&self, mesh: &Mesh3d, comp: usize) -> Vec<u32> {
        mesh.boundary_tags()
            .into_iter()
            .filter(|&t| match self.get(t) {
                FlowBc3d::Outflow => true,
                FlowBc3d::Symmetry => mesh.boundary_tag_normal_axis(t) != Some(comp),
                _ => false,
            })
            .collect()
    }
}
