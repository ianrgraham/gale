//! The simulation [`State`] — the single source of truth.
//!
//! Following the HOOMD-blue model (`docs/api-design.md` §3.1), one `State` owns
//! everything an operation may read or mutate: the mesh/topology, the named DG
//! [`FieldSet`], and the simulation [`Time`]. gale's State is *field-based* (DG
//! coefficients per element), not particle-based — the orchestration model
//! transfers from HOOMD, the data model is gale's own.
//!
//! `State<M>` is generic over the mesh dimension via [`DgMesh`] (`docs/3d-strategy.md`
//! §4); the default `M = Mesh2d` keeps all 2D code source-compatible. Initial-
//! condition helpers that need physical coordinates are provided per dimension
//! (`add_field_from` takes a 2-arg closure on `State<Mesh2d>`, a 3-arg closure on
//! `State<Mesh3d>`).

use super::field::{Field, FieldId, FieldSet};
use crate::dg::dgmesh::DgMesh;
use crate::dg::mesh::Mesh2d;
use crate::dg::mesh3d::Mesh3d;

/// Simulation clock. `step` is read-only to every operation except the integrator,
/// which is the sole authority that advances time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Time {
    pub t: f64,
    pub step: u64,
}

/// The single mutable source of truth for a simulation, generic over mesh dimension.
#[derive(Clone, Debug)]
pub struct State<M: DgMesh = Mesh2d> {
    pub mesh: M,
    pub fields: FieldSet,
    pub time: Time,
}

impl<M: DgMesh> State<M> {
    /// Create an empty-field state over `mesh`.
    pub fn new(mesh: M) -> Self {
        Self { mesh, fields: FieldSet::new(), time: Time::default() }
    }

    /// Degrees of freedom per scalar component: `n_elements · n_nodes`. The inner
    /// length of every field component.
    #[inline]
    pub fn ndof(&self) -> usize {
        self.mesh.ndof()
    }

    /// Register a new zeroed field with `n_comp` components and return its handle.
    pub fn add_field(&mut self, name: impl Into<String>, n_comp: usize) -> FieldId {
        let ndof = self.ndof();
        self.fields.add(name, n_comp, ndof)
    }

    /// Borrow a field by name (panics if absent — use [`FieldSet::get`] for the
    /// fallible form).
    pub fn field(&self, name: &str) -> &Field {
        self.fields.get(name).unwrap_or_else(|| panic!("no field `{name}`"))
    }

    /// Mutably borrow a field by name (panics if absent).
    pub fn field_mut(&mut self, name: &str) -> &mut Field {
        self.fields.get_mut(name).unwrap_or_else(|| panic!("no field `{name}`"))
    }
}

impl State<Mesh2d> {
    /// Register a field initialized component-wise from closures of physical
    /// coordinates `(x, y)`. One closure per component.
    pub fn add_field_from<F>(&mut self, name: impl Into<String>, closures: &[F]) -> FieldId
    where
        F: Fn(f64, f64) -> f64,
    {
        let ndof = self.ndof();
        let mut comps = vec![vec![0.0; ndof]; closures.len()];
        let nn = self.mesh.refq.n_nodes();
        for (e, el) in self.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                for (c, f) in closures.iter().enumerate() {
                    comps[c][e * nn + k] = f(x, y);
                }
            }
        }
        self.fields.insert(Field::from_components(name, comps))
    }
}

impl State<Mesh3d> {
    /// Register a field initialized component-wise from closures of physical
    /// coordinates `(x, y, z)`. One closure per component.
    pub fn add_field_from<F>(&mut self, name: impl Into<String>, closures: &[F]) -> FieldId
    where
        F: Fn(f64, f64, f64) -> f64,
    {
        let ndof = self.ndof();
        let mut comps = vec![vec![0.0; ndof]; closures.len()];
        let nn = self.mesh.refh.n_nodes();
        for (e, el) in self.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y, z) = (el.geom.x[k], el.geom.y[k], el.geom.z[k]);
                for (c, f) in closures.iter().enumerate() {
                    comps[c][e * nn + k] = f(x, y, z);
                }
            }
        }
        self.fields.insert(Field::from_components(name, comps))
    }
}
