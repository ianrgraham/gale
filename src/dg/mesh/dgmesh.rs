//! The `DgMesh` trait — the minimal mesh interface the dimension-generic
//! framework needs (`docs/3d-strategy.md` §4). Implemented by both
//! [`Mesh2d`](super::mesh::Mesh2d) and [`Mesh3d`](super::mesh3d::Mesh3d), it lets
//! `State<M>` / `Simulation<M>` be generic over the spatial dimension while the
//! dimension-specific operators are built from the concrete mesh inside the
//! integrators' base-rhs closures.
//!
//! It exposes only what the *generic* framework code touches — element count,
//! nodes-per-element (hence total DOF), and the physical volume. The
//! `Device`/`DomainDecomposition` layer already works on the element count alone,
//! so it composes for free.

use super::mesh::Mesh2d;
use super::mesh3d::Mesh3d;

/// A DG mesh viewed dimension-agnostically.
pub trait DgMesh {
    /// Number of elements.
    fn n_elements(&self) -> usize;
    /// Nodes per element (`(p+1)^dim`).
    fn n_nodes(&self) -> usize;
    /// Total scalar DOF count (`n_elements · n_nodes`), the inner length of a field
    /// component.
    fn ndof(&self) -> usize {
        self.n_elements() * self.n_nodes()
    }
    /// Total physical volume/area (∑ `detJ·w`).
    fn measure(&self) -> f64;
}

impl DgMesh for Mesh2d {
    fn n_elements(&self) -> usize {
        self.elements.len()
    }
    fn n_nodes(&self) -> usize {
        self.refq.n_nodes()
    }
    fn measure(&self) -> f64 {
        self.area()
    }
}

impl DgMesh for Mesh3d {
    fn n_elements(&self) -> usize {
        self.elements.len()
    }
    fn n_nodes(&self) -> usize {
        self.refh.n_nodes()
    }
    fn measure(&self) -> f64 {
        self.volume()
    }
}
