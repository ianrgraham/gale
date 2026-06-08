//! Discontinuous Galerkin core.
//!
//! Build order (see `docs/implicit-solver-strategy.md` §7):
//! reference element → mesh (face-based) → operators → scalar elliptic solve → flow.
//!
//! ## Source layout
//!
//! The module source files are grouped into folders by role to keep the directory
//! navigable — `reference/`, `geometry/`, `mesh/`, `operators/`, `amr/`,
//! `immersed/`, plus `distributed.rs` at the root. The grouping is **physical
//! only** (via `#[path]`): the module tree stays flat, so every item is reached as
//! `dg::<module>::Item` (or the re-exports below) regardless of which folder its
//! file lives in. This keeps all intra-crate `super::` paths and external
//! `gale::dg::*` call sites stable.

// reference element: 1D operators, tensor-product quad/hex reference elements
#[path = "reference/reference.rs"]
pub mod reference;
#[path = "reference/quad.rs"]
pub mod quad;
#[path = "reference/hex.rs"]
pub mod hex;

// geometry: per-element metric terms + face geometry (2D edges, 3D faces)
#[path = "geometry/geometry.rs"]
pub mod geometry;
#[path = "geometry/geometry3d.rs"]
pub mod geometry3d;
#[path = "geometry/face.rs"]
pub mod face;
#[path = "geometry/face3d.rs"]
pub mod face3d;

// mesh: face-based connectivity (2D/3D), the DgMesh trait, nonconforming meshes
#[path = "mesh/mesh.rs"]
pub mod mesh;
#[path = "mesh/mesh3d.rs"]
pub mod mesh3d;
#[path = "mesh/dgmesh.rs"]
pub mod dgmesh;
#[path = "mesh/nonconforming.rs"]
pub mod nonconforming;

// operators: hyperbolic, elliptic (Poisson), Stokes, viscoelastic, filter, mg
#[path = "operators/hyperbolic.rs"]
pub mod hyperbolic;
#[path = "operators/hyperbolic3d.rs"]
pub mod hyperbolic3d;
#[path = "operators/poisson.rs"]
pub mod poisson;
#[path = "operators/poisson3d.rs"]
pub mod poisson3d;
#[path = "operators/stokes.rs"]
pub mod stokes;
#[path = "operators/bc.rs"]
pub mod bc;
#[path = "operators/stokes3d.rs"]
pub mod stokes3d;
#[path = "operators/viscoelastic.rs"]
pub mod viscoelastic;
#[path = "operators/viscoelastic3d.rs"]
pub mod viscoelastic3d;
#[path = "operators/filter.rs"]
pub mod filter;
#[path = "operators/multigrid.rs"]
pub mod multigrid;

// adaptive mesh refinement (2D/3D)
#[path = "amr/amr.rs"]
pub mod amr;
#[path = "amr/amr3d.rs"]
pub mod amr3d;

// immersed boundary method: rigid bodies (2D/3D) + deformable membranes
#[path = "immersed/immersed.rs"]
pub mod immersed;
#[path = "immersed/immersed3d.rs"]
pub mod immersed3d;
#[path = "immersed/membrane.rs"]
pub mod membrane;

// multi-GPU / multi-rank domain decomposition + halo exchange
pub mod distributed;

pub use amr::{
    adapt_scalar, remap_component_flat, remap_scalar, smoothness_per_cell, RefineQuad,
    SmoothnessIndicator,
};
pub use amr3d::{RefineHex, SmoothnessIndicator3d};
pub use bc::{BoundaryConditions, FlowBc};
pub use dgmesh::DgMesh;
pub use distributed::{distributed_advection_rhs, halo_exchange, partition_blocks};
pub use face::{quad_faces, Edge, FaceData};
pub use filter::ModalFilter;
pub use face3d::{hex_faces, Face, HexFaceData};
pub use geometry::QuadGeometry;
pub use geometry3d::HexGeometry;
pub use hex::Reference3dHex;
pub use hyperbolic3d::{ConservationLaw3d, Hyperbolic3d, LinearAdvection3d};
pub use mesh3d::{HexElement, Mesh3d, Neighbor3};
pub use immersed::{Disk, ImmersedSolid, RigidBody, Shape, VolumePenalization};
pub use immersed3d::{Ellipsoid, ImmersedSolid3d, Sphere, VolumePenalization3d};
pub use hyperbolic::{
    Burgers, ConservationLaw, Euler, Hyperbolic, IncompressibleConvection, LinearAdvection,
    VolumeForm,
};
pub use mesh::{Element, Mesh2d, Neighbor};
pub use multigrid::PMultigrid;
pub use nonconforming::{NcAdvection, NcMesh, NcNeighbor};
pub use poisson::Poisson;
pub use poisson3d::Poisson3d;
pub use stokes3d::Stokes3d;
pub use quad::Reference2dQuad;
pub use reference::Reference1d;
pub use stokes::{ConvectionScheme, Stokes};
pub use viscoelastic::{
    log_conformation, upwind_advection_lift, ConformationInflow, ConstitutiveModel,
    LogConfOldroydB, OldroydB, ViscoelasticFlow,
};
pub use viscoelastic3d::{sym_apply3, sym_eig3, LogConfOldroydB3d, OldroydB3d};
