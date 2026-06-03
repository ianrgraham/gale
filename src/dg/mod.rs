//! Discontinuous Galerkin core.
//!
//! Build order (see `docs/implicit-solver-strategy.md` §7):
//! reference element → mesh (face-based) → operators → scalar elliptic solve → flow.

pub mod amr;
pub mod dgmesh;
pub mod distributed;
pub mod face;
pub mod face3d;
pub mod filter;
pub mod geometry;
pub mod geometry3d;
pub mod hex;
pub mod hyperbolic;
pub mod hyperbolic3d;
pub mod immersed;
pub mod membrane;
pub mod mesh;
pub mod mesh3d;
pub mod multigrid;
pub mod nonconforming;
pub mod poisson;
pub mod poisson3d;
pub mod stokes3d;
pub mod viscoelastic3d;
pub mod quad;
pub mod reference;
pub mod stokes;
pub mod viscoelastic;

pub use amr::{
    adapt_scalar, remap_component_flat, remap_scalar, smoothness_per_cell, RefineQuad,
    SmoothnessIndicator,
};
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
pub use viscoelastic::{ConstitutiveModel, LogConfOldroydB, OldroydB, ViscoelasticFlow};
pub use viscoelastic3d::{sym_apply3, sym_eig3, LogConfOldroydB3d, OldroydB3d};
