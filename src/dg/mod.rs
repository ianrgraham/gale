//! Discontinuous Galerkin core.
//!
//! Build order (see `docs/implicit-solver-strategy.md` §7):
//! reference element → mesh (face-based) → operators → scalar elliptic solve → flow.

pub mod amr;
pub mod face;
pub mod filter;
pub mod geometry;
pub mod hyperbolic;
pub mod immersed;
pub mod membrane;
pub mod mesh;
pub mod multigrid;
pub mod nonconforming;
pub mod poisson;
pub mod quad;
pub mod reference;
pub mod stokes;
pub mod viscoelastic;

pub use amr::{adapt_scalar, remap_scalar, RefineQuad, SmoothnessIndicator};
pub use face::{quad_faces, Edge, FaceData};
pub use filter::ModalFilter;
pub use geometry::QuadGeometry;
pub use immersed::{Disk, ImmersedSolid, RigidBody, Shape, VolumePenalization};
pub use hyperbolic::{
    Burgers, ConservationLaw, Euler, Hyperbolic, IncompressibleConvection, LinearAdvection,
    VolumeForm,
};
pub use mesh::{Element, Mesh2d, Neighbor};
pub use multigrid::PMultigrid;
pub use nonconforming::{NcAdvection, NcMesh, NcNeighbor};
pub use poisson::Poisson;
pub use quad::Reference2dQuad;
pub use reference::Reference1d;
pub use stokes::{ConvectionScheme, Stokes};
pub use viscoelastic::{ConstitutiveModel, LogConfOldroydB, OldroydB, ViscoelasticFlow};
