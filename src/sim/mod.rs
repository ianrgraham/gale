//! gale's user-facing simulation framework.
//!
//! This module implements the HOOMD-blue × Trixi.jl architecture specified in
//! `docs/api-design.md`: a central `Simulation` owning a `State`, `Operations`,
//! and a `Device`; a `Semidiscretization` producing an rhs from additive `Term`s;
//! one `Integrator` trait spanning explicit method-of-lines and structured
//! schemes; and `Compute`/`Updater`/`Writer`/`Tuner` operations gated by
//! `Trigger`s.
//!
//! It is being grown by the incremental migration path in api-design §6 — wrapping
//! the validated bespoke kernels rather than rewriting them. Step 1 (this commit)
//! establishes the field-based [`State`] core.

pub mod amr;
pub mod device;
pub mod dynamics;
pub mod field;
pub mod ibm;
pub mod integrate;
pub mod simulation;
pub mod stagehook;
pub mod state;
pub mod term;

pub use amr::{AmrUpdater, AmrUpdater3d};
pub use device::{Device, DomainDecomposition, Partition};
pub use dynamics::{
    BaseRhs, BodyForce, BodyForce3d, DualSplitting, DualSplitting3d, FieldVec, FnStateTerm, Mol,
    NoStateHook, SspRk3State, StateIntegrator, StateSemi, StateSemidiscretization, StateStageHook,
    StateTerm, ViscoModel, ViscoelasticDualSplitting,
};
pub use field::{Field, FieldId, FieldSet};
pub use ibm::{
    BodyHandle, MovingPenalizationHook, MultiMovingPenalizationHook, Penalization3dDrag,
    Penalization3dHook, PenalizationDrag, PenalizationHook, SuspensionHandle,
};
pub use integrate::{ClosureSemi, Integrator, NoHook, Semi, SspRk3, StageHook};
pub use simulation::{
    Always, Compute, OnStep, Operations, Periodic, Simulation, Trigger, Triggered, Updater, Writer,
};
pub use stagehook::FilterHook;
pub use state::{State, Time};
pub use term::{Semidiscretization, SourceTerm, Term};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::mesh::Mesh2d;

    fn mesh() -> Mesh2d {
        Mesh2d::rectangular(3, 4, 2, [0.0, 2.0], [0.0, 1.0])
    }

    #[test]
    fn ndof_matches_mesh() {
        let m = mesh();
        let expected = m.n_elements() * m.refq.n_nodes();
        let s = State::new(m);
        assert_eq!(s.ndof(), expected);
        assert!(s.fields.is_empty());
        assert_eq!(s.time, Time::default());
    }

    #[test]
    fn add_field_allocates_zeros_in_operator_layout() {
        let m = mesh();
        let ndof = m.n_elements() * m.refq.n_nodes();
        let mut s = State::new(m);
        let u = s.add_field("velocity", 2);
        let p = s.add_field("pressure", 1);

        // Handles resolve back from names.
        assert_eq!(s.fields.id("velocity"), Some(u));
        assert_eq!(s.fields.id("pressure"), Some(p));
        assert_eq!(s.fields.id("missing"), None);

        // Layout is exactly [n_comp][ndof], the shape the operators consume.
        let vel = s.field("velocity");
        assert_eq!(vel.n_comp, 2);
        assert_eq!(vel.ndof, ndof);
        assert_eq!(vel.components().len(), 2);
        assert!(vel.components().iter().all(|c| c.len() == ndof));
        assert!(vel.components().iter().flatten().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "duplicate field name")]
    fn duplicate_field_panics() {
        let mut s = State::new(mesh());
        s.add_field("u", 1);
        s.add_field("u", 1);
    }

    #[test]
    fn init_from_closures_samples_physical_coords() {
        let m = mesh();
        let mut s = State::new(m);
        // f0 = x, f1 = y so we can check sampling at the nodes directly.
        s.add_field_from("xy", &[|x: f64, _y: f64| x, |_x: f64, y: f64| y]);

        let nn = s.mesh.refq.n_nodes();
        let f = s.field("xy");
        for (e, el) in s.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                assert_eq!(f.component(0)[e * nn + k], el.geom.x[k]);
                assert_eq!(f.component(1)[e * nn + k], el.geom.y[k]);
            }
        }
    }

    #[test]
    fn assign_roundtrips_through_vecvec_adapter() {
        // The bridge to the validated kernels: an operator produces Vec<Vec<f64>>,
        // we assign it into a field, and read back the identical layout.
        let mut s = State::new(mesh());
        let id = s.add_field("c", 3);
        let ndof = s.ndof();

        let produced: Vec<Vec<f64>> =
            (0..3).map(|v| (0..ndof).map(|i| (v * 100 + i) as f64).collect()).collect();

        s.fields.by_id_mut(id).assign(&produced);

        let back = s.fields.by_id(id).components();
        assert_eq!(back, produced.as_slice());
    }

    #[test]
    fn from_components_rejects_ragged() {
        let ragged = vec![vec![0.0; 4], vec![0.0; 3]];
        let r = std::panic::catch_unwind(|| Field::from_components("bad", ragged));
        assert!(r.is_err());
    }
}
