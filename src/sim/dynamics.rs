//! State-level dynamics: the multi-field semidiscretization seam.
//!
//! The vector-level [`Semi`](super::integrate::Semi) advances one equation's
//! `[n_vars][ndof]` state. The State-level seam here advances the **evolving
//! fields of a [`State`]** together, which is what cross-field coupling needs: a
//! [`StateTerm`] can read one field (e.g. the conformation tensor) and accumulate
//! into another field's derivative (e.g. momentum) — the polymer-stress coupling
//! the four-homes rule (`docs/api-design.md` §3.4) assigns to a `Term`.
//!
//! Borrow strategy: nothing stores a borrow of the mesh. Base operators are built
//! transiently from `state.mesh` *inside* [`StateSemi::rhs`] (cheap — they hold
//! refs + config), so a [`Simulation`](super::simulation::Simulation) can own the
//! `State`, the semidiscretization, and the integrator without self-reference.

use super::field::FieldId;
use super::state::State;
use crate::dg::dgmesh::DgMesh;
use crate::dg::mesh::Mesh2d;
use crate::dg::mesh3d::Mesh3d;
use crate::dg::stokes::Stokes;
use crate::dg::stokes3d::Stokes3d;
use crate::dg::viscoelastic::{ConstitutiveModel, LogConfOldroydB, OldroydB, ViscoelasticFlow};
use std::collections::BTreeMap;

/// A bundle of per-field values keyed by [`FieldId`] — used for both the evolving
/// field values and their time derivatives during a step. Components are stored in
/// the `[n_comp][ndof]` layout the operators use; ids are visited in sorted order
/// so linear combinations are deterministic.
#[derive(Clone, Debug, Default)]
pub struct FieldVec {
    data: BTreeMap<usize, Vec<Vec<f64>>>,
}

impl FieldVec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy the listed fields' components out of `state`.
    pub fn extract<M: DgMesh>(state: &State<M>, ids: &[FieldId]) -> Self {
        let mut data = BTreeMap::new();
        for &id in ids {
            data.insert(id.0, state.fields.by_id(id).components().to_vec());
        }
        Self { data }
    }

    /// A zeroed clone with the same fields/shapes as `self`.
    pub fn zeros_like(&self) -> Self {
        let mut data = BTreeMap::new();
        for (&k, comps) in &self.data {
            data.insert(k, comps.iter().map(|c| vec![0.0; c.len()]).collect());
        }
        Self { data }
    }

    /// Overwrite field `id`'s value (used by base operators).
    pub fn set(&mut self, id: FieldId, comps: Vec<Vec<f64>>) {
        self.data.insert(id.0, comps);
    }

    /// Borrow field `id`'s components.
    pub fn get(&self, id: FieldId) -> &[Vec<f64>] {
        &self.data[&id.0]
    }

    /// Mutably borrow field `id`'s components (used by terms to accumulate).
    /// Inserts a zeroed entry if absent — but the entry shape can only be known
    /// once present, so callers accumulate into existing base entries.
    pub fn get_mut(&mut self, id: FieldId) -> &mut Vec<Vec<f64>> {
        self.data.get_mut(&id.0).expect("field not present in FieldVec; set its base first")
    }

    /// Write all held fields back into `state`.
    pub fn scatter_into<M: DgMesh>(&self, state: &mut State<M>) {
        for (&k, comps) in &self.data {
            state.fields.by_id_mut(FieldId(k)).assign(comps);
        }
    }

    /// Linear combination `Σ cᵢ · vᵢ`, accumulated in (field, component, dof) order
    /// so single-field results match the vector-level integrator bit-for-bit. All
    /// inputs must share the same field set and shapes.
    pub fn comb(coeffs: &[(f64, &FieldVec)]) -> FieldVec {
        assert!(!coeffs.is_empty());
        let mut out = coeffs[0].1.zeros_like();
        for &(c, v) in coeffs {
            for (&k, comps) in &v.data {
                let dst = out.data.get_mut(&k).expect("mismatched field sets in comb");
                for (od, sd) in dst.iter_mut().zip(comps) {
                    for (o, s) in od.iter_mut().zip(sd) {
                        *o += c * *s;
                    }
                }
            }
        }
        out
    }
}

/// An additive contribution to the State-level rhs. Unlike the vector-level
/// [`Term`](super::term::Term), a `StateTerm` sees the **whole** `State`, so it can
/// read any field and accumulate into any (evolving) field's derivative.
pub trait StateTerm<M: DgMesh = Mesh2d> {
    fn accumulate(&self, state: &State<M>, t: f64, dot: &mut FieldVec);
}

/// Adapter wrapping a closure as a [`StateTerm`].
pub struct FnStateTerm<F>(pub F);

impl<M: DgMesh, F> StateTerm<M> for FnStateTerm<F>
where
    F: Fn(&State<M>, f64, &mut FieldVec),
{
    fn accumulate(&self, state: &State<M>, t: f64, dot: &mut FieldVec) {
        (self.0)(state, t, dot)
    }
}

/// A State-level semi-discrete operator: produces `∂ₜ(evolving fields)`.
pub trait StateSemi<M: DgMesh = Mesh2d> {
    /// The fields this operator evolves in time.
    fn evolving(&self) -> Vec<FieldId>;
    /// Fill `dot` with the time derivative of every evolving field, given the full
    /// `state` at time `t`. `dot` arrives zeroed with the right shapes.
    fn rhs(&self, state: &State<M>, t: f64, dot: &mut FieldVec);
}

/// Per-field base rhs: given the full state and time, returns that field's base
/// (un-coupled) time derivative in `[n_comp][ndof]` layout. Built to construct its
/// operator transiently from `state.mesh`, avoiding any stored mesh borrow.
pub type BaseRhs<M = Mesh2d> = Box<dyn Fn(&State<M>, f64) -> Vec<Vec<f64>>>;

/// The concrete State-level semidiscretization: one base rhs per evolving field
/// plus a list of additive cross-field [`StateTerm`]s.
pub struct StateSemidiscretization<M: DgMesh = Mesh2d> {
    bases: Vec<(FieldId, BaseRhs<M>)>,
    terms: Vec<Box<dyn StateTerm<M>>>,
}

impl<M: DgMesh> StateSemidiscretization<M> {
    pub fn new() -> Self {
        Self { bases: Vec::new(), terms: Vec::new() }
    }

    /// Register the base rhs for an evolving field.
    pub fn field(mut self, id: FieldId, base: BaseRhs<M>) -> Self {
        self.bases.push((id, base));
        self
    }

    /// Append an additive cross-field term.
    pub fn with_term(mut self, term: impl StateTerm<M> + 'static) -> Self {
        self.terms.push(Box::new(term));
        self
    }
}

impl<M: DgMesh> Default for StateSemidiscretization<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: DgMesh> StateSemi<M> for StateSemidiscretization<M> {
    fn evolving(&self) -> Vec<FieldId> {
        self.bases.iter().map(|(id, _)| *id).collect()
    }

    fn rhs(&self, state: &State<M>, t: f64, dot: &mut FieldVec) {
        // Base rhs for each evolving field (overwrite), then additive terms (+=).
        for (id, base) in &self.bases {
            dot.set(*id, base(state, t));
        }
        for term in &self.terms {
            term.accumulate(state, t, dot);
        }
    }
}

/// A per-stage `state ← g(state)` operation at the State level (limiters, SVV
/// filter, implicit penalization projection). `stage` is the just-completed RK
/// stage (0-based).
pub trait StateStageHook<M: DgMesh = Mesh2d> {
    fn after_stage(&self, state: &mut State<M>, stage: usize);
}

/// The no-op State-level stage hook.
pub struct NoStateHook;

impl<M: DgMesh> StateStageHook<M> for NoStateHook {
    fn after_stage(&self, _state: &mut State<M>, _stage: usize) {}
}

/// A State-level time integrator: advances the evolving fields of `state` by one
/// step, applying `hook` between stages and updating `state.time`.
///
/// The integrator **owns its dynamics**. Method-of-lines integrators ([`Mol`])
/// hold a [`StateSemi`] and consume its additive rhs; structured schemes
/// (`DualSplitting`, IMEX) hold their own configuration and orchestrate their
/// stages internally. Both honor this one trait, so the
/// [`Simulation`](super::simulation::Simulation) drives either uniformly — the
/// "one Integrator trait, multiple families" contract of `docs/api-design.md`
/// §3.2.
pub trait StateIntegrator<M: DgMesh = Mesh2d> {
    fn dt(&self) -> f64;
    fn step(&self, state: &mut State<M>, hook: &dyn StateStageHook<M>);
}

/// Explicit SSP-RK3 (Shu–Osher) *scheme* over a multi-field [`State`]. This is the
/// stage logic only; bind it to a [`StateSemi`] with [`Mol`] to obtain a
/// [`StateIntegrator`]. The combination arithmetic mirrors the vector-level
/// [`SspRk3`](super::integrate::SspRk3) so a single-field problem advances
/// bit-for-bit identically.
#[derive(Clone, Copy, Debug)]
pub struct SspRk3State {
    pub dt: f64,
}

impl SspRk3State {
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Advance `state` by one SSP-RK3 step against `semi`, applying `hook` after
    /// each stage.
    pub fn advance<M: DgMesh>(
        &self,
        semi: &dyn StateSemi<M>,
        state: &mut State<M>,
        hook: &dyn StateStageHook<M>,
    ) {
        let t = state.time.t;
        let dt = self.dt;
        let ev = semi.evolving();
        let u0 = FieldVec::extract(state, &ev);

        let eval = |state: &State<M>, t: f64| -> FieldVec {
            let mut dot = u0.zeros_like();
            semi.rhs(state, t, &mut dot);
            dot
        };

        let k1 = eval(state, t);
        let u1 = FieldVec::comb(&[(1.0, &u0), (dt, &k1)]);
        u1.scatter_into(state);
        hook.after_stage(state, 0);

        let u1 = FieldVec::extract(state, &ev);
        let k2 = eval(state, t + dt);
        let u2 = FieldVec::comb(&[(0.75, &u0), (0.25, &u1), (0.25 * dt, &k2)]);
        u2.scatter_into(state);
        hook.after_stage(state, 1);

        let u2 = FieldVec::extract(state, &ev);
        let k3 = eval(state, t + 0.5 * dt);
        let un = FieldVec::comb(&[(1.0 / 3.0, &u0), (2.0 / 3.0, &u2), (2.0 / 3.0 * dt, &k3)]);
        un.scatter_into(state);
        hook.after_stage(state, 2);

        state.time.t = t + dt;
        state.time.step += 1;
    }
}

/// Method-of-lines integrator: binds a [`StateSemi`] (the additive spatial
/// operator) to an explicit scheme (here [`SspRk3State`]). Owns the semi, so the
/// [`Simulation`](super::simulation::Simulation) holds a single self-contained
/// [`StateIntegrator`].
pub struct Mol<S> {
    pub semi: S,
    pub scheme: SspRk3State,
}

impl<S> Mol<S> {
    pub fn new(semi: S, scheme: SspRk3State) -> Self {
        Self { semi, scheme }
    }
}

impl<M: DgMesh, S: StateSemi<M>> StateIntegrator<M> for Mol<S> {
    fn dt(&self) -> f64 {
        self.scheme.dt()
    }
    fn step(&self, state: &mut State<M>, hook: &dyn StateStageHook<M>) {
        self.scheme.advance(&self.semi, state, hook);
    }
}

/// A nodal body force `(force_x, force_y)` over all DOFs, computed from the state
/// and time — e.g. an external drive plus the polymer-stress divergence `∇·τ_p`
/// read from the conformation field.
pub type BodyForce<M = Mesh2d> = Box<dyn Fn(&State<M>, f64) -> (Vec<f64>, Vec<f64>)>;

/// **Structured** incompressible Navier–Stokes integrator: BDF1 dual-splitting
/// (explicit convection + body force → pressure-Poisson projection → implicit
/// viscous Helmholtz solve). Unlike [`Mol`], it has no additive semidiscretization
/// — it orchestrates its own stages, drawing on the validated [`Stokes`] operator
/// (constructed transiently from `state.mesh`, so nothing stores a mesh borrow).
///
/// This is the second [`StateIntegrator`] family, demonstrating the
/// "one Integrator trait, multiple families" contract (`docs/api-design.md` §3.2):
/// explicit method-of-lines ([`Mol`]) and structured projection coexist behind the
/// same trait. The post-step [`StateStageHook`] is the home for the implicit
/// volume-penalization (IBM) projection.
pub struct DualSplitting {
    pub dt: f64,
    /// Solvent (kinematic) viscosity `η_s`.
    pub nu: f64,
    /// SIPG penalty parameter.
    pub alpha: f64,
    velocity: FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64) -> f64>,
    body_force: BodyForce,
}

impl DualSplitting {
    /// New incompressible NS integrator advancing the 2-component `velocity` field,
    /// with zero-velocity walls and no body force by default.
    pub fn new(velocity: FieldId, dt: f64, nu: f64, alpha: f64) -> Self {
        Self {
            dt,
            nu,
            alpha,
            velocity,
            bc_u: Box::new(|_, _, _| 0.0),
            bc_v: Box::new(|_, _, _| 0.0),
            body_force: Box::new(|s: &State, _t: f64| {
                let n = s.ndof();
                (vec![0.0; n], vec![0.0; n])
            }),
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v)` as functions
    /// of `(x, y, t)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self
    }

    /// Set the nodal body force, computed from the full state and the new time
    /// level — e.g. `∇·τ_p` plus an external drive.
    pub fn body_force(
        mut self,
        f: impl Fn(&State, f64) -> (Vec<f64>, Vec<f64>) + 'static,
    ) -> Self {
        self.body_force = Box::new(f);
        self
    }
}

impl StateIntegrator for DualSplitting {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut State, hook: &dyn StateStageHook) {
        let t_new = state.time.t + self.dt;
        let stokes = Stokes::new(&state.mesh, self.alpha, self.nu, self.dt);
        let (ux, uy) = {
            let v = state.fields.by_id(self.velocity);
            (v.component(0).to_vec(), v.component(1).to_vec())
        };
        let (bx, by) = (self.body_force)(state, t_new);
        let (nux, nuy) =
            stokes.step_ns_forced(&ux, &uy, t_new, &self.bc_u, &self.bc_v, &bx, &by);
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
        }
        // Per-stage hook: home for the implicit volume-penalization projection.
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}

/// Which constitutive model the viscoelastic integrator uses. The model borrows
/// the mesh, so it cannot be stored in a `'static` boxed integrator; instead the
/// integrator stores this selector + parameters and builds the concrete model
/// transiently from `state.mesh` each step. (Extending to user-defined models
/// would mean a model-builder hook; the two validated models are covered here.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViscoModel {
    /// Direct Oldroyd-B conformation transport.
    OldroydB,
    /// Log-conformation Oldroyd-B (Fattal–Kupferman) for high-Wi robustness.
    LogConf,
}

/// **Coupled** viscoelastic integrator: one tightly-coupled advance of velocity
/// **and** the conformation field, in the dual-splitting split order — velocity
/// updates using `∇·τ_p` from the *old* conformation, then the conformation
/// advances with the *new* velocity. Wraps the validated
/// [`ViscoelasticFlow::step`], so it reproduces that operator bit-for-bit.
///
/// This is the third [`StateIntegrator`] family. The coupling is owned by the
/// integrator (not expressed as a separate composable op) precisely because the
/// split order does not fit the `updaters → integrator → writers` schedule — the
/// faithful, validated mapping chosen in `docs/api-design.md` (VE coupling fork).
pub struct ViscoelasticDualSplitting {
    pub dt: f64,
    pub eta_s: f64,
    pub eta_p: f64,
    pub lambda: f64,
    pub alpha: f64,
    pub model: ViscoModel,
    velocity: FieldId,
    conformation: FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64) -> f64>,
    fx: Box<dyn Fn(f64, f64, f64) -> f64>,
    fy: Box<dyn Fn(f64, f64, f64) -> f64>,
}

impl ViscoelasticDualSplitting {
    /// New coupled integrator over the 2-component `velocity` and 3-component
    /// `conformation` (symmetric tensor `(xx, xy, yy)`) fields. Zero walls and no
    /// external drive by default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        velocity: FieldId,
        conformation: FieldId,
        dt: f64,
        eta_s: f64,
        eta_p: f64,
        lambda: f64,
        alpha: f64,
        model: ViscoModel,
    ) -> Self {
        Self {
            dt,
            eta_s,
            eta_p,
            lambda,
            alpha,
            model,
            velocity,
            conformation,
            bc_u: Box::new(|_, _, _| 0.0),
            bc_v: Box::new(|_, _, _| 0.0),
            fx: Box::new(|_, _, _| 0.0),
            fy: Box::new(|_, _, _| 0.0),
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self
    }

    /// Set the external body force / drive `(fx, fy)` as functions of `(x, y, t)`.
    pub fn drive(
        mut self,
        fx: impl Fn(f64, f64, f64) -> f64 + 'static,
        fy: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.fx = Box::new(fx);
        self.fy = Box::new(fy);
        self
    }

    /// The model's equilibrium conformation `[xx, xy, yy]` over `mesh` (identity
    /// for direct Oldroyd-B; `Ψ = log I = 0` for log-conformation). Use to
    /// initialize the conformation field.
    pub fn equilibrium(&self, state: &State) -> [Vec<f64>; 3] {
        match self.model {
            ViscoModel::OldroydB => OldroydB::new(&state.mesh, self.lambda, self.eta_p).equilibrium(),
            ViscoModel::LogConf => {
                LogConfOldroydB::new(&state.mesh, self.lambda, self.eta_p).equilibrium()
            }
        }
    }
}

impl StateIntegrator for ViscoelasticDualSplitting {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut State, hook: &dyn StateStageHook) {
        let t_new = state.time.t + self.dt;

        // Read velocity + conformation, then run the coupled advance against a
        // transient ViscoelasticFlow built from state.mesh. The flow (which borrows
        // state.mesh) is fully consumed inside this block before we mutate fields.
        let (nux, nuy, npsi) = {
            let (ux, uy) = {
                let v = state.fields.by_id(self.velocity);
                (v.component(0).to_vec(), v.component(1).to_vec())
            };
            let c = {
                let f = state.fields.by_id(self.conformation);
                [f.component(0).to_vec(), f.component(1).to_vec(), f.component(2).to_vec()]
            };
            match self.model {
                ViscoModel::OldroydB => {
                    let m = OldroydB::new(&state.mesh, self.lambda, self.eta_p);
                    let ve = ViscoelasticFlow::with_model(&state.mesh, self.eta_s, self.dt, self.alpha, m);
                    ve.step(&ux, &uy, &c, t_new, &self.bc_u, &self.bc_v, &self.fx, &self.fy)
                }
                ViscoModel::LogConf => {
                    let m = LogConfOldroydB::new(&state.mesh, self.lambda, self.eta_p);
                    let ve = ViscoelasticFlow::with_model(&state.mesh, self.eta_s, self.dt, self.alpha, m);
                    ve.step(&ux, &uy, &c, t_new, &self.bc_u, &self.bc_v, &self.fx, &self.fy)
                }
            }
        };

        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
        }
        {
            let cf = state.fields.by_id_mut(self.conformation);
            for j in 0..3 {
                cf.component_mut(j).copy_from_slice(&npsi[j]);
            }
        }
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}

/// A nodal 3D body force `(fx, fy, fz)` computed from the state and time.
pub type BodyForce3d = Box<dyn Fn(&State<Mesh3d>, f64) -> (Vec<f64>, Vec<f64>, Vec<f64>)>;

/// **Structured** 3D incompressible Navier–Stokes integrator: BDF1 dual-splitting,
/// the `StateIntegrator<Mesh3d>` wrapping the validated [`Stokes3d`] operator
/// (built transiently from `state.mesh`). The 3D analogue of [`DualSplitting`];
/// with [`Penalization3dHook`](super::ibm::Penalization3dHook) as its post-step
/// stage hook it makes flow-past-a-sphere assemblable through the HOOMD API.
pub struct DualSplitting3d {
    pub dt: f64,
    pub nu: f64,
    pub alpha: f64,
    velocity: FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    bc_w: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    body_force: BodyForce3d,
}

impl DualSplitting3d {
    /// New 3D NS integrator advancing the 3-component `velocity` field, zero walls
    /// and no body force by default.
    pub fn new(velocity: FieldId, dt: f64, nu: f64, alpha: f64) -> Self {
        Self {
            dt,
            nu,
            alpha,
            velocity,
            bc_u: Box::new(|_, _, _, _| 0.0),
            bc_v: Box::new(|_, _, _, _| 0.0),
            bc_w: Box::new(|_, _, _, _| 0.0),
            body_force: Box::new(|s: &State<Mesh3d>, _t: f64| {
                let n = s.ndof();
                (vec![0.0; n], vec![0.0; n], vec![0.0; n])
            }),
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v, bc_w)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        bc_w: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self.bc_w = Box::new(bc_w);
        self
    }

    /// Set the nodal body force `(fx, fy, fz)` computed from the state and new time.
    pub fn body_force(
        mut self,
        f: impl Fn(&State<Mesh3d>, f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) + 'static,
    ) -> Self {
        self.body_force = Box::new(f);
        self
    }
}

impl StateIntegrator<Mesh3d> for DualSplitting3d {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut State<Mesh3d>, hook: &dyn StateStageHook<Mesh3d>) {
        let t_new = state.time.t + self.dt;
        let stokes = Stokes3d::new(&state.mesh, self.alpha, self.nu, self.dt);
        let (ux, uy, uz) = {
            let v = state.fields.by_id(self.velocity);
            (v.component(0).to_vec(), v.component(1).to_vec(), v.component(2).to_vec())
        };
        let (bx, by, bz) = (self.body_force)(state, t_new);
        let (nux, nuy, nuz) = stokes.step_ns_forced(
            &ux, &uy, &uz, t_new, &self.bc_u, &self.bc_v, &self.bc_w, &bx, &by, &bz,
        );
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
            v.component_mut(2).copy_from_slice(&nuz);
        }
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::hyperbolic::{Hyperbolic, LinearAdvection};
    use crate::dg::mesh::Mesh2d;
    use crate::sim::integrate::{ClosureSemi, Integrator, SspRk3};
    use std::f64::consts::PI;

    /// Single-field State-level SSP-RK3 must match the vector-level SspRk3
    /// bit-for-bit (same arithmetic, same order).
    #[test]
    fn state_ssprk3_matches_vector_level_single_field() {
        let (ax, ay) = (0.7, -0.3);
        let mesh = Mesh2d::rectangular_periodic(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();

        let mut st = State::new(mesh.clone());
        let uid = st.add_field_from("u", &[|x: f64, y: f64| {
            (2.0 * PI * x).sin() * (2.0 * PI * y).cos()
        }]);

        // Vector-level reference.
        let hyp = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
        let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
        let vsemi =
            ClosureSemi::new(1, hyp.ndof(), |s: &[Vec<f64>], t: f64| hyp.rhs(s, t, &bc));
        let vinteg = SspRk3::new(1e-3);
        let mut vstate = st.field("u").components().to_vec();

        // State-level operator: base advection built transiently from state.mesh.
        let ssemi = StateSemidiscretization::new().field(
            uid,
            Box::new(move |s: &State, t: f64| {
                let h = Hyperbolic::new(&s.mesh, LinearAdvection { ax, ay });
                let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
                h.rhs(s.field("u").components(), t, &bc)
            }),
        );
        let sinteg = SspRk3State::new(1e-3);

        for n in 0..30 {
            vstate = vinteg.step(&vsemi, &vstate, n as f64 * 1e-3);
            sinteg.advance(&ssemi, &mut st, &NoStateHook);
            assert_eq!(st.field("u").component(0), vstate[0].as_slice());
        }
        // sanity
        let _ = nn;
        assert_eq!(st.time.step, 30);
    }

    /// Cross-field coupling via MMS. Two advected scalars with linear coupling:
    ///   ∂ₜa + v·∇a = b + s_a,   ∂ₜb + v·∇b = a + s_b
    /// The `+b`/`+a` couplings are cross-field StateTerms; s_a, s_b are sources
    /// chosen so a = φ·f(t), b = φ·h(t) is exact. The composed run must recover the
    /// exact solution; a control without the coupling terms must not.
    #[test]
    fn cross_field_coupling_recovers_manufactured_solution() {
        let (ax, ay) = (1.0, 0.5);
        let mesh = Mesh2d::rectangular_periodic(6, 8, 8, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();

        let phi = |x: f64, y: f64| (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
        let a_e = move |x: f64, y: f64, t: f64| phi(x, y) * (-t).exp();
        let b_e = move |x: f64, y: f64, t: f64| phi(x, y) * (0.5 * t).cos();

        let run = |with_coupling: bool| -> State {
            let mut st = State::new(mesh.clone());
            let aid = st.add_field_from("a", &[move |x: f64, y: f64| a_e(x, y, 0.0)]);
            let bid = st.add_field_from("b", &[move |x: f64, y: f64| b_e(x, y, 0.0)]);

            // Base rhs per field: -v·∇(field) + source(field, t), sampled at nodes.
            // Sources are module-level fn-pointers so the BaseRhs closures are 'static.
            let base = move |fieldname: &'static str,
                             src: fn(f64, f64, f64) -> f64|
                  -> BaseRhs {
                Box::new(move |s: &State, t: f64| {
                    let hyp = Hyperbolic::new(&s.mesh, LinearAdvection { ax, ay });
                    let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
                    let mut d = hyp.rhs(s.field(fieldname).components(), t, &bc);
                    let nn = s.mesh.refq.n_nodes();
                    for (e, el) in s.mesh.elements.iter().enumerate() {
                        for k in 0..nn {
                            d[0][e * nn + k] += src(el.geom.x[k], el.geom.y[k], t);
                        }
                    }
                    d
                })
            };

            let mut semi = StateSemidiscretization::new()
                .field(aid, base("a", a_src))
                .field(bid, base("b", b_src));

            if with_coupling {
                // dot[a] += b ; dot[b] += a
                semi = semi
                    .with_term(FnStateTerm(move |s: &State, _t: f64, dot: &mut FieldVec| {
                        let bvals = s.field("b").component(0).to_vec();
                        let da = dot.get_mut(aid);
                        for (o, bv) in da[0].iter_mut().zip(&bvals) {
                            *o += *bv;
                        }
                    }))
                    .with_term(FnStateTerm(move |s: &State, _t: f64, dot: &mut FieldVec| {
                        let avals = s.field("a").component(0).to_vec();
                        let db = dot.get_mut(bid);
                        for (o, av) in db[0].iter_mut().zip(&avals) {
                            *o += *av;
                        }
                    }));
            }

            let integ = SspRk3State::new(2e-4);
            for _ in 0..500 {
                integ.advance(&semi, &mut st, &NoStateHook);
            }
            st
        };

        // We need the source functions as fn-pointers but they capture ax/ay/phi.
        // Re-express them as module-level via the helpers below.
        let st = run(true);
        let tf = st.time.t;

        let rel_err = |st: &State, exact: &dyn Fn(f64, f64, f64) -> f64, name: &str| -> f64 {
            let fld = st.field(name);
            let mut e2 = 0.0;
            let mut n2 = 0.0;
            for (e, el) in st.mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    let ue = exact(el.geom.x[k], el.geom.y[k], tf);
                    let d = fld.component(0)[e * nn + k] - ue;
                    e2 += el.geom.jw[k] * d * d;
                    n2 += el.geom.jw[k] * ue * ue;
                }
            }
            (e2 / n2).sqrt()
        };

        let ra = rel_err(&st, &move |x, y, t| a_e(x, y, t), "a");
        let rb = rel_err(&st, &move |x, y, t| b_e(x, y, t), "b");
        assert!(ra < 1e-4, "coupled field a rel err {ra:e}");
        assert!(rb < 1e-4, "coupled field b rel err {rb:e}");

        let ctrl = run(false);
        let ca = rel_err(&ctrl, &move |x, y, t| a_e(x, y, t), "a");
        let cb = rel_err(&ctrl, &move |x, y, t| b_e(x, y, t), "b");
        assert!(ca > 1e-2 || cb > 1e-2, "control without coupling should be off: {ca:e}, {cb:e}");
    }

    // Module-level source functions for the MMS test (need to be fn-pointers, so
    // the advection constants are baked in as literals matching the test).
    fn a_src(x: f64, y: f64, t: f64) -> f64 {
        let (ax, ay) = (1.0, 0.5);
        let phi = (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
        let phix = 2.0 * PI * (2.0 * PI * x).cos() * (2.0 * PI * y).cos();
        let phiy = -2.0 * PI * (2.0 * PI * x).sin() * (2.0 * PI * y).sin();
        let f = (-t).exp();
        let fp = -(-t).exp();
        let b = phi * (0.5 * t).cos();
        phi * fp + (ax * phix + ay * phiy) * f - b
    }

    fn b_src(x: f64, y: f64, t: f64) -> f64 {
        let (ax, ay) = (1.0, 0.5);
        let phi = (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
        let phix = 2.0 * PI * (2.0 * PI * x).cos() * (2.0 * PI * y).cos();
        let phiy = -2.0 * PI * (2.0 * PI * x).sin() * (2.0 * PI * y).sin();
        let h = (0.5 * t).cos();
        let hp = -0.5 * (0.5 * t).sin();
        let a = phi * (-t).exp();
        phi * hp + (ax * phix + ay * phiy) * h - a
    }

    /// The structured `DualSplitting` integrator, driven through the same
    /// State-level API as the explicit `Mol`, must reproduce the decaying
    /// Taylor–Green vortex (an exact NS solution) — and match a direct `Stokes`
    /// loop bit-for-bit (faithful wrapping of the validated operator).
    #[test]
    fn dual_splitting_recovers_taylor_green() {
        use crate::dg::stokes::Stokes;

        let nu = 1.0;
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let t_end = 0.1;
        let nsteps = 20u64;
        let dt = t_end / nsteps as f64;

        // Exact Taylor–Green (move-captures nu only ⇒ 'static, boxable as BCs).
        let eu = move |x: f64, y: f64, t: f64| {
            -(PI * x).cos() * (PI * y).sin() * (-2.0 * PI * PI * nu * t).exp()
        };
        let ev = move |x: f64, y: f64, t: f64| {
            (PI * x).sin() * (PI * y).cos() * (-2.0 * PI * PI * nu * t).exp()
        };

        // Framework run: velocity field + DualSplitting integrator.
        let mut st = State::new(mesh.clone());
        let vid = st.add_field_from(
            "velocity",
            &[
                Box::new(move |x: f64, y: f64| eu(x, y, 0.0)) as Box<dyn Fn(f64, f64) -> f64>,
                Box::new(move |x: f64, y: f64| ev(x, y, 0.0)),
            ],
        );
        let integ = DualSplitting::new(vid, dt, nu, 5.0).boundary(eu, ev);
        for _ in 0..nsteps {
            integ.step(&mut st, &NoStateHook);
        }

        // Reference: the validated direct Stokes loop, identical setup.
        let stokes = Stokes::new(&mesh, 5.0, nu, dt);
        let mut rux: Vec<f64> = Vec::new();
        let mut ruy: Vec<f64> = Vec::new();
        for (e, el) in mesh.elements.iter().enumerate() {
            let _ = e;
            for k in 0..nn {
                rux.push(eu(el.geom.x[k], el.geom.y[k], 0.0));
                ruy.push(ev(el.geom.x[k], el.geom.y[k], 0.0));
            }
        }
        let zero = |_: f64, _: f64, _: f64| 0.0;
        let mut t = 0.0;
        for _ in 0..nsteps {
            t += dt;
            let (nx, ny) = stokes.step_ns(&rux, &ruy, t, eu, ev, zero, zero);
            rux = nx;
            ruy = ny;
        }

        // Bit-for-bit: the framework wraps the operator faithfully.
        assert_eq!(st.field("velocity").component(0), rux.as_slice());
        assert_eq!(st.field("velocity").component(1), ruy.as_slice());

        // Physics: the framework run matches the exact vortex.
        let mut e2 = 0.0;
        let mut n2 = 0.0;
        let v = st.field("velocity");
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                let du = v.component(0)[e * nn + k] - eu(x, y, t_end);
                let dv = v.component(1)[e * nn + k] - ev(x, y, t_end);
                e2 += el.geom.jw[k] * (du * du + dv * dv);
                n2 += el.geom.jw[k] * (eu(x, y, t_end).powi(2) + ev(x, y, t_end).powi(2));
            }
        }
        // Relative L2 error. BDF1 dual-splitting is first-order in time; at dt=5e-3
        // this is ≈5e-3 relative (≈5e-4 absolute, inside the validated Stokes bound).
        // The bit-for-bit equality above already pins the physics to that operator;
        // this is a sanity floor.
        let rel = (e2 / n2).sqrt();
        assert!(rel < 1e-2, "Taylor–Green relative L2 error too large: {rel:e}");
    }

    /// The coupled viscoelastic integrator, driven through the framework, must
    /// match a direct ViscoelasticFlow loop **bit-for-bit** — the definitive proof
    /// that it wraps the validated operator faithfully (velocity *and* conformation,
    /// in the correct split order). Bit-for-bit equality holds at every step, so a
    /// short run suffices; the steady-state η₀ parabola is the validated operator's
    /// own result (dg::viscoelastic::coupled_channel_recovers_total_viscosity),
    /// inherited here by exact equality rather than re-run (which would be a slow
    /// 600-step double solve).
    #[test]
    fn viscoelastic_dual_splitting_matches_flow_bit_for_bit() {
        use crate::dg::viscoelastic::ViscoelasticFlow;

        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let (eta_s, eta_p, lambda, g) = (0.5, 0.5, 0.5, 1.0);
        let eta0 = eta_s + eta_p;
        let dt = 0.02;
        let nn = mesh.refq.n_nodes();
        let u_exact = move |y: f64| (g / (2.0 * eta0)) * y * (1.0 - y);
        let bc_u = move |_x: f64, y: f64, _t: f64| u_exact(y);
        let bc_v = |_: f64, _: f64, _: f64| 0.0;
        let drive_x = move |_: f64, _: f64, _: f64| g;
        let zero_f = |_: f64, _: f64, _: f64| 0.0;

        // Framework run.
        let mut st = State::new(mesh.clone());
        let vid = st.add_field("velocity", 2);
        let cid = st.add_field("conformation", 3);
        let integ = ViscoelasticDualSplitting::new(
            vid,
            cid,
            dt,
            eta_s,
            eta_p,
            lambda,
            5.0,
            ViscoModel::OldroydB,
        )
        .boundary(bc_u, bc_v)
        .drive(drive_x, zero_f);
        // Initialize conformation to equilibrium (identity for Oldroyd-B).
        let eq = integ.equilibrium(&st);
        {
            let cf = st.fields.by_id_mut(cid);
            for j in 0..3 {
                cf.component_mut(j).copy_from_slice(&eq[j]);
            }
        }
        let nsteps = 40;
        for _ in 0..nsteps {
            integ.step(&mut st, &NoStateHook);
        }

        // Reference: the validated direct ViscoelasticFlow loop, identical setup.
        let ve = ViscoelasticFlow::new(&mesh, eta_s, eta_p, lambda, dt, 5.0);
        let mut rux = vec![0.0; mesh.n_elements() * nn];
        let mut ruy = rux.clone();
        let mut rc = ve.model.equilibrium();
        let mut t = 0.0;
        for _ in 0..nsteps {
            t += dt;
            let (nx, ny, nc) = ve.step(&rux, &ruy, &rc, t, bc_u, bc_v, drive_x, zero_f);
            rux = nx;
            ruy = ny;
            rc = nc;
        }

        // Bit-for-bit faithful wrapping: velocity and all conformation components.
        assert_eq!(st.field("velocity").component(0), rux.as_slice());
        assert_eq!(st.field("velocity").component(1), ruy.as_slice());
        for j in 0..3 {
            assert_eq!(st.field("conformation").component(j), rc[j].as_slice());
        }
        // Sanity: the channel is developing (nonzero, finite velocity).
        let umax = st.field("velocity").component(0).iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(umax.is_finite() && umax > 1e-3, "channel did not develop: umax={umax}");
    }
}
