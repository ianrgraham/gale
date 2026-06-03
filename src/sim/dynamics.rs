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
    pub fn extract(state: &State, ids: &[FieldId]) -> Self {
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
    pub fn scatter_into(&self, state: &mut State) {
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
pub trait StateTerm {
    fn accumulate(&self, state: &State, t: f64, dot: &mut FieldVec);
}

/// Adapter wrapping a closure as a [`StateTerm`].
pub struct FnStateTerm<F>(pub F);

impl<F> StateTerm for FnStateTerm<F>
where
    F: Fn(&State, f64, &mut FieldVec),
{
    fn accumulate(&self, state: &State, t: f64, dot: &mut FieldVec) {
        (self.0)(state, t, dot)
    }
}

/// A State-level semi-discrete operator: produces `∂ₜ(evolving fields)`.
pub trait StateSemi {
    /// The fields this operator evolves in time.
    fn evolving(&self) -> Vec<FieldId>;
    /// Fill `dot` with the time derivative of every evolving field, given the full
    /// `state` at time `t`. `dot` arrives zeroed with the right shapes.
    fn rhs(&self, state: &State, t: f64, dot: &mut FieldVec);
}

/// Per-field base rhs: given the full state and time, returns that field's base
/// (un-coupled) time derivative in `[n_comp][ndof]` layout. Built to construct its
/// operator transiently from `state.mesh`, avoiding any stored mesh borrow.
pub type BaseRhs = Box<dyn Fn(&State, f64) -> Vec<Vec<f64>>>;

/// The concrete State-level semidiscretization: one base rhs per evolving field
/// plus a list of additive cross-field [`StateTerm`]s.
pub struct StateSemidiscretization {
    bases: Vec<(FieldId, BaseRhs)>,
    terms: Vec<Box<dyn StateTerm>>,
}

impl StateSemidiscretization {
    pub fn new() -> Self {
        Self { bases: Vec::new(), terms: Vec::new() }
    }

    /// Register the base rhs for an evolving field.
    pub fn field(mut self, id: FieldId, base: BaseRhs) -> Self {
        self.bases.push((id, base));
        self
    }

    /// Append an additive cross-field term.
    pub fn with_term(mut self, term: impl StateTerm + 'static) -> Self {
        self.terms.push(Box::new(term));
        self
    }
}

impl Default for StateSemidiscretization {
    fn default() -> Self {
        Self::new()
    }
}

impl StateSemi for StateSemidiscretization {
    fn evolving(&self) -> Vec<FieldId> {
        self.bases.iter().map(|(id, _)| *id).collect()
    }

    fn rhs(&self, state: &State, t: f64, dot: &mut FieldVec) {
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
pub trait StateStageHook {
    fn after_stage(&self, state: &mut State, stage: usize);
}

/// The no-op State-level stage hook.
pub struct NoStateHook;

impl StateStageHook for NoStateHook {
    fn after_stage(&self, _state: &mut State, _stage: usize) {}
}

/// A State-level time integrator: advances the evolving fields of `state` by one
/// step, applying `hook` between stages and updating `state.time`.
pub trait StateIntegrator {
    fn dt(&self) -> f64;
    fn step(&self, semi: &dyn StateSemi, state: &mut State, hook: &dyn StateStageHook);
}

/// Explicit SSP-RK3 (Shu–Osher) over a multi-field [`State`]. The combination
/// arithmetic mirrors the vector-level [`SspRk3`](super::integrate::SspRk3) so a
/// single-field problem advances bit-for-bit identically.
#[derive(Clone, Copy, Debug)]
pub struct SspRk3State {
    pub dt: f64,
}

impl SspRk3State {
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }
}

impl StateIntegrator for SspRk3State {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, semi: &dyn StateSemi, state: &mut State, hook: &dyn StateStageHook) {
        let t = state.time.t;
        let dt = self.dt;
        let ev = semi.evolving();
        let u0 = FieldVec::extract(state, &ev);

        let eval = |state: &State, t: f64| -> FieldVec {
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
            sinteg.step(&ssemi, &mut st, &NoStateHook);
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
                integ.step(&semi, &mut st, &NoStateHook);
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
}
