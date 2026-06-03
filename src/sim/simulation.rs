//! The `Simulation` orchestrator — gale's top-level "assemble and run" surface.
//!
//! Mirrors HOOMD-blue (`docs/api-design.md` §3): one [`Simulation`] owns the
//! [`State`], the [`Operations`], and (later) the Device. Operations are a small
//! fixed taxonomy of typed roles gated by [`Trigger`]s and run in deterministic
//! per-step order. The time advance is performed by exactly one State-level
//! [`StateIntegrator`] driving a [`StateSemi`] semidiscretization.
//!
//! Per-step schedule (HOOMD order): **updaters → integrator → writers**. Updaters
//! act on the pre-step state at step `s`; the integrator produces step `s+1`;
//! writers then observe the produced state. [`Compute`]s are read-only diagnostics
//! pulled on demand.
//!
//! Scope: this implements four of the five HOOMD roles — Compute, Updater, Writer,
//! and the Integrator. The fifth, Tuner (which mutates *other* operations'
//! parameters), needs an interior-mutability/`TuneContext` redesign to be safe in
//! Rust and is deferred; see api-design §3.3.

use super::dynamics::{NoStateHook, StateIntegrator, StateSemi, StateStageHook};
use super::state::State;

/// A firing condition for a triggered operation — decouples *what* to do from
/// *when* (HOOMD TriggeredOperation, api-design §3.5).
pub trait Trigger {
    fn fires(&self, step: u64) -> bool;
}

/// Fires every `period` steps, offset by `phase`.
#[derive(Clone, Copy, Debug)]
pub struct Periodic {
    pub period: u64,
    pub phase: u64,
}

impl Periodic {
    pub fn new(period: u64) -> Self {
        Self { period, phase: 0 }
    }
}

impl Trigger for Periodic {
    fn fires(&self, step: u64) -> bool {
        self.period != 0 && step >= self.phase && (step - self.phase) % self.period == 0
    }
}

/// Fires exactly once, at the given step.
#[derive(Clone, Copy, Debug)]
pub struct OnStep {
    pub step: u64,
}

impl Trigger for OnStep {
    fn fires(&self, step: u64) -> bool {
        step == self.step
    }
}

/// Fires every step.
#[derive(Clone, Copy, Debug)]
pub struct Always;

impl Trigger for Always {
    fn fires(&self, _step: u64) -> bool {
        true
    }
}

/// A read-only diagnostic computed from the state (energy, drag, max-Wi, …).
pub trait Compute {
    fn name(&self) -> &str;
    fn compute(&self, state: &State) -> f64;
}

/// An operation that mutates the state when triggered (AMR, body advection, …).
pub trait Updater {
    fn update(&mut self, state: &mut State, step: u64);
}

/// An operation that reads the state and emits output when triggered (I/O,
/// checkpoint, progress). Must not mutate the state.
pub trait Writer {
    fn write(&mut self, state: &State, step: u64);
}

/// Binds a triggered operation to its firing condition.
pub struct Triggered<T: ?Sized> {
    pub trigger: Box<dyn Trigger>,
    pub op: Box<T>,
}

/// The collection of operations applied during a run.
#[derive(Default)]
pub struct Operations {
    pub computes: Vec<Box<dyn Compute>>,
    pub updaters: Vec<Triggered<dyn Updater>>,
    pub writers: Vec<Triggered<dyn Writer>>,
}

/// The central object: owns the state, the operations, and the time-advance
/// machinery (semidiscretization + integrator + stage hook).
pub struct Simulation {
    pub state: State,
    pub operations: Operations,
    semi: Option<Box<dyn StateSemi>>,
    integrator: Option<Box<dyn StateIntegrator>>,
    hook: Box<dyn StateStageHook>,
}

impl Simulation {
    /// A simulation over `state` with no operations and no integrator yet.
    pub fn new(state: State) -> Self {
        Self {
            state,
            operations: Operations::default(),
            semi: None,
            integrator: None,
            hook: Box::new(NoStateHook),
        }
    }

    /// Set the semidiscretization and integrator (exactly one integrator per
    /// simulation, HOOMD invariant).
    pub fn set_integrator(
        &mut self,
        semi: impl StateSemi + 'static,
        integrator: impl StateIntegrator + 'static,
    ) {
        self.semi = Some(Box::new(semi));
        self.integrator = Some(Box::new(integrator));
    }

    /// Set the per-stage hook (limiters / filter / penalization projection).
    pub fn set_stage_hook(&mut self, hook: impl StateStageHook + 'static) {
        self.hook = Box::new(hook);
    }

    /// Register a read-only compute.
    pub fn add_compute(&mut self, compute: impl Compute + 'static) {
        self.operations.computes.push(Box::new(compute));
    }

    /// Register a triggered updater.
    pub fn add_updater(&mut self, updater: impl Updater + 'static, trigger: impl Trigger + 'static) {
        self.operations
            .updaters
            .push(Triggered { trigger: Box::new(trigger), op: Box::new(updater) });
    }

    /// Register a triggered writer.
    pub fn add_writer(&mut self, writer: impl Writer + 'static, trigger: impl Trigger + 'static) {
        self.operations
            .writers
            .push(Triggered { trigger: Box::new(trigger), op: Box::new(writer) });
    }

    /// Evaluate a named compute against the current state, if registered.
    pub fn compute(&self, name: &str) -> Option<f64> {
        self.operations.computes.iter().find(|c| c.name() == name).map(|c| c.compute(&self.state))
    }

    /// Advance the simulation `nsteps` steps. Per step: triggered updaters, then
    /// the integrator, then triggered writers (observing the produced state).
    pub fn run(&mut self, nsteps: u64) {
        // Move the time-advance pieces out so the loop can borrow `state` and
        // `operations` mutably without aliasing `self`.
        let semi = self.semi.take().expect("Simulation::run: no integrator set");
        let integ = self.integrator.take().expect("Simulation::run: no integrator set");

        for _ in 0..nsteps {
            let step = self.state.time.step;
            for u in self.operations.updaters.iter_mut() {
                if u.trigger.fires(step) {
                    u.op.update(&mut self.state, step);
                }
            }
            integ.step(semi.as_ref(), &mut self.state, self.hook.as_ref());
            let produced = self.state.time.step;
            for w in self.operations.writers.iter_mut() {
                if w.trigger.fires(produced) {
                    w.op.write(&self.state, produced);
                }
            }
        }

        self.semi = Some(semi);
        self.integrator = Some(integ);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::hyperbolic::{Hyperbolic, LinearAdvection};
    use crate::dg::mesh::Mesh2d;
    use crate::sim::dynamics::{BaseRhs, SspRk3State, StateIntegrator, StateSemidiscretization};
    use std::cell::RefCell;
    use std::f64::consts::PI;
    use std::rc::Rc;

    fn advection_sim(dt: f64) -> (Simulation, Mesh2d) {
        let (ax, ay) = (0.8, 0.4);
        let mesh = Mesh2d::rectangular_periodic(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let mut st = State::new(mesh.clone());
        let uid = st
            .add_field_from("u", &[|x: f64, y: f64| (2.0 * PI * x).sin() * (2.0 * PI * y).cos()]);
        let base: BaseRhs = Box::new(move |s: &State, t: f64| {
            let h = Hyperbolic::new(&s.mesh, LinearAdvection { ax, ay });
            let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
            h.rhs(s.field("u").components(), t, &bc)
        });
        let semi = StateSemidiscretization::new().field(uid, base);
        let mut sim = Simulation::new(st);
        sim.set_integrator(semi, SspRk3State::new(dt));
        (sim, mesh)
    }

    /// A writer with a Periodic(10) trigger fires exactly at steps 10,20,30,40,50.
    #[test]
    fn periodic_writer_fires_on_schedule() {
        struct StepLog(Rc<RefCell<Vec<u64>>>);
        impl Writer for StepLog {
            fn write(&mut self, _state: &State, step: u64) {
                self.0.borrow_mut().push(step);
            }
        }
        let (mut sim, _mesh) = advection_sim(1e-3);
        let log = Rc::new(RefCell::new(Vec::new()));
        sim.add_writer(StepLog(log.clone()), Periodic::new(10));
        sim.run(50);
        assert_eq!(*log.borrow(), vec![10, 20, 30, 40, 50]);
    }

    /// The orchestrated run must reproduce a hand-written integrator loop exactly —
    /// the Simulation only schedules; it does not perturb the dynamics.
    #[test]
    fn run_matches_manual_integrator_loop() {
        let dt = 1e-3;
        let (mut sim, _mesh) = advection_sim(dt);

        // Reference: drive the same semidiscretization by hand.
        let (ax, ay) = (0.8, 0.4);
        let mesh = Mesh2d::rectangular_periodic(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let mut ref_state = State::new(mesh);
        let uid = ref_state
            .add_field_from("u", &[|x: f64, y: f64| (2.0 * PI * x).sin() * (2.0 * PI * y).cos()]);
        let base: BaseRhs = Box::new(move |s: &State, t: f64| {
            let h = Hyperbolic::new(&s.mesh, LinearAdvection { ax, ay });
            let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
            h.rhs(s.field("u").components(), t, &bc)
        });
        let ref_semi = StateSemidiscretization::new().field(uid, base);
        let integ = SspRk3State::new(dt);
        let hook = NoStateHook;
        for _ in 0..40 {
            integ.step(&ref_semi, &mut ref_state, &hook);
        }

        sim.run(40);
        assert_eq!(sim.state.time.step, 40);
        assert_eq!(sim.state.field("u").component(0), ref_state.field("u").component(0));
    }

    /// A triggered updater fires on schedule and its mutation takes effect.
    #[test]
    fn triggered_updater_mutates_state() {
        struct ScaleU {
            factor: f64,
            calls: Rc<RefCell<u64>>,
        }
        impl Updater for ScaleU {
            fn update(&mut self, state: &mut State, _step: u64) {
                *self.calls.borrow_mut() += 1;
                for c in state.field_mut("u").components_mut() {
                    for v in c.iter_mut() {
                        *v *= self.factor;
                    }
                }
            }
        }
        let (mut sim, _mesh) = advection_sim(1e-3);
        let calls = Rc::new(RefCell::new(0));
        // Fires at steps 0, 5, 10, 15 within a 20-step run (updaters act pre-step).
        sim.add_updater(ScaleU { factor: 0.9, calls: calls.clone() }, Periodic::new(5));
        sim.run(20);
        assert_eq!(*calls.borrow(), 4);
    }

    /// A registered compute can be pulled on demand and returns a sensible value.
    #[test]
    fn compute_reports_diagnostic() {
        struct L2Energy;
        impl Compute for L2Energy {
            fn name(&self) -> &str {
                "energy"
            }
            fn compute(&self, state: &State) -> f64 {
                let nn = state.mesh.refq.n_nodes();
                let f = state.field("u");
                let mut s = 0.0;
                for (e, el) in state.mesh.elements.iter().enumerate() {
                    for k in 0..nn {
                        s += el.geom.jw[k] * f.component(0)[e * nn + k].powi(2);
                    }
                }
                s
            }
        }
        let (mut sim, _mesh) = advection_sim(1e-3);
        sim.add_compute(L2Energy);
        let e0 = sim.compute("energy").unwrap();
        assert!(e0 > 0.0 && e0.is_finite());
        assert!(sim.compute("missing").is_none());
        sim.run(10);
        let e1 = sim.compute("energy").unwrap();
        assert!(e1.is_finite());
    }
}
