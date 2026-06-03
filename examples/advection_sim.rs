//! HOOMD-style assemble-and-run demo for the gale `sim` framework.
//!
//! Run with the ordinary toolchain (pure host code, no GPU backend needed):
//!
//! ```text
//! cargo run --release --example advection_sim
//! ```
//!
//! This is the §4 sketch of `docs/api-design.md` exercised against what the
//! framework implements today: build a `State`, attach a `Semidiscretization`
//! (here, scalar linear advection), choose an `Integrator`, register a `Compute`
//! diagnostic and a `Writer` on a `Trigger`, then `run`. No loop logic lives in
//! the driver — the `Simulation` schedules everything.

use gale::dg::hyperbolic::{Hyperbolic, LinearAdvection};
use gale::dg::mesh::Mesh2d;
use gale::sim::dynamics::{BaseRhs, SspRk3State, StateSemidiscretization};
use gale::sim::{Compute, Periodic, Simulation, State, Writer};
use std::f64::consts::PI;

/// L2 energy ∫ u² of the transported scalar.
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

/// Prints the current step and L2 energy each time it fires.
struct Progress;
impl Writer for Progress {
    fn write(&mut self, state: &State, step: u64) {
        let e = L2Energy.compute(state);
        println!("  step {step:>5}   t = {:>7.4}   ∫u² = {:.6}", state.time.t, e);
    }
}

fn main() {
    // 1. Mesh + state: a transported scalar on the periodic unit square.
    let (ax, ay) = (1.0, 0.5);
    let mesh = Mesh2d::rectangular_periodic(5, 8, 8, [0.0, 1.0], [0.0, 1.0]);
    let mut state = State::new(mesh);
    let uid = state.add_field_from("u", &[|x: f64, y: f64| {
        (2.0 * PI * x).sin() * (2.0 * PI * y).cos()
    }]);

    // 2. Physics: scalar advection. The base operator is built transiently from
    //    state.mesh inside the rhs, so the Simulation owns everything cleanly.
    let base: BaseRhs = Box::new(move |s: &State, t: f64| {
        let hyp = Hyperbolic::new(&s.mesh, LinearAdvection { ax, ay });
        let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
        hyp.rhs(s.field("u").components(), t, &bc)
    });
    let semi = StateSemidiscretization::new().field(uid, base);

    // 3. Assemble: one Simulation owns the State, the Operations, and (here) the
    //    integrator. Register a diagnostic and a progress writer on a trigger.
    let mut sim = Simulation::new(state);
    sim.set_integrator(semi, SspRk3State::new(1e-3));
    sim.add_compute(L2Energy);
    sim.add_writer(Progress, Periodic::new(200));

    // 4. Run. Linear advection with the central/Rusanov flux is energy-stable, so
    //    ∫u² should stay ~constant (mild numerical dissipation).
    let e0 = sim.compute("energy").unwrap();
    println!("gale advection demo — periodic unit square, p=5, 8×8 elements");
    println!("  step {:>5}   t = {:>7.4}   ∫u² = {:.6}  (initial)", 0, 0.0, e0);
    sim.run(1000);
    let e1 = sim.compute("energy").unwrap();

    println!(
        "done: {} steps, ∫u² {:.6} → {:.6}  ({:+.3}%)",
        sim.state.time.step,
        e0,
        e1,
        100.0 * (e1 - e0) / e0
    );
}
