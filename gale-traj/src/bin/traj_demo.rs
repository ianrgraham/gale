//! Smoke/demo for the HDF5 trajectory writer: run a pure-CPU scalar-advection sim (a Gaussian
//! blob translating on a periodic mesh) and dump it to one `.h5` trajectory. No GPU — this just
//! exercises the writer end-to-end and produces a file for the Python viewer.
//!
//! Run: cargo run -p gale-traj --bin traj-demo   →   writes /tmp/advection.h5

use gale::dg::{Hyperbolic, LinearAdvection, Mesh2d};
use gale::sim::{ClosureSemi, Integrator, SspRk3, State};
use gale_traj::TrajectoryWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (nx, ny) = (16usize, 16usize);
    let (ax, ay) = (0.6, 0.4);
    let dt = 1.5e-3;
    let nsteps = 600u64;
    let dump_every = 3u64;
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/advection.h5".to_string());

    let mesh = Mesh2d::rectangular_periodic(p, nx, ny, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    // Localized Gaussian blob (≈0 at the boundaries ⇒ effectively periodic), advected diagonally.
    let init = |x: f64, y: f64| {
        let (dx, dy) = (x - 0.3, y - 0.3);
        (-(dx * dx + dy * dy) / (2.0 * 0.08 * 0.08)).exp()
    };

    let hyp = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
    let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
    let csemi = ClosureSemi::new(1, hyp.ndof(), |s: &[Vec<f64>], t: f64| hyp.rhs(s, t, &bc));
    let scheme = SspRk3::new(dt);

    let mut field: Vec<Vec<f64>> = {
        let mut st = State::new(mesh.clone());
        st.add_field_from("u", &[init]);
        vec![st.field("u").component(0).to_vec()]
    };

    let mut w = TrajectoryWriter::create(&out, p, 2)?;
    let topo = w.write_mesh2d(&mesh)?;
    let f32v = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    w.write_frame(0.0, 0, topo, ne, nn, &[("u", f32v(&field[0]), 1)], None)?;

    let mut t = 0.0;
    for step in 1..=nsteps {
        field = scheme.step(&csemi, &field, t);
        t += dt;
        if step % dump_every == 0 {
            w.write_frame(t, step, topo, ne, nn, &[("u", f32v(&field[0]), 1)], None)?;
        }
    }

    println!(
        "wrote {} frames ({ne} elems × {nn} nodes, p={p}) → {out}",
        w.n_frames()
    );
    Ok(())
}
