//! Stage 3e: an end-to-end **device-resident ADAPTIVE** NS flow. Device-resident dual-splitting
//! steps (assembly + solves via `GpuPoissonNc`, no per-step field transfer) run between
//! host-orchestrated remeshes; on each remesh the fields are remapped (`remap_component_flat`, the
//! only host↔device round-trip — and only every N steps, not per step). Validated against a fully
//! host-orchestrated adaptive reference following the IDENTICAL adapt schedule + remap, so they must
//! agree to solver tolerance. Demonstrates the GPU-resident-per-step adaptive loop end to end.
//! Run: cargo oxide run --bin device-resident-adapt-check

use gale::dg::{remap_component_flat, Mesh2d, Poisson};
use gale_gpu::operators::poisson_nc::GpuPoissonNc;

const P: usize = 3;
const NX: usize = 8;
const NY: usize = 8;
const ALPHA: f64 = 5.0;
const NU: f64 = 0.05;
const DT: f64 = 5e-3;
const TOL: f64 = 1e-9;
const MAXIT: usize = 5000;

fn lid(_x: f64, y: f64) -> f64 {
    if y > 1.0 - 1e-9 { 1.0 } else { 0.0 }
}

/// Build the per-mesh constants (jw, lifts).
fn consts(mesh: &Mesh2d) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    let lambda = 1.0 / (NU * DT);
    let vel = Poisson::with_reaction(mesh, ALPHA, lambda);
    (
        vel.rhs(&vec![1.0; ndof], |_, _| 0.0),
        vel.rhs(&vec![0.0; ndof], lid),
        vel.rhs(&vec![0.0; ndof], |_, _| 0.0),
        vec![0.0; ndof],
    )
}

fn grad(mesh: &Mesh2d, f: &[f64], comp: usize) -> Vec<f64> {
    let nn = mesh.refq.n_nodes();
    let mut g = vec![0.0; f.len()];
    for (e, el) in mesh.elements.iter().enumerate() {
        let sl = &f[e * nn..(e + 1) * nn];
        let gv = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
        g[e * nn..(e + 1) * nn].copy_from_slice(&gv);
    }
    g
}

/// HOST-orchestrated NS segment: `nsteps` dual-splitting steps on `mesh` (host stages + GPU NC solve).
fn host_segment(mesh: &Mesh2d, ux: &mut Vec<f64>, uy: &mut Vec<f64>, nsteps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    let lambda = 1.0 / (NU * DT);
    let tags = mesh.boundary_tags();
    let (jw, lift_vx, lift_vy, lift_p) = consts(mesh);
    let h = GpuPoissonNc::new(mesh, ALPHA)?;
    for _ in 0..nsteps {
        let (gxx, gyx) = (grad(mesh, ux, 0), grad(mesh, ux, 1));
        let (gxy, gyy) = (grad(mesh, uy, 0), grad(mesh, uy, 1));
        let mut uhx = vec![0.0; ndof];
        let mut uhy = vec![0.0; ndof];
        for i in 0..ndof {
            uhx[i] = ux[i] - DT * (ux[i] * gxx[i] + uy[i] * gyx[i]);
            uhy[i] = uy[i] - DT * (ux[i] * gxy[i] + uy[i] * gyy[i]);
        }
        let div: Vec<f64> = grad(mesh, &uhx, 0).iter().zip(grad(mesh, &uhy, 1).iter()).map(|(a, b)| a + b).collect();
        let bp: Vec<f64> = (0..ndof).map(|i| jw[i] * (-div[i] / DT) + lift_p[i]).collect();
        let pp = h.solve(&bp, 0.0, &tags, true, TOL, MAXIT)?.0;
        let (px, py) = (grad(mesh, &pp, 0), grad(mesh, &pp, 1));
        for i in 0..ndof {
            uhx[i] -= DT * px[i];
            uhy[i] -= DT * py[i];
        }
        let bx: Vec<f64> = (0..ndof).map(|i| lambda * jw[i] * uhx[i] + lift_vx[i]).collect();
        let by: Vec<f64> = (0..ndof).map(|i| lambda * jw[i] * uhy[i] + lift_vy[i]).collect();
        *ux = h.solve(&bx, lambda, &[], false, TOL, MAXIT)?.0;
        *uy = h.solve(&by, lambda, &[], false, TOL, MAXIT)?.0;
    }
    Ok(())
}

/// DEVICE-RESIDENT NS segment: `nsteps` steps entirely on the GPU (assembly + solves via GpuPoissonNc),
/// fields uploaded once at the start and downloaded once at the end (no per-step transfer).
fn device_segment(mesh: &Mesh2d, ux: &mut Vec<f64>, uy: &mut Vec<f64>, nsteps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let lambda = 1.0 / (NU * DT);
    let tags = mesh.boundary_tags();
    let (jw, lift_vx, lift_vy, lift_p) = consts(mesh);
    let h = GpuPoissonNc::new(mesh, ALPHA)?;
    let (jw_d, lvx, lvy, lp) = (h.upload(&jw)?, h.upload(&lift_vx)?, h.upload(&lift_vy)?, h.upload(&lift_p)?);
    let mut dux = h.upload(ux)?;
    let mut duy = h.upload(uy)?;
    let (mut uhx, mut uhy) = (h.alloc()?, h.alloc()?);
    let (mut ga, mut gb, mut gc, mut gd) = (h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?);
    let (mut cx, mut cy, mut div, mut pp, mut rhs) = (h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?);
    for _ in 0..nsteps {
        h.gradient_dev(&dux, &mut ga, &mut gb)?;
        h.gradient_dev(&duy, &mut gc, &mut gd)?;
        h.fma2_dev(&mut cx, &dux, &ga, &duy, &gb)?;
        h.fma2_dev(&mut cy, &dux, &gc, &duy, &gd)?;
        h.copy_dev(&mut uhx, &dux)?;
        h.copy_dev(&mut uhy, &duy)?;
        h.axpy_dev(&mut uhx, &cx, -DT)?;
        h.axpy_dev(&mut uhy, &cy, -DT)?;
        h.gradient_dev(&uhx, &mut ga, &mut gb)?;
        h.gradient_dev(&uhy, &mut gc, &mut gd)?;
        h.copy_dev(&mut div, &ga)?;
        h.axpy_dev(&mut div, &gd, 1.0)?;
        h.scal_dev(&mut div, -1.0 / DT)?;
        h.rhs_madd_dev(&mut rhs, &jw_d, &div, &lp, 1.0)?;
        h.solve_dev(&rhs, None, &mut pp, 0.0, &tags, true, TOL, MAXIT)?;
        h.gradient_dev(&pp, &mut ga, &mut gb)?;
        h.axpy_dev(&mut uhx, &ga, -DT)?;
        h.axpy_dev(&mut uhy, &gb, -DT)?;
        h.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lvx, lambda)?;
        h.solve_dev(&rhs, None, &mut dux, lambda, &[], false, TOL, MAXIT)?;
        h.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lvy, lambda)?;
        h.solve_dev(&rhs, None, &mut duy, lambda, &[], false, TOL, MAXIT)?;
    }
    *ux = h.download(&dux)?;
    *uy = h.download(&duy)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fixed adapt schedule: progressively refine a corner/diagonal region (shared by both paths).
    let schedule: Vec<(Vec<(usize, usize)>, usize)> = vec![
        (vec![], 4),
        (vec![(2, 2), (3, 3)], 4),
        (vec![(2, 2), (3, 3), (4, 4), (5, 5)], 4),
        (vec![(3, 3), (4, 4)], 4), // a coarsen step (drop (2,2),(5,5))
    ];
    println!("=== Stage 3e: device-resident adaptive NS vs host-orchestrated adaptive — {NX}×{NY} p={P}, {} adapts ===", schedule.len());

    let run = |device: bool| -> Result<(Vec<f64>, Vec<f64>, usize), Box<dyn std::error::Error>> {
        let mut prev: Vec<(usize, usize)> = vec![];
        let n0 = NX * NY * mesh_nn(); // ndof of the uniform base
        let (mut ux, mut uy) = (vec![0.0; n0], vec![0.0; n0]);
        let mut last_ne = 0;
        for (set, nsteps) in &schedule {
            // Remap fields from the previous refine-state to this one (host bookkeeping; the only
            // host↔device traffic, and only on adapt).
            ux = remap_component_flat(P, NX, NY, &prev, &ux, set);
            uy = remap_component_flat(P, NX, NY, &prev, &uy, set);
            let mesh = Mesh2d::cartesian_refined(P, NX, NY, [0.0, 1.0], [0.0, 1.0], set);
            last_ne = mesh.n_elements();
            if device {
                device_segment(&mesh, &mut ux, &mut uy, *nsteps)?;
            } else {
                host_segment(&mesh, &mut ux, &mut uy, *nsteps)?;
            }
            prev = set.clone();
        }
        Ok((ux, uy, last_ne))
    };

    let (hux, huy, ne) = run(false)?;
    let (dux, duy, _) = run(true)?;
    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };
    let umax = hux.iter().zip(&huy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));
    let (ru, rv) = (rel(&dux, &hux), rel(&duy, &huy));
    println!("  final mesh {ne} elems, umax={umax:.4}  rel ux={ru:.3e}  rel uy={rv:.3e}");
    if umax.is_finite() && umax > 1e-3 && ru < 1e-6 && rv < 1e-6 {
        println!("OK: device-resident adaptive NS matches the host-orchestrated adaptive trajectory.");
        Ok(())
    } else {
        eprintln!("FAIL: device-resident adaptive run diverges from host.");
        std::process::exit(1);
    }
}

fn mesh_nn() -> usize {
    (P + 1) * (P + 1)
}
