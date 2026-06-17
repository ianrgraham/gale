//! Stage 3f (first piece): validate the on-device **adapt-flag** kernel — the GPU adapt DECISION.
//! Two checks:
//!   (1) Pure decision logic over a hand-built `(se, level)` grid that exercises all three branches
//!       (refine/coarsen/keep) at every level boundary, vs the trivial host rule.
//!   (2) End-to-end compose `smoothness_se_gpu` → `amr_flag_gpu` on a real field, vs host
//!       `SmoothnessIndicator::indicator` + the same threshold rule. This is the device path that the
//!       self-driving adapt loop runs (indicator + flag with no host readback of the field).
//! Run: cargo oxide run --bin amr-flag-check

use gale::dg::{Mesh2d, SmoothnessIndicator};
use gale_gpu::operators::amr::{amr_flag_gpu, smoothness_se_gpu};

/// Host reference for the flag decision (mirror of the kernel).
fn host_flag(se: f64, level: u32, refine_thr: f64, coarsen_thr: f64, l_max: u32) -> i32 {
    if se > refine_thr && level < l_max {
        1
    } else if se < coarsen_thr && level > 0 {
        -1
    } else {
        0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (refine_thr, coarsen_thr, l_max) = (1e-4, 1e-8, 2u32);

    // ---- (1) decision logic over all branches ----
    // Sweep se across {below coarsen, between, above refine} × levels {0..=l_max}.
    let se_vals = [1e-12, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2, 1.0];
    let mut se = Vec::new();
    let mut lvl = Vec::new();
    for l in 0..=l_max {
        for &s in &se_vals {
            se.push(s);
            lvl.push(l);
        }
    }
    let gflag = amr_flag_gpu(&se, &lvl, refine_thr, coarsen_thr, l_max)?;
    let mut mism = 0usize;
    for i in 0..se.len() {
        let h = host_flag(se[i], lvl[i], refine_thr, coarsen_thr, l_max);
        if h != gflag[i] {
            mism += 1;
            eprintln!("  MISMATCH se={:.1e} lvl={} host={} gpu={}", se[i], lvl[i], h, gflag[i]);
        }
    }
    let (nref, ncrs, nkeep) = gflag.iter().fold((0, 0, 0), |(r, c, k), &f| match f {
        1 => (r + 1, c, k),
        -1 => (r, c + 1, k),
        _ => (r, c, k + 1),
    });
    println!("(1) decision grid: {} cases, {nref} refine / {ncrs} coarsen / {nkeep} keep, {mism} mismatches", se.len());

    // ---- (2) end-to-end smoothness_se_gpu → amr_flag_gpu vs host indicator+rule ----
    let (p, nx, ny) = (3usize, 6usize, 6usize);
    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    // A field with a sharp localized feature → some elements smooth, some rough.
    let mut field = vec![0.0; ne * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            // steep tanh front along x=0.5 + a localized gaussian bump
            field[e * nn + k] =
                (40.0_f64 * (x - 0.5)).tanh() + (-200.0_f64 * ((x - 0.3).powi(2) + (y - 0.7).powi(2))).exp();
        }
    }
    let si = SmoothnessIndicator::new(p);
    let se_gpu = smoothness_se_gpu(&mesh, &field, &si)?;
    // Per-element level: pretend a checkerboard of levels 0/1 so coarsen can fire on smooth cells.
    let level: Vec<u32> = (0..ne).map(|e| (e % 2) as u32).collect();
    let gflag = amr_flag_gpu(&se_gpu, &level, refine_thr, coarsen_thr, l_max)?;

    let mut mism2 = 0usize;
    let mut worst_se = 0.0f64;
    for e in 0..ne {
        let se_host = si.indicator(&field[e * nn..(e + 1) * nn]);
        worst_se = worst_se.max(se_host);
        let h = host_flag(se_host, level[e], refine_thr, coarsen_thr, l_max);
        if h != gflag[e] {
            mism2 += 1;
            eprintln!("  e={e} MISMATCH se_host={se_host:.3e} se_gpu={:.3e} lvl={} host={h} gpu={}", se_gpu[e], level[e], gflag[e]);
        }
    }
    let (nref, ncrs, nkeep) = gflag.iter().fold((0, 0, 0), |(r, c, k), &f| match f {
        1 => (r + 1, c, k),
        -1 => (r, c + 1, k),
        _ => (r, c, k + 1),
    });
    println!("(2) end-to-end on {ne} elems (max Se={worst_se:.2e}): {nref} refine / {ncrs} coarsen / {nkeep} keep, {mism2} mismatches");

    if mism == 0 && mism2 == 0 {
        println!("OK: device adapt-flag matches host decision (logic + end-to-end indicator→flag).");
        Ok(())
    } else {
        eprintln!("FAIL: device adapt-flag diverges from host ({} + {} mismatches).", mism, mism2);
        std::process::exit(1);
    }
}
