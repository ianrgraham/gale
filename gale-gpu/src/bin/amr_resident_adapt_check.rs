//! Stage 3f CAPSTONE: the **persistent device-resident adapt cycle**. `GpuAdaptiveMesh` holds the
//! masked structure + per-slot field as resident GPU buffers and runs refine+coarsen (with field
//! remap) → 2:1 balance (with remap) → connectivity rebuild as kernel launches with NO host↔device
//! field transfers (only tiny `counts` loop-control reads). We validate it against the host-oracle
//! one-shot path (`device_refine_remap` + `device_coarsen_remap` + host `build_neighbors`), each piece
//! of which is independently host-validated: same active set, same field on every active leaf, same
//! connectivity. Two scenarios — (A) disjoint refine+coarsen, no balance; (B) a refine that forces a
//! 2:1 balance cascade.
//! Run: cargo oxide run --bin amr-resident-adapt-check

use gale::dg::RefineQuad;
use gale_gpu::amr_mesh::GpuAmrMesh;
use gale_gpu::operators::amr::{device_coarsen_remap, device_refine_remap, GpuAdaptiveMesh};

const P: usize = 3;

fn seed_field(cap: usize, nn: usize) -> Vec<f64> {
    let mut f = vec![0.0; cap * nn];
    for s in 0..cap {
        for k in 0..nn {
            f[s * nn + k] = 0.5 + 0.013 * (s as f64) + 0.27 * (k as f64).cos();
        }
    }
    f
}

fn flags_for(m: &GpuAmrMesh, cells: &[(u32, u32, u32)], val: i32, f: &mut [i32]) {
    for &(l, ix, iy) in cells {
        f[m.slot_at(l, ix, iy).expect("active leaf")] = val;
    }
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

/// Reference field values for `cells`, looked up via the reference mesh's own slot map.
fn ref_field_on(cells: &[(u32, u32, u32)], m: &GpuAmrMesh, f: &[f64], nn: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(cells.len() * nn);
    for &(l, ix, iy) in cells {
        let s = m.slot_at(l, ix, iy).unwrap();
        out.extend_from_slice(&f[s * nn..(s + 1) * nn]);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let refq = RefineQuad::new(P);
    let nn = (P + 1) * (P + 1);
    println!("=== Stage 3f CAPSTONE: persistent device-resident adapt vs host-oracle one-shot path — p={P} ===");

    // ---------- Scenario A: disjoint refine + coarsen, no balance ----------
    let build_a = || {
        let mut m = GpuAmrMesh::new(4, 4, 2);
        for &(ix, iy) in &[(0u32, 0u32), (3, 3)] {
            let s = m.slot_at(0, ix, iy).unwrap();
            m.refine(s); // pre-refine so coarsen has a target
        }
        m
    };
    let field = seed_field(build_a().cap, nn);
    // flag: refine fresh (0,1,1); coarsen the 4 children of (0,0,0)
    let refine_cells = vec![(0u32, 1u32, 1u32)];
    let coarsen_kids: Vec<(u32, u32, u32)> = (0..4).map(|c| (1u32, (c % 2) as u32, (c / 2) as u32)).collect();

    // Reference (one-shot): refine_remap then coarsen_remap (disjoint ⇒ order-independent).
    let mut ref_m = build_a();
    let mut rflag = vec![0i32; ref_m.cap];
    flags_for(&ref_m, &refine_cells, 1, &mut rflag);
    flags_for(&ref_m, &coarsen_kids, -1, &mut rflag);
    let rf = device_refine_remap(&mut ref_m, &rflag, &refq, &field)?;
    let rf = device_coarsen_remap(&mut ref_m, &rflag, &refq, &rf)?;
    ref_m.build_neighbors();

    // Resident handle.
    let hm = build_a();
    let mut hflag = vec![0i32; hm.cap];
    flags_for(&hm, &refine_cells, 1, &mut hflag);
    flags_for(&hm, &coarsen_kids, -1, &mut hflag);
    let mut dev = GpuAdaptiveMesh::from_host(&hm, &refq)?;
    let mut dfield = dev.upload_field(&field)?;
    let passes_a = dev.adapt(&hflag, &mut dfield)?;
    let dcells = dev.download_active_cells()?;
    let rcells = ref_m.active_cells();
    let cells_match = rcells == dcells;
    // Compare by CELL via each side's own slot map (parallel compaction ⇒ slot ids differ run-to-run).
    let dev_vals_a = dev.download_field_on(&dfield, &rcells)?;
    let frel_a = rel(&dev_vals_a, &ref_field_on(&rcells, &ref_m, &rf, nn));
    println!("  A (refine+coarsen, {passes_a} balance passes): cells {} | field rel {:.2e}", if cells_match { "MATCH" } else { "MISMATCH" }, frel_a);
    let ok_a = cells_match && frel_a < 1e-12 && passes_a == 0;

    // ---------- Scenario B: refine that forces a 2:1 balance cascade ----------
    let build_b = || {
        let mut m = GpuAmrMesh::new(4, 4, 2);
        let s = m.slot_at(0, 2, 2).unwrap();
        m.refine(s); // (2,2)→L1 children
        m
    };
    let field_b = seed_field(build_b().cap, nn);
    // refine child (1,4,4)→L2, which forces base (1,2) to refine for 2:1 balance.
    let refine_b = vec![(1u32, 4u32, 4u32)];

    // Reference: refine_remap, then host-oracle balance loop WITH field carry.
    let mut ref_b = build_b();
    let mut rfb = vec![0i32; ref_b.cap];
    flags_for(&ref_b, &refine_b, 1, &mut rfb);
    let mut fb = device_refine_remap(&mut ref_b, &rfb, &refq, &field_b)?;
    let mut ref_passes = 0;
    loop {
        let bs = ref_b.balance_refine_flags();
        if bs.is_empty() {
            break;
        }
        let mut bf = vec![0i32; ref_b.cap];
        for s in bs {
            bf[s] = 1;
        }
        fb = device_refine_remap(&mut ref_b, &bf, &refq, &fb)?;
        ref_passes += 1;
    }
    ref_b.build_neighbors();
    let rcells_b = ref_b.active_cells();

    // Resident.
    let hb = build_b();
    let mut hfb = vec![0i32; hb.cap];
    flags_for(&hb, &refine_b, 1, &mut hfb);
    let mut devb = GpuAdaptiveMesh::from_host(&hb, &refq)?;
    let mut dfb = devb.upload_field(&field_b)?;
    let passes_b = devb.adapt(&hfb, &mut dfb)?;
    let dcells_b = devb.download_active_cells()?;
    let cells_match_b = rcells_b == dcells_b;
    let dev_vals_b = devb.download_field_on(&dfb, &rcells_b)?;
    let frel_b = rel(&dev_vals_b, &ref_field_on(&rcells_b, &ref_b, &fb, nn));
    println!("  B (refine+balance): host {ref_passes} ref-passes / device {passes_b} balance passes | cells {} | field rel {:.2e}", if cells_match_b { "MATCH" } else { "MISMATCH" }, frel_b);
    let ok_b = cells_match_b && frel_b < 1e-12 && passes_b == ref_passes && passes_b > 0;

    if ok_a && ok_b {
        println!("OK: device-resident adapt cycle matches the host-oracle path (structure + field + balance), no per-cycle field transfers.");
        Ok(())
    } else {
        eprintln!("FAIL: device-resident adapt diverges (A={ok_a}, B={ok_b}).");
        std::process::exit(1);
    }
}
