//! Stage 3f: validate **device field remap across adapt** — `device_refine_remap` /
//! `device_coarsen_remap` carry a per-slot field through the structural adapt on the GPU. Oracles:
//!   (1) refine: each child slot's remapped field equals host `RefineQuad::prolong(parent, cx, cy)`
//!       bit-for-bit (exact for degree ≤ p);
//!   (2) coarsen: the parent slot's remapped field equals host `RefineQuad::restrict([children])`,
//!       and the cell average is conserved through refine→coarsen round-trip.
//! Run: cargo oxide run --bin amr-remap-device-check

use gale::dg::RefineQuad;
use gale_gpu::amr_mesh::GpuAmrMesh;
use gale_gpu::operators::amr::{device_coarsen_remap, device_refine_remap};

const P: usize = 3;

fn flags_for(m: &GpuAmrMesh, cells: &[(u32, u32, u32)], val: i32) -> Vec<i32> {
    let mut f = vec![0i32; m.cap];
    for &(l, ix, iy) in cells {
        f[m.slot_at(l, ix, iy).expect("active leaf")] = val;
    }
    f
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
    (d / n).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let refq = RefineQuad::new(P);
    let nn = (P + 1) * (P + 1);
    println!("=== Stage 3f: device field remap (prolong/restrict) vs host RefineQuad — p={P} ===");

    // ---- (1) refine remap ----
    let mut m = GpuAmrMesh::new(4, 4, 2);
    // Seed every slot's field with a per-slot pattern (only refined parents' fields are consumed).
    let mut field = vec![0.0f64; m.cap * nn];
    for s in 0..m.cap {
        for k in 0..nn {
            field[s * nn + k] = (s as f64) * 0.01 + (k as f64).sin();
        }
    }
    let r1: Vec<(u32, u32, u32)> = vec![(0, 1, 1), (0, 2, 2), (0, 3, 0)];
    // Capture parent fields BEFORE adapt.
    let parent_fields: Vec<(usize, Vec<f64>)> = r1
        .iter()
        .map(|&(l, ix, iy)| {
            let s = m.slot_at(l, ix, iy).unwrap();
            (s, field[s * nn..(s + 1) * nn].to_vec())
        })
        .collect();
    let f1 = flags_for(&m, &r1, 1);
    let out = device_refine_remap(&mut m, &f1, &refq, &field)?;

    let mut worst = 0.0f64;
    for (r, pf) in &parent_fields {
        for c in 0..4 {
            let cs = m.children[r * 4 + c] as usize;
            let (cx, cy) = (c % 2, c / 2);
            let host_child = refq.prolong(pf, cx, cy);
            let dev_child = &out[cs * nn..(cs + 1) * nn];
            worst = worst.max(rel(dev_child, &host_child));
        }
    }
    let ok1 = worst < 1e-12;
    println!("  (1) refine remap: {} parents prolonged, worst rel vs host prolong = {worst:.2e} — {}", parent_fields.len(), if ok1 { "MATCH" } else { "MISMATCH" });

    // ---- (2) coarsen remap + conservation round-trip ----
    // m now has level-1 children of the r1 cells. Coarsen (1,1)'s 4 children back.
    let parent_cell = (0u32, 1u32, 1u32);
    let kids: Vec<(u32, u32, u32)> = (0..4).map(|c| (1u32, 2 * parent_cell.1 + (c % 2) as u32, 2 * parent_cell.2 + (c / 2) as u32)).collect();
    // Gather child slots + fields (post-refine) for the host restrict oracle.
    let child_slots: Vec<usize> = kids.iter().map(|&(l, ix, iy)| m.slot_at(l, ix, iy).unwrap()).collect();
    let child_fields: [Vec<f64>; 4] = std::array::from_fn(|c| out[child_slots[c] * nn..(child_slots[c] + 1) * nn].to_vec());
    let host_parent = refq.restrict(&child_fields);

    let f2 = flags_for(&m, &kids, -1);
    let out2 = device_coarsen_remap(&mut m, &f2, &refq, &out)?;
    let pslot = m.slot_at(parent_cell.0, parent_cell.1, parent_cell.2).unwrap();
    let dev_parent = &out2[pslot * nn..(pslot + 1) * nn];
    let r_restrict = rel(dev_parent, &host_parent);

    // Conservation: ∫ over the parent cell == Σ ∫ over the 4 children (reference cell-average, equal weights).
    let w = refq.weights();
    let cell_int = |f: &[f64]| -> f64 {
        let n = P + 1;
        let mut s = 0.0;
        for j in 0..n {
            for i in 0..n {
                s += w[i] * w[j] * f[i + j * n];
            }
        }
        s
    };
    let parent_int = cell_int(dev_parent);
    let kids_int: f64 = child_fields.iter().map(|f| 0.25 * cell_int(f)).sum(); // each child is 1/4 area
    let cons = (parent_int - kids_int).abs() / parent_int.abs().max(1e-300);
    let ok2 = r_restrict < 1e-12 && cons < 1e-12;
    println!("  (2) coarsen remap: rel vs host restrict = {r_restrict:.2e}, cell-average conservation = {cons:.2e} — {}", if ok2 { "MATCH" } else { "MISMATCH" });

    if ok1 && ok2 {
        println!("OK: device field remap matches host prolong/restrict and conserves the cell average.");
        Ok(())
    } else {
        eprintln!("FAIL: device field remap diverges from host.");
        std::process::exit(1);
    }
}
