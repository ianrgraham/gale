//! Stage 3f: validate **on-device connectivity rebuild** — `device_connectivity` fills the face/mortar
//! neighbour arrays (`block_nbr`/`block_nbr2`/`block_nbr_kind`) on the GPU via the `pos2slot` index,
//! exercising every face kind (same / coarser / finer / boundary). Oracle = host
//! `GpuAmrMesh::build_neighbors`. Both run on the IDENTICALLY-constructed mesh (same refine calls ⇒
//! same slot ids) so the arrays must match element-for-element over active leaves. This is the mortar
//! list the device non-conforming operator consumes — now produced with no host loop.
//! Run: cargo oxide run --bin amr-connectivity-device-check

use gale_gpu::amr_mesh::GpuAmrMesh;
use gale_gpu::operators::amr::device_connectivity;

/// A balanced multi-level mesh with same/coarser/finer interfaces.
fn make_mesh() -> GpuAmrMesh {
    let mut m = GpuAmrMesh::new(4, 4, 2);
    for &(l, ix, iy) in &[(0u32, 1u32, 1u32), (0, 2, 2)] {
        let s = m.slot_at(l, ix, iy).unwrap();
        m.refine(s);
    }
    // refine a child of (1,1) → introduces a level-2 region (finer interfaces)
    let s = m.slot_at(1, 3, 3).unwrap();
    m.refine(s);
    // restore 2:1 balance via the host oracle loop
    loop {
        let f = m.balance_refine_flags();
        if f.is_empty() {
            break;
        }
        for s in f {
            m.refine(s);
        }
    }
    m
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stage 3f: device connectivity rebuild vs host build_neighbors — 4×4, l_max=2 ===");

    let mut hm = make_mesh();
    hm.build_neighbors();

    let mut dm = make_mesh();
    device_connectivity(&mut dm)?;

    // Compare over active leaves, all 4 faces.
    let mut mism = 0usize;
    let mut faces = 0usize;
    let (mut n_same, mut n_coarse, mut n_fine, mut n_bdry) = (0, 0, 0, 0);
    for s in 0..hm.cap {
        if hm.active[s] != 1 {
            continue;
        }
        for d in 0..4 {
            let o = s * 4 + d;
            faces += 1;
            match hm.block_nbr_kind[o] {
                1 => n_same += 1,
                2 => n_coarse += 1,
                3 => n_fine += 1,
                _ => n_bdry += 1,
            }
            if hm.block_nbr_kind[o] != dm.block_nbr_kind[o]
                || hm.block_nbr[o] != dm.block_nbr[o]
                || hm.block_nbr2[o] != dm.block_nbr2[o]
            {
                mism += 1;
                if mism <= 5 {
                    eprintln!(
                        "  MISMATCH slot {s} dir {d}: host (kind {},nbr {},nbr2 {}) device (kind {},nbr {},nbr2 {})",
                        hm.block_nbr_kind[o], hm.block_nbr[o], hm.block_nbr2[o], dm.block_nbr_kind[o], dm.block_nbr[o], dm.block_nbr2[o]
                    );
                }
            }
        }
    }
    println!("  {} active leaves, {faces} faces ({n_same} same / {n_coarse} coarser / {n_fine} finer / {n_bdry} boundary), {mism} mismatches", hm.active_cells().len());

    if mism == 0 && n_fine > 0 && n_coarse > 0 {
        println!("OK: device connectivity matches host build_neighbors across all face kinds.");
        Ok(())
    } else {
        eprintln!("FAIL: device connectivity diverges from host (or didn't exercise all kinds).");
        std::process::exit(1);
    }
}
