//! Stage 3f: validate **on-device 2:1 balance**. We build a deliberately UNBALANCED masked mesh (a
//! level-0 cell ends up face-adjacent to a level-2 cell), then check:
//!   (1) one device balance-flag pass (`device_balance_flags`, via the `pos2slot` positional index)
//!       flags exactly the slots the host oracle `GpuAmrMesh::balance_refine_flags` does;
//!   (2) the full device balance loop (`device_balance`) lands on the SAME active set as the host
//!       balance loop, and the result is genuinely 2:1-balanced (no remaining flags).
//! The host oracle is independently-written coordinate logic (host AMR proper has no multi-level
//! balance), so this is a real cross-check, not device-vs-device.
//! Run: cargo oxide run --bin amr-balance-device-check

use gale_gpu::amr_mesh::GpuAmrMesh;
use gale_gpu::operators::amr::{device_balance, device_balance_flags};

/// Build an unbalanced mesh: refine base (2,2) → a child → forcing a level-0 / level-2 adjacency.
fn make_unbalanced() -> GpuAmrMesh {
    let mut m = GpuAmrMesh::new(4, 4, 2);
    let s = m.slot_at(0, 2, 2).unwrap();
    assert!(m.refine(s)); // base (2,2) → level-1 children
    let s = m.slot_at(1, 4, 4).unwrap();
    assert!(m.refine(s)); // child (1,4,4) → level-2 children; (2,8,8) now abuts level-0 base (1,2)
    m
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stage 3f: device 2:1 balance vs host oracle — 4×4, l_max=2 ===");

    // ---- (1) one-pass flag agreement ----
    let m = make_unbalanced();
    let host_flags: Vec<usize> = m.balance_refine_flags();
    let dev_flags = device_balance_flags(&m)?;
    let dev_slots: Vec<usize> = (0..m.cap).filter(|&s| dev_flags[s] == 1).collect();
    let mut hf = host_flags.clone();
    hf.sort_unstable();
    let mut df = dev_slots.clone();
    df.sort_unstable();
    let ok1 = hf == df;
    println!("  (1) balance-flag pass: host flags {host_flags:?}, device flags {dev_slots:?} — {}", if ok1 { "MATCH" } else { "MISMATCH" });

    // ---- (2) full balance loop: device vs host, then confirm balanced ----
    let mut hm = make_unbalanced();
    loop {
        let f = hm.balance_refine_flags();
        if f.is_empty() {
            break;
        }
        for s in f {
            hm.refine(s);
        }
    }
    let mut dm = make_unbalanced();
    let passes = device_balance(&mut dm)?;

    let (ha, da) = (hm.active_cells(), dm.active_cells());
    let ok2 = ha == da;
    let still_unbal = device_balance_flags(&dm)?.iter().filter(|&&f| f == 1).count();
    println!("  (2) full balance: host {} active, device {} active ({passes} device passes) — {}", ha.len(), da.len(), if ok2 { "MATCH" } else { "MISMATCH" });
    println!("      post-balance remaining unbalanced leaves: {still_unbal} (must be 0)");

    if ok1 && ok2 && still_unbal == 0 {
        println!("OK: device 2:1 balance reproduces the host balance loop and yields a balanced mesh.");
        Ok(())
    } else {
        eprintln!("FAIL: device 2:1 balance diverges from host.");
        std::process::exit(1);
    }
}
