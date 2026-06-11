//! Multi-GPU gating measurement (research §7): PCIe P2P round-trip latency between the two
//! Titan Vs + single-GPU finest-level matvec time vs DoF ⇒ the crossover DoF where a
//! distributed matvec could hide the halo exchange. Run BEFORE building the distributed solver.
//!   cargo oxide build --arch sm_70   (then run target/release/pcie-crossover)
//! Env: MG_P (default 4), MG_REPS (default 200).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p: usize = std::env::var("MG_P").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let reps: u32 = std::env::var("MG_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    gale_gpu::pcie_crossover(p, reps)
}
