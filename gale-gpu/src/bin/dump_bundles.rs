//! Diagnostic: list the embedded CUDA artifact bundles in this binary — name, target arch, and which
//! payload kinds are present (Cubin/Ptx/NvvmIr/Ltoir) — and dump the first loadable payload to /tmp so
//! `cuobjdump` can report whether it's real SASS (sm_NN) or virtual PTX (compute_NN). Used to diagnose
//! the ncu/compute-sanitizer `cuModuleLoadData` 209 (CUPTI breaks load-time PTX JIT).
//! Run: cargo oxide run --bin dump-bundles

use cuda_core::embedded::{artifact_bundles_from_binary_path, artifact_bundles_from_current_exe, ArtifactPayloadKind as K};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Inspect another binary's bundle if a path is given (this bin has no kernels of its own).
    let bundles = match std::env::args().nth(1) {
        Some(path) => artifact_bundles_from_binary_path(&path)?,
        None => artifact_bundles_from_current_exe()?,
    };
    println!("{} embedded bundle(s)", bundles.len());
    for (i, b) in bundles.iter().enumerate() {
        let kinds: Vec<&str> = [
            (K::Cubin, "Cubin"),
            (K::Ptx, "Ptx"),
            (K::NvvmIr, "NvvmIr"),
            (K::Ltoir, "Ltoir"),
        ]
        .iter()
        .filter_map(|(k, n)| b.payload(*k).map(|p| {
            // stash sizes
            Box::leak(format!("{n}({} B)", p.len()).into_boxed_str()) as &str
        }))
        .collect();
        println!("  [{i}] name={:?} target={:?} payloads=[{}]", b.name, b.target, kinds.join(", "));
        if i == 0 {
            if let Some(c) = b.payload(K::Cubin) {
                std::fs::write("/tmp/bundle0.cubin", c)?;
                println!("      wrote /tmp/bundle0.cubin ({} B)", c.len());
            }
            if let Some(p) = b.payload(K::Ptx) {
                std::fs::write("/tmp/bundle0.ptx", p)?;
                println!("      wrote /tmp/bundle0.ptx ({} B)", p.len());
            }
        }
    }
    Ok(())
}
