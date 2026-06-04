//! Reproducer / regression gate for cuda-oxide issue #58 (writes through
//! `get_mut()` into a nested array element are silently dropped).
//!
//! Why gale cares: matrix-free DG kernels write per-element DOF arrays — exactly
//! the `DisjointSlice<[T; N]>` + inner-element-write pattern #58 reports broken.
//!
//! This probe also tests the hypothesis (see docs/cuda-oxide-codegen-notes.md)
//! that #58 is **unrelated** to the typed-pointer bitcast bug: both kernels here
//! are libdevice-free, so they take the PTX/`llc` path and never touch the
//! dialect-llvm NVVM-IR text exporter where the bitcast bug lives.
//!
//! Two kernels write the same logical result two ways:
//!   A. `a.get_mut(i).map(|e| *e = V)`  — the #58-suspect form.
//!   B. `for e in a { *e = V }`          — the form #58 says works.
//! If A leaves zeros and B writes V, #58 reproduces and is a place-lowering bug
//! independent of the (libdevice-only) text-exporter bitcast issue.
//!
//! Run: cargo oxide run --bin probe-nested-write

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const SIZE: usize = 4;
const NELEM: usize = 64;
const V: f32 = 42.0;

#[cuda_module]
mod kernels {
    use super::*;

    /// #58-suspect form: write through `get_mut` into a nested array element.
    #[kernel]
    pub fn via_get_mut(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        if let Some(a) = out.get_mut(idx) {
            for i in 0..SIZE {
                a.get_mut(i).map(|e| *e = V);
            }
        }
    }

    /// Control form #58 reports as working: iterate `&mut` over the array.
    #[kernel]
    pub fn via_iter(mut out: DisjointSlice<[f32; SIZE]>) {
        let idx = thread::index_1d();
        if let Some(a) = out.get_mut(idx) {
            for e in a {
                *e = V;
            }
        }
    }
}

fn check(label: &str, got: &[[f32; SIZE]]) -> bool {
    let ok = got.iter().all(|row| row.iter().all(|&x| x == V));
    let written: usize = got.iter().flatten().filter(|&&x| x == V).count();
    println!(
        "  [{}] {label}: {}/{} elements written to {V}",
        if ok { "PASS" } else { "FAIL" },
        written,
        NELEM * SIZE
    );
    ok
}

fn main() {
    println!("=== #58 reproducer: nested-array writes through get_mut vs iter ===\n");
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("module load");
    let cfg = LaunchConfig::for_num_elems(NELEM as u32);

    let zeros = vec![[0.0f32; SIZE]; NELEM];

    let mut a = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    module.via_get_mut(&stream, cfg, &mut a).expect("launch A");
    let got_a = a.to_host_vec(&stream).unwrap();

    let mut b = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    module.via_iter(&stream, cfg, &mut b).expect("launch B");
    let got_b = b.to_host_vec(&stream).unwrap();

    let a_ok = check("get_mut(i)", &got_a);
    let b_ok = check("for e in a", &got_b);

    println!();
    if !a_ok && b_ok {
        println!("→ #58 REPRODUCES: get_mut writes dropped, iter form works.");
        println!("  Both are libdevice-free (PTX path), so this is a place/value");
        println!("  lowering bug, NOT the typed-pointer text-exporter bitcast bug.");
    } else if a_ok && b_ok {
        println!("→ #58 does NOT reproduce here (both forms wrote correctly).");
    } else {
        println!("→ Unexpected result — investigate (b_ok={b_ok}, a_ok={a_ok}).");
    }
}
