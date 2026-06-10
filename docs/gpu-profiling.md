# GPU profiling (Nsight) for gale-gpu

Nsight works on the cuda-oxide–generated binaries (CUPTI traces any CUDA process, regardless of
how the cubin was produced; the `#[kernel]` export names show up directly). Two tools:

- **Nsight Systems (`nsys`)** — timeline + per-kernel/per-API totals. **No special permission.**
  Answers "where does wall-clock go: GPU kernels vs host launch vs sync vs copies".
- **Nsight Compute (`ncu`)** — per-kernel HW counters (occupancy, %BW, roofline). **Needs the
  profiling-counter permission** (see below).

Validated on: Titan V (sm_70), driver 575, nsys 2025.1, ncu 2025.2.

## Build gotcha

`cargo oxide run` auto-detects the device arch (sm_70 here), but **`cargo oxide build` defaults to
a different arch** → the standalone binary dies with `DriverError(209, "no kernel image…")`. For a
binary you'll run yourself (under a profiler), build with the arch explicit:

```
cargo oxide build --arch sm_70 mg-profile
```

## The profiling target

`mg-profile` (src/bin/mg_profile.rs) does one warmup + one profiled MG-PCG solve — a clean single
solve, not the full `mg-wallclock-bench` sweep. Config via env: `MG_GRID` (default 64), `MG_P` (4),
`MG_OP` (`pressure` | `velocity`).

## nsys (timeline + totals) — no permission needed

```
cargo oxide build --arch sm_70 mg-profile
nsys profile --cuda-event-trace=false --force-overwrite true -o /tmp/mg \
    target/release/mg-profile
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum /tmp/mg.nsys-rep
```

- `cuda_gpu_kern_sum` — per-kernel GPU time (which kernel is the long pole).
- `cuda_api_sum` — host-side API time: `cuLaunchKernel` (launch overhead), `cuMemcpyDtoHAsync`
  (downloads + residual polls), `cuStreamSynchronize` (host waits), `cuMemAllocAsync` (scratch).
- Open `/tmp/mg.nsys-rep` in the Nsight Systems GUI for the visual timeline (gaps between kernels =
  GPU idle waiting on the host).

### First findings (64² p=4 pressure, two solves)

- GPU kernels: `operator` 40% + `gradient` 22% = **62%** (matvec is the long pole); reductions
  (`dot_partial`+`reduce_scalar`) ~12%; `jacobi` 8%; the h-transfers (`h_prolong`/`h_restrict`)
  only ~2.5% — h-coarsening is cheap.
- Host API: `cuStreamSynchronize` only **0.2%** — the per-iter sync the on-device-scalar work
  removed is genuinely gone. But `cuLaunchKernel` is **68%** (~10.8k launches/solve, ~4 µs each):
  host launch issue (~44 ms/solve) now *exceeds* GPU kernel time (~16 ms/solve) ⇒ the solve is
  **launch-bound**. `cuMemAllocAsync` ~9% = the per-solve V-cycle scratch (re-allocated each solve).

⇒ next-pass levers: cut launch count (CUDA graphs to replay the fixed V-cycle kernel sequence;
or fuse gradient+operator), and persist the V-cycle scratch in the handle.

## ncu (per-kernel HW counters) — needs profiling permission

`ncu` reads GPU performance counters, gated by the driver. Default here is restricted
(`cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` → `1`), so non-root `ncu` fails with
`ERR_NVGPUCTRPERM`. Two ways to enable:

- **Per-run (simplest):** run as root — `sudo ncu …`.
- **Permanent (non-root ncu):** allow profiling for all users, then reload the driver / reboot:
  ```
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
  sudo update-initramfs -u && sudo reboot     # or unload+reload the nvidia modules
  ```

Bound the work — `ncu` replays each kernel many times for counters, so without limits it would
replay all ~10k launches. Use a kernel filter + launch count (the warmup solve runs first, so the
profiled kernels are later — skip with `-s` if needed):

```
sudo ncu --set full -k 'regex:gradient|operator|dot_partial' -c 12 -o /tmp/mg_ncu \
    target/release/mg-profile
ncu-ui /tmp/mg_ncu.ncu-rep      # roofline, occupancy, memory throughput per kernel
```

`--set full` gives the roofline + memory workload analysis; `--set basic` is faster. The Titan V
peaks: FP64 ≈ 6.9 TFLOP/s, HBM2 ≈ 652.8 GB/s (ridge AI ≈ 10.6 FLOP/byte) — most DG kernels are
memory-bound (see docs/gpu-roofline-and-optimization.md).
