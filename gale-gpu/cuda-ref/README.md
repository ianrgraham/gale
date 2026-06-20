# cuda-ref — C++/CUDA reference prototypes for the DG-SIPG GPU work

Standalone `nvcc`/`g++` ports of the `gale-gpu` kernels and solvers. We prototype an algorithm or pin a
hardware bound here — independently of the Rust (cuda-oxide) codegen — before porting it into the
production crate. The `.cu`/`.cpp` **source is committed; compiled binaries are NOT** — build into
`out/` (gitignored; see `.gitignore`).

Prototypes:
- `matvec.cu` (+ `lean-matvec.cu`, `transpose-matvec.cu`) — the DG-SIPG matvec **bandwidth oracle**:
  proves whether the Rust matvec is codegen-limited or genuinely bandwidth-bound.
- `mgpcg.cu` / `mgpcg.cpp` — the GPU / CPU **MG-PCG prototype**: validated the Chebyshev hp-multigrid
  smoother (iteration counts + ms/solve, and the FP32-smoother iteration-hold test) before the
  cuda-oxide port.

## Build (into `out/`, kept out of git)

    mkdir -p out
    # matvec bandwidth oracle
    nvcc -O3 -arch=sm_70 --extended-lambda -Wno-deprecated-gpu-targets matvec.cu -o out/matvec
    ./out/matvec 256
    # MG-PCG prototype (GPU + CPU)
    nvcc -O3 -arch=sm_70 --default-stream per-thread -Wno-deprecated-gpu-targets mgpcg.cu -o out/mgpcg_gpu
    g++  -O2 -std=c++17 mgpcg.cpp -o out/mgpcg

`-arch=sm_70` = Titan V. `ncu` works on these (plain launches, not while-graphs), e.g.

    ncu --kernel-name 'regex:gradient|operator|op_cheby' --launch-count 4 --section SpeedOfLight ./out/matvec 256

## Key result

Titan V, 256² p=3: C++ `gradient` 46.7 µs @ 76% DRAM (bandwidth-bound) ≈ Rust `gradient` 46.8 µs
(`MG_P=3 MG_GRID=256 cargo oxide run --bin op_profile`) — the cuda-oxide codegen matches `nvcc`; the
matvec already saturates HBM2. The MG-PCG prototype is what established that the Chebyshev win is
*structural* (h- vs p-multigrid). See `docs/gale-v2-refactor.md` (§4 benchmark ledger) and
`docs/plan-matvec-threading-redesign.md`.
