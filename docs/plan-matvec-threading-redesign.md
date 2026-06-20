# Plan: matvec/smoother threading-model redesign for HBM2 saturation (Titan V / GV100)

## SESSION 9 (2026-06-19) — memory-prefetch (NVIDIA blog) tested on BOTH paths: doesn't apply (structural)

Tried the NVIDIA memory-prefetching blog technique (grid-stride loop, prefetch element i+1's loads into
registers while computing i) to hide arith's residual L2-hit latency. CLARIFICATION first: "arith reads u
ONCE" = the DRAM MINIMUM (optimal, nothing to work around); prefetch hides LATENCY, not read count.
- `op_transpose_pf` (grid-stride + ue prefetch): 61–103 µs (SLOWER than 56). ue is only 20% of loads (the
  NEIGHBOUR loads are the bulk + register-infeasible to buffer); grid-stride dropped the already-17% occ.
- `operator_fused_arith_pf` (grid-stride + NEIGHBOUR prefetch, the real target): 63.5 µs (SLOWER than 47.7),
  worse with fewer blocks. Bit-exact (4.12e-14).

WHY it can't help here (MEASURED, definitive — the precise mechanism is REGISTER→OCCUPANCY, not "saturation"):
the prefetch buffer (next batch's 8 neighbour values held in regs across the whole batch) raised arith
**56 → 82 registers**, which (Volta is register-limited) HALVED occupancy **51% → 26.6%**. But occupancy IS
arith's latency-hiding mechanism (many warps → scheduler runs another while one waits on an L2 hit), so cutting
it dropped **eligible-warps/sched 1.45 → 0.82** (LESS hiding, not more), DRAM 46→34%, 48→66 µs. The prefetch's
own benefit can't offset it because **arith already hoists ~16 loads/thread ahead of use** (already high MLP),
so extra cross-batch prefetch is redundant — you can't buy more of what you already have, while the register
bill is real. It's the Volkov occupancy↔ILP tradeoff run the WRONG way: arith already sits at a good balance
(ILP from hoisting + 51% occ); prefetch spends occupancy to buy ILP it doesn't need. The blog's win-case is a
kernel that's latency-bound AND register-light AND low-occupancy; arith is latency-bound but neither. arith 44µs stands.

---

## SESSION 8 (2026-06-19) — PROFILED the element-parallel path + measured DRAM bytes: arith is at the DRAM minimum

Gave the element-parallel path the profiling rigor it lacked (ncu SOL + scheduler + source PC-sampling +
DRAM-bytes), and measured actual DRAM traffic for all variants. Two definitive findings:

**(1) `op_transpose` is 100% global-memory-LATENCY-bound at 17% occupancy** (ncu: No-Eligible-warp 84%,
0.23 eligible warps/sched, IPC 0.62/4; source PC-sampling: LDG 85% + STG 15% of stalls, ZERO on
IMAD/I2F/division). The integer division + 64-bit address math I worried about are NOT stalls (stripping
the unnecessary `(long)` casts changed nothing). It's register-capped occupancy starving the latency hiding.

**(2) DRAM-bytes measurement reframes the whole thing — `arith` is at the DRAM MINIMUM:**
| kernel | DRAM read | write | total | note |
|---|---|---|---|---|
| `arith` | **8.40 MB** | 6.38 | **14.77 MB** | reads u EXACTLY ONCE; all neighbour reads L2-cached |
| `op_transpose` | 12.12 MB | 6.43 | 18.55 MB | transposed layout scatters nodes ⇒ WORSE L2 locality, +25% DRAM |
| `op_assembled` | 32.80 MB | 6.36 | 39.16 MB | reloads full neighbours, no reuse |

`arith` reads u = ndof·8 = 8.40 MB = once. ⇒ its real DRAM floor is 14.77 MB / 547 GB/s = **27 µs**, and at
44 µs it runs at **61% of peak BW on its actual traffic** (1.63× its real floor). The remaining 39% is pure
latency (L2-hit neighbour reads ~200 cyc + FP64-dep chains) that occupancy(56%)+ILP can't hide further.

**Verdict: the element-parallel path is not an untapped lever — it's strictly WORSE** (register-bound AND
+25% DRAM from poor L2 locality of the transposed layout). `arith` (node-parallel, [element][node] layout)
reads u once, gets full L2 reuse on neighbours, and is at 61% of peak BW — genuinely near-optimal. Lesson:
measure DRAM bytes (not just a bw_copy floor) to know the real ceiling.

**(3) Built + optimized the 2D-element-TILE (`operator_fused_tiled`, matvec.cu) — the last lever, also net-NEG.**
Node-parallel layout but a block stages a Tx×Ty tile of elements in shared so IN-TILE neighbour faces read
the neighbour edge from SHARED (~30 cyc) instead of L2 (~200 cyc); halo from global; balanced-tree contractions
+ arithmetic neighbours. Correct (4.12e-14). Tile sweep 4×4..16×4: best 4×4 = **71 µs vs arith 47.7** (slower
everywhere). Profile: **Compute(SM)/L1TEX-pipe-bound — L1TEX 63%, DRAM dropped to 31%**, occupancy 47%, 2
barriers. WHY it fails: arith's neighbour reads are ALREADY L2-cached and their latency is ALREADY HIDDEN by
56% occupancy — so "neighbour from shared" fixes a non-binding latency and pays with shared/L1-pipe THROUGHPUT
+ a 2nd barrier + lower occupancy. Staging-in-shared is consistently net-negative for this kernel (this, the
original US-staged op_fused, and op_transpose's L2 penalty all agree).

**FINAL — five mappings, each bound by a different resource, all ≥ arith's 44 µs:**
arith (node-parallel) 44 (latency, 61% peak BW, reads u ONCE) < transposed sum-fact 56 (regs+L2) < tiled 71
(L1TEX-pipe) < op_transpose-variants < assembled 104 (constant-BW). The design space is genuinely exhausted.
**arith is the answer. STOP optimizing the matvec.**

---

## SESSION 7 (2026-06-19) — opposite mapping (1-thread-per-element) confirms the limit from the other side

Tried a fundamentally different mapping to escape the node-parallel shared/barrier/divergent-D walls:
**one-thread-per-element + TRANSPOSED `[node][element]` layout** (`cuda-ref/transpose-matvec.cu`,
`op_transpose`). Each thread does a whole 16-node element in REGISTERS — no shared, no barrier, no
cross-thread LDS; transposed layout makes all loads/stores coalesced (own + neighbours); D read
uniformly ⇒ `__constant__` broadcast WORKS here (it didn't node-parallel). Correct (4.12e-14). RESULT:
**56 µs — register-bound: 164 regs/thread (no spill), 17% occupancy** — slower than `arith` (44 µs).
Holding the element's intermediates (`ue`+`PR`+`PS`+`rf` = 64 doubles = 128 regs) caps occupancy and
the 16× per-thread ILP can't compensate. Block-size sweep flat (56–60 µs).

Also tried the ASSEMBLED local operator (`op_assembled`): out = cA·ue + Σ_dir cB[dir]·nbr, the 16×16
matrices assembled numerically (host replica → unit vectors → cA/cB) in constant memory. It DID solve the
register wall (**164 → 80 regs**, correct 9e-15 interior) — but is **103.8 µs, much slower**: dense 16×16
matvecs are **4× the FLOPs** of sum-factorization (1280 vs ~320 FMAs) AND **1280 constant reads/thread
saturate the constant-cache** (1 broadcast/cycle). Wrong trade.

**KEY — THREE fundamentally different mappings all converge ≥1.4× the BW floor:**
| mapping | time | bound by |
|---|---|---|
| node-parallel sum-factorization (`arith`) | **44 µs** | latency (BEST) |
| element-parallel sum-factorization (`op_transpose`) | 56 µs | registers (164 ⇒ 17% occ) |
| element-parallel assembled stencil (`op_assembled`) | 104 µs | constant-cache BW + 4× FLOPs |

Independent approaches hitting the same ~1.4× ceiling = **definitive: the matvec is at its genuine
practical limit on the Titan V.** `arith` (44 µs, 1.42× floor) is the proven best. The design space is
exhausted (trees, hoist, pipeline, L1, arithmetic-nbr, coarsening, occupancy, constant-D, source-PC-sampling,
1-thread-per-element sum-fact, assembled stencil). **STOP optimizing the matvec** — and note it's not even
the per-step bottleneck (iteration count / Phase E is). cuda-ref/transpose-matvec.cu kept (op_transpose + op_assembled).

---

## SESSION 6 (2026-06-19) — source-level PC-sampling: `arith` is confirmed at its practical limit

Used `ncu --page source` PC-sampling (binary built `-lineinfo`) to attribute stalls to SASS instructions
on `arith`. By op type: **LDS (shared loads) 56%, LDG (global) 37%, STS 5%.** The LDS are dominated by
re-reading the tiny diff matrix `D` ~24×/thread (shared-bytes/wavefront only 26.5% — broadcast-heavy;
bank conflicts modest at 2.5%). HYPOTHESIS: move `D` to `__constant__` (broadcast cache). RESULT:
**FALSIFIED — `operator_fused_cd` = 66.6 µs, SLOWER than arith's 47.7.** Reason: the contraction reads
`D[i*4+k]` with `i` varying per lane (4 distinct rows/warp); **constant memory serializes divergent
addresses** (fast only for uniform broadcast), whereas shared resolves the 4 addresses in one
conflict-free transaction. ⇒ The 56% LDS PC-samples are warps *parked at the load while waiting on the
downstream FP64 dependency* (PC sampling blames the consumer's load, not the real stall) — NOT a
shared-latency bottleneck. Confirms `arith` (D in shared) is at its practical limit; the residual is
inherent FP64-dependency + global-load latency we've already minimized. **DON'T move D to constant/regs.**
(Tooling note: ncu `--page source` gives SASS-level stalls from the CLI; the C++ source-line heatmap needs
the Nsight Compute GUI — open the `-lineinfo` `.ncu-rep` there, NOT the Nsight VSCode *debug* extension.)

---

## SESSION 5 (2026-06-18) — TWO CORRECTIONS: `ns` had a RACE; `arith` (no face_nbr load) is the real best

**(1) `operator_fused_ns` had a latent DATA RACE — its "51 µs / best" was INVALID.** Removing the first
`__syncthreads` (the "no u-staging" win) also removed the barrier that protected the **shared diff matrix
`DS`** (`if(t<nn) sm[t]=d[t]` then read with no barrier). `ns` got lucky (the `face_nbr` loads delayed the
gradient enough); the faster `arith` exposed it (~2% of nodes wrong, nondeterministic, correct under
memcheck). **Lesson: the "skip the first barrier" optimization was a mirage** — fixed correctly (`ns`
reads `d` from global, no DS staging), `ns` = 54–58 µs, i.e. NO gain over `pipe`. The "no-barrier"
speedup *was* the race. (The Rust kernels stage with a barrier ⇒ they are SAFE; this race was only in the
C++ experiment. But it means the earlier "port `ns`" advice was based on a racy kernel.)

**(2) The genuine remaining win: `operator_fused_arith` — compute the neighbour ARITHMETICALLY.** On the
uniform grid the neighbour element is `e±1`/`e±N`, so skip the `face_nbr[]` global load entirely. That
removes a **two-level load-dependency chain** (read face_nbr → derive addr → load neighbour); the
neighbour address is now known immediately. Correct (memcheck-clean, 4.12e-14):

| kernel (CORRECT, epb=6) | time | vs floor (30.7µs) | note |
|---|---|---|---|
| `pipe` | 54.7 µs | 1.78× | balanced trees + pipelined loads + staging barrier |
| `ns` (race fixed) | 53.7 µs | 1.75× | "no first barrier" was the race ⇒ no real gain |
| **`operator_fused_arith`** | **43.6 µs** | **1.42×** | + neighbour via arithmetic (no `face_nbr` load) |

DRAM 46% / Compute 55% / 51% occ — removing `face_nbr` traffic shifted it compute-balanced.

**Corrected session arc: 85 µs (baseline) → 43.6 µs (`arith`) = ~1.95×, 2.8× → 1.42× the BW floor.**
**Port target is now `arith`, NOT `ns`:** balanced-tree contractions + hoisted/pre-barrier-issued
neighbour loads + L1 row/col reads + **arithmetic neighbours** + KEEP the DS/US staging barrier (the race
lesson). GpuPoissonMg is a uniform structured grid so the arithmetic-neighbour assumption holds (the same
assumption `operator` already makes for its closed-form face normals). cuda-ref/matvec.cu has all variants.

---

## SESSION 4 (2026-06-18) — CORRECTION: the matvec was NOT at its floor — 1.6× recovered by reducing data dependencies

Session 3's "near-optimal, don't touch it" conclusion was WRONG (it tested occupancy/blocksize/order
but never tried restructuring the *arithmetic*). Reducing data dependencies + pipelining the memory
gave a real **1.6×** on `operator_fused`, all in the C++ ref (`cuda-ref/matvec.cu`), bit-reproducible
to 4.1e-14 (pure FP reassociation). The key: **nvcc does NOT reassociate FP64 without -ffast-math, so
the dependency chains survive exactly as written — manual restructuring exposes ILP/MLP the compiler
will not.** Three stacked changes, each measured:

| variant (256² p=3, best epb) | time | vs base | vs BW floor (30.7µs) | what changed |
|---|---|---|---|---|
| `operator_fused` (baseline) | 85 µs | — | 2.80× | linear accumulation, loads consumed in place |
| `operator_fused_opt` | 65–72 µs | 1.17–1.25× | 2.1–2.4× | **balanced-tree contractions** `(a+b)+(c+d)` (depth 4→2, attacks `wait`/FP64-dep); **hoist neighbour loads** to regs (MLP); **reuse hoisted q for the jump** (kills a redundant scattered load) |
| `operator_fused_pipe` (epb≈6-8) | 54.7 µs | 1.59× | 1.80× | **+ software-pipeline: issue the scattered neighbour-u loads BEFORE `__syncthreads`** so they fly during the barrier + gradient contraction (attacks `long_scoreboard`/load-latency) |
| **`operator_fused_ns`** (epb=6) | **50.8 µs** | **1.67×** | **1.67×** | **+ NO u-staging / NO first barrier**: a p=3 element's 16 nodes = one 128B L1 line, so read the gradient row/col straight from L1 instead of staging in shared (drops the `US` tile + one `__syncthreads`) |

Profiling tracked the bottleneck moving: baseline stalls `long_scoreboard` 27% + `wait` 22%; trees cut
`wait` 3.26→2.4; pipelining cut `long_scoreboard` 8.5→6.2; the no-stage cut `barrier` 1.66→0.85,
`short_scoreboard` 1.86→1.12, `mio` 1.51→0.89. **DRAM throughput 37% → 55%**; 51% occupancy (reg-limited
to 6 blocks). Final wall = `long_scoreboard` 7.2 (irreducible global-load latency) at occupancy we can't
raise. **Occupancy was never the lever; data-dependency structure + load pipelining were.**

**Experiments that FAILED (don't repeat):** (a) thread-coarsening 2 elements/thread — **fundamental
capacity wall, not just registers**: 2 elements = 2× working set; in registers → 94 regs → 27% occ;
forced down via `-maxrregcount` → spills to local mem → catastrophic (64→230 µs); moved to shared →
~12 KB/block → SHARED-limited at ~2 blocks → same low occ. No storage location escapes the occupancy
collapse on 16-node elements ⇒ coarse's MLP never beats its capacity cost. DON'T retry. (b) `-maxrregcount`
to force occupancy — spills, much slower (56 regs = the ILP sweet spot, and 56% occ is its equilibrium);
(c) `PreferredSharedMemoryCarveout=0` (max L1) — shrank the shared pool, 51→87 µs; (d) higher order p=4 —
worse vs floor; (e) warp-shuffle mapping — would RAISE registers (PR/PS move shared→regs) and only attacks
the now-small barrier/short_scoreboard stalls; wrong direction for a reg-limited, long_scoreboard-bound kernel.

(f) **lean-then-coarse** (`cuda-ref/lean-matvec.cu`) — the "cut registers FIRST, then coarsen" idea,
done properly: a LEAN single-element kernel (single-accumulator loops, inline neighbour reads, no
hoisting) is **48 regs**, and coarsening it (2 elems/thread) is **56 regs / 52% occ** — register cut from
94→56 CONFIRMED, bit-exact (0.0). BUT `op_coarse_lean` = 74 µs > `op_lean` 69 µs > `operator_fused_ns`
51 µs. Coarsening loses even at the right register budget because **intra-element hoisting/pipelining is a
more efficient use of registers than inter-element coarsening**: hoisting spends regs only on the
latency-critical neighbour loads, coarsening doubles ALL state + shared. So for a kernel whose latency is
a few specific loads, target them directly (hoist) — don't coarsen. DON'T retry coarse in any form.

**`operator_fused_ns` ~51 µs (1.67× floor, 55% DRAM, 56 regs/56% occ) is the practical Titan-V limit for
this kernel.** The residual `long_scoreboard` is irreducible global-load latency; hiding it harder needs
storage (more warps or more MLP) the SM budget can't supply on 16-node elements, and coarsening (the only
way to add MLP) is a worse register-spend than the hoisting `ns` already does. Stop tuning this kernel —
port `ns` to Rust and shift remaining effort to Phase E (iteration count).

**Net: 79–85 µs → 50.8 µs = ~1.6×, 2.8× → 1.67× the BW floor, 37% → 55% DRAM — all bit-reproducible to
4.1e-14 (FP reassociation).** Getting below ~1.5× floor would need a warp-per-element barrier-free
(shuffle) mapping or a transposed [node][elem] layout for coalescing — both bigger rewrites, untested,
and they attack `barrier`/`short_scoreboard` (now small) more than the dominant `long_scoreboard`, so
the upside is limited. **`ns` is the kernel to port to Rust.** Patterns: balanced-tree contractions,
hoist + pre-barrier-issue scattered loads, read same-cache-line element data from L1 (skip staging).

**Action:** port these three patterns (balanced-tree contractions, hoisted loads, pre-barrier load
issue) to the Rust `operator_fused`/`operator_jacobi_fused` for ~1.6× on the fine-level matvec/smoother
(the FP-reassociation needs the §6 tolerance sign-off — `ve-check` to ~1e-13, not bit-exact). The same
patterns very likely help the other hot kernels. NOTE: this revises Session 3 — the matvec is a real
lever after all, *in addition to* Phase E (iteration count). Both are worth doing.

---

## SESSION 3 (2026-06-18) — C++ deep profiling [its "at the floor" conclusion is SUPERSEDED by Session 4]

Full ncu deep-dive on the C++ ref (`cuda-ref/matvec.cu`, standalone ⇒ ncu works perfectly), 256² p=3.
This **definitively closes** the "can we make the matvec faster" question. Bottom line: **gradient is
at its bandwidth floor (optimal); the solo `operator_fused` is latency-bound and NO threading knob
moves it** — occupancy, block size, and polynomial order were all tested and none help.

**Bandwidth calibration (the yardstick):** a pure streaming `out=2u` kernel hits **547 GB/s** (84% of
the 653 GB/s theoretical — the realistic achievable ceiling). The matvec's essential traffic (read u +
write out = 16.8 MB) ⇒ a **BW floor of 30.7 µs**. Every kernel is graded against *its own* essential traffic.

| kernel (256² p=3) | time | vs its BW floor | ncu verdict |
|---|---|---|---|
| `gradient` (writes gx+gy=25MB) | 46.9 µs | **1.03×** | **BW-bound (76% DRAM) — already optimal** |
| two-kernel `gradient+operator` | 145 µs | — | the gx/gy round-trip is pure overhead |
| **solo `operator_fused`** | **84 µs** | **2.77×** | **latency-bound — the only one with headroom** |

**`operator_fused` stall breakdown (ncu WarpStateStats, cyc/instr of 14.85):** long_scoreboard 3.95
(27%, **global-load latency** — u + neighbour-u reads), wait 3.26 (22%, **FP64 result-dependency
chains**), not_selected 1.52 (healthy), barrier 1.23, math_pipe 1.15, mio 0.77. Neither DRAM (37%) nor
the FP64 pipe (41%) is saturated ⇒ classic latency-bound: only **1.46 eligible warps/sched** (of 8.5),
schedulers idle 42% of cycles.

**Every threading knob TESTED and REJECTED (this is the definitive part):**
- **Occupancy is NOT the lever (Volkov, proven).** `-maxrregcount` sweep: 56 regs (default) = 84.7 µs
  is the SWEET SPOT; 48→91 µs, 40→132 µs, 32→679 µs (spilling). Forcing higher occupancy by cutting
  registers is strictly WORSE — the registers buy ILP that hides more latency than extra warps would.
  (The kernel is reg+shared co-limited to 6 blocks = 56% occupancy, and that is *optimal*, not a defect.)
- **Block size / `epb` doesn't matter.** Sweep 64→512 threads/block: flat 81–88 µs (epb=8 marginally
  best at 81.4 vs 84 at epb=12 — noise).
- **Higher order does NOT help the solo matvec.** p=4 (n1=5): `gradient` stays at 1.02× its BW floor,
  but `operator_fused` goes to **4.28× floor (worse than p=3's 2.77×)** — deeper FP64 chains + the
  per-face neighbour recompute add more latency than the higher AI saves.

**Why this is the floor:** matvec arithmetic intensity at p=3 is ~0.3–0.7 FLOP/byte vs machine balance
~11 — two orders below, so it is *correctly* memory/latency-bound. The only ways to go faster are
(a) move less data — DONE (the solo fusion already removed the gx/gy round-trip: 145→84 µs), (b) hide
latency better — EXHAUSTED (occupancy/ILP at the sweet spot), or (c) raise AI via higher p — tested,
backfires. The remaining 2.77×-over-floor gap is **irreducible latency** inherent to low-order SIPG
(scattered neighbour reads + short FP64 dependency chains) on a Titan V. The kernels are near-optimal.

**Therefore (re-confirmed, now with hard C++ evidence): the per-step lever is NOT the matvec.** It is
iteration count / kernels-per-iteration (Phase E). Each matvec is already near its hardware limit; the
step is slow because there are thousands of them. `cuda-ref/matvec.cu` (now with `bw_copy` calibration
+ `EPB=` override) is the standing oracle — re-grade any future kernel against the 547 GB/s / 30.7 µs floor.

---

## SESSION 2 (2026-06-18) — C++ CUDA cross-check OVERTURNS the plan's core evidence

A standalone C++ CUDA port of the matvec (`gale-gpu/cuda-ref/matvec.cu`, nvcc -O3 -arch=sm_70) was
built to answer: is the matvec latency-bound because of STRUCTURE or because of cuda-oxide CODEGEN?
**Answer: NEITHER — the matvec was never actually latency-bound. The §2 ncu evidence was an artifact.**

Clean **CUDA-event** timing (200 back-to-back launches, no graph, no ncu replay), 256² p=3, side by side:

| kernel        | Rust (cuda-oxide, `op_profile`) | C++ (nvcc) | C++ ncu DRAM% |
|---------------|---------------------------------|------------|---------------|
| `gradient`    | **46.8 µs**                     | 46.7 µs    | **76%** (BW-bound) |
| `operator`    | 107 µs                          | 96 µs      | 57%           |
| solo fused    | (neutral, see §S1)              | 86 µs      | 34% (compute) |

**Rust `gradient` == C++ `gradient` to within noise (46.8 vs 46.7 µs), and C++ ncu shows it at 76% DRAM
— BANDWIDTH-BOUND.** So:
- **The plan's headline §2 evidence ("<20% HBM2, latency-bound at maxed occupancy, 245 µs") is WRONG.**
  It was measured with `ncu --graph-profiling graph` on kernels living inside conditional while-graphs,
  which ncu itself flags as unsupported ("Kernel nodes of a graph which can have conditional nodes are
  not supported") — the durations/BW% it reported are garbage. The matvec already saturates HBM2.
- **Phase B (register sum-factorization / warp-shuffle ILP) is therefore NOT worth doing** — it targets
  a latency problem that does not exist; you'd be restructuring an already-bandwidth-bound kernel.
- **The C++ port DID confirm the fusion direction is sound**: C++ two-kernel (gradient+operator) = 142 µs
  vs C++ solo `operator_fused` = 86 µs (**1.65×**). The Rust solo fusion is bit-exact and a genuine
  per-call win at the fine level — it just doesn't move per-STEP time because (per §S1) the fine matvec
  is a small fraction of the step; the step is dominated by the COUNT of coarse-level + CG-vector kernels.
- **`op_profile` is the right tool for matvec perf** (clean event timing): `MG_P=3 MG_GRID=256 cargo oxide
  run --bin op_profile`. Do NOT trust `ncu --graph-profiling graph` durations/BW for the while-graph
  solve kernels — cross-check against op_profile (out-of-graph) or the C++ ref.

**Net corrected direction:** the matvec kernels are fine (BW-bound, == C++). The ONLY real per-step lever
is **iteration count / kernels-per-iteration** (Phase E): ~15 outer-PCG iters/solve × ~6 solves/step, each
V-cycle a stack of smooths+transfers+dots. Cut V-cycle count (Chebyshev/stronger precond) and/or fuse the
CG-vector ops. Keep `cuda-ref/matvec.cu` as the BW oracle for any future kernel work.

---

## SESSION 1 RESULTS (2026-06-18) — partially superseded by Session 2 above (the ncu numbers were artifacts)

**What was done (all bit-exact, validated `resident-ve-check` rel vel 1.120e-15 / rel Ψ 6.008e-15
unchanged, `ve-check` + `resident-ve-kernels-check` pass):**
1. **Phase D0 — `upwind_lift` node-parallel** (logconf.rs): rewrote the thread-0-serial face loop to
   one-thread-per-node register-gather (closed-form face membership, branchless inflow). **Real win:
   412 µs → 240 µs/call at 256² (1.7×), 113 µs at 128² (~3.6×)**; now ~35% DRAM-bound vs the old
   serial kernel. Bit-exact (`upwind-lift rel err = 0.000e0`). KEEP. (It's only ~0.2% of GPU time at
   256², so negligible on per-step — but a clean kernel win + the node-gather warm-up the plan wanted.)
2. **Phase A2/C2 — SOLO matvec** (poisson.rs `operator_fused`, `operator_jacobi_fused`): folded
   `gradient`+`operator`(/`_jacobi`) into ONE kernel — own gradient recomputed from staged `u`, AND
   the interior-face neighbour normal-derivative recomputed on the fly from the neighbour's `u` row
   (valid on the uniform axis-aligned mesh `GpuPoissonMg` already assumes). **Eliminates the
   `gradient` kernel entirely (was 18.8% GPU time) and the `gx`/`gy` global buffers.** Wired into
   `matvec!`/`matvec0!`/`matvec_c!`/`smooth_sweep!` (non-SBM branch; SBM keeps the 2-pass path).

**The measured result: PER-STEP TIME UNCHANGED (~263 → ~271 ms/step at 256², within ±15 ms noise).**
nsys (`--cuda-graph-trace=node`) before/after, matvec-chain GPU time 853 → 837 ms/4-steps (2% better,
unmeasurable on wall-clock). The solo fusion is **bit-exact and perf-NEUTRAL**, not the 5–8× win the
plan predicted, because:

**THE PLAN'S PREMISE WAS WRONG. The fine matvec is NOT a concentrated, fusable bottleneck.** The
full-step nsys breakdown (256², 4 steps, while-graph) shows the cost is the **aggregate VOLUME of
thousands of small kernels**, dominated by the SMOOTHER *count*, not fine-matvec inefficiency:
- `operator_jacobi(_fused)` smoother **51%** — but **median 8 µs (dispatch-floor, coarse levels)**,
  max 188 µs (fine). ~21,000 instances / 4 steps = **~5,300 smoother sweeps PER STEP**.
- `operator_fused` matvec 14%; then a long tail of `h_prolong` 4.9%, `dot_partial` 4.7%, `restrict`
  3.4%, `prolong` 3.4%, `cg_xr` 3.0%, `axpy` 2.5%, `sub` 2.1%, `scal` 1.9%, `h_restrict` 1.8%,
  `xpby_s` 1.7%, `reduce_scalar`/`reduce_beta`, … — the V-cycle transfer + CG-vector kernels.
- Back-of-envelope: ~5,300 smooths/step ⇒ ~44 V-cycles/solve × ~6 elliptic solves/step. **44
  V-cycles to reach 1e-6 is POOR MG convergence** (good MG ≈ 8–12) — so the dominant lever is
  **iteration COUNT** (Phase E #4: stronger smoother/preconditioner), NOT the matvec thread mapping.

**Verdict / next-session direction:** matvec fusion is done and banked (neutral, frees memory +
removes the gradient launch); fusing the fine matvec harder (Phase B register/shuffle ILP) can at
best shave the ~14–51% the two matvec kernels cost, and they're latency-bound across mostly tiny
coarse launches — diminishing returns. **The real >2× is in Phase E, specifically cutting the V-cycle
count**: a Chebyshev (matvec-only) smoother and/or a stronger preconditioner so the deflated singular
pressure solve needs ~10 cycles not ~44. Measure `pcg_cond` iters/solve first (profile shows ~92
outer-PCG checks/step). Phase B (ILP) is secondary and only worth it after the iteration count drops.

**Status:** Phase D0 + solo-matvec fusion DONE & bit-exact (neutral perf). Premise corrected above.

---

**Status (original):** PLAN — not started. Author handoff doc (context will be cleared before implementation).
**Owner kernel target:** the DG-SEM volume operator chain `gradient → operator` and the Jacobi
smoother `operator_jacobi` in `gale-gpu/src/operators/poisson.rs` (the conforming `GpuPoissonMg`
path used by the device-resident VE step `GpuResidentVe`).
**Goal:** take the elliptic-solve inner loop from **<20% HBM2 bandwidth (latency-bound at maxed
occupancy)** to **bandwidth-bound (~70%+ of 653 GB/s)** at 256², by restructuring the thread→work
mapping (ILP/MLP + operator fusion), NOT by adding occupancy (already exhausted).

---

## 0. Why this matters / success criteria

The device-resident Kolmogorov VE sim (`GpuResidentVe`) is **correct and 100% device-resident**
(100% SM util, self-driving on-device while-graphs). The remaining differentiator is *utilization*:
at 256² the step is **elliptic-solve-bound** and the solve kernels run at **<20% of HBM2 bandwidth**.
This plan closes that gap.

**Definition of done (all required):**
1. `ve-check` passes bit-for-bit (the redesign must be numerically identical or within the documented
   FP-reassociation tolerance; default target = **bit-exact**, see §6).
2. ncu on the redesigned matvec at 256² shows **eligible-warps/scheduler ↑**, **issued IPC ↑ (from
   1.73 toward >3 of 4)**, and **Max DRAM Bandwidth ↑ from ~18% toward ≥70%** — OR a clear
   evidence-based statement of the new bottleneck.
3. Measured **per-step at 256² drops** materially (target: matvec/smoother 5–8× ⇒ whole step from
   ~280 ms/step at tol 1e-6 toward <100 ms/step). Report before/after numbers.
4. The profiling gate in `.claude/skills/gale-gpu-perf` is completed in the implementation session.

---

## 1. Current state (what exists — do not rebuild)

- **`GpuResidentVe`** (`gale-gpu/src/resident.rs`): the 100%-device-resident VE dual-splitting step.
  All handles (`hp` pressure, `hv` velocity Helmholtz, `hdiff` κ stress diffusion, `lcg` log-conf) on
  ONE shared stream; `with_while_graph(true)` runs each solve's PCG loop as an on-device conditional
  graph (zero per-iter host sync). Uniform mesh only (uses `GpuPoissonMg`). Validated
  (`resident_ve_check`).
- **`traj-resident-kolmogorov`** (`gale-gpu/src/bin/traj_resident_kolmogorov.rs`): device-resident
  Kolmogorov trajectory bin → `u`/`trC` h5 for `gale-traj/python/spectrum.py`. Already has
  `RK_TOL=1e-6` default (the tolerance win, §2).
- **`resident-ve-perf`** (`gale-gpu/src/bin/resident_ve_perf.rs`): perf/profiling harness for
  `GpuResidentVe`. Has `RVP_N` (res), `RVP_STEPS`, `RVP_TOL`. **Use this for all profiling** (clean,
  no h5 I/O).
- **Already-banked win — solve tolerance:** 1e-9 → 1e-6 = **1.7×** at 256² (471→280 ms/step), measured.
  Turbulence-appropriate (time-discretization error dominates). Baked into both bins. NOT what this
  plan is about, but the baseline for "after" numbers is **tol=1e-6**.
- **The kernels already use multi-element-per-block** (`matvec_cfgs`, poisson.rs:117: `epb =
  192/nn`). That earlier "more warps to hide latency" fix is *why occupancy is already 85%.* **Do not
  re-attempt "more elements per block" as the fix — it is done and occupancy is exhausted.**

Uncommitted at handoff (commit these or note them): `traj_resident_kolmogorov.rs` (new),
`Cargo.toml` (new bin entry), `resident_ve_perf.rs` (RVP_TOL), `flow.rs` (VE_TIMING instrumentation +
the committed MG-NC wiring from `c4259ec`). The `flow.rs` VE_TIMING macro is diagnostic-only (env
`VE_TIMING`); keep or remove at discretion.

---

## 2. The evidence (256², ncu via `--graph-profiling graph`, fine-level `gradient`)

Profiled with (note `--graph-profiling graph`: the solve kernels live in conditional while-graphs and
are otherwise invisible to ncu/nsys — "Kernel nodes of a graph which can have conditional nodes are
not supported"):
```
CUDA_OXIDE_TARGET=sm_70 RVP_N=256 RVP_STEPS=4 RVP_TOL=1e-6 ncu --graph-profiling graph \
  --kernel-name 'regex:gradient|operator_jacobi' --launch-count 8 \
  --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
  --section ComputeWorkloadAnalysis --section Occupancy --section LaunchStats \
  -f -o /tmp/ncu_deep ./target/release/resident-ve-perf
```

| metric | value | meaning |
|---|---|---|
| Achieved occupancy | **85%** (13.5/16 warps/sched) | near Volta max → **occupancy is NOT the lever** |
| Eligible warps/sched/cycle | **1.55** of 13.5 | ~12 resident warps stalled every cycle |
| Issued IPC | **1.73 / 4**; issued/sched 0.43 | schedulers idle 57% of cycles |
| Warp cycles per issued instr | **31** | each instruction waits ~31 cycles |
| Top stalls | **lg_throttle ~17, no_instruction ~14, long_scoreboard, math_pipe/mio_throttle** | LSU saturated + memory latency + short-kernel fetch + FP64/shared pipe |
| Max DRAM BW / Mem Busy | **15–20% / 8.5%** | not bandwidth-bound — huge headroom |
| L1 hit / L2 hit | 2.7% / 67% | served from L2, latency-exposed |
| regs/thread, block, grid | 32, 192 (=12 elem×16 nodes), 5462 | 32×2048=65536 ⇒ register-capped at max occ |
| Duration (fine gradient) | ~236–306 µs each | ×hundreds/step (every CG iter × MG level) |

**Diagnosis (definitive):** latency/throttle-bound at near-maximum occupancy. 54 resident warps and
still cannot hide latency ⇒ adding warps cannot help. Classic "maxed occupancy, still stalled ⇒ need
ILP/MLP, not occupancy" (Volkov, *Better Performance at Lower Occupancy*).

---

## 3. Root cause = the thread→work mapping

Current kernels (`poisson.rs`): **one thread = one node** (nn=16 for p=3), `epb` elements/block.
Per thread: **1 global load → `__syncthreads` → ~8 FP64 FMAs from shared → 2 global stores**.

1. **Too little independent work per thread → no ILP/MLP.** One outstanding load; nothing to overlap
   the ~14-cycle L1TEX/L2 latency. 13.5 warps each issuing every ~31 cycles = the measured 0.43
   issued/sched.
2. **The `gx,gy` DRAM round-trip.** `gradient` (poisson.rs:159) WRITES gx,gy to global; `operator`
   (poisson.rs:214) READS them back. Essential traffic is u(8 MB)+Au(8 MB)=16 MB; the round-trip adds
   write 16 + read 16 = **matvec moves ~48 MB when it needs 16** → this is the `lg_throttle` (LSU
   saturated with stores).
3. **`no_instruction` ~14** — kernels are so short warps keep draining/re-fetching.
4. **Shared-memory staging costs L1 capacity for ~zero L1 benefit.** On Volta L1 and shared are the
   SAME 128 KB/SM SRAM (unified; carveout selectable {0,8,16,32,64,96} KB) — every shared byte is an
   L1 byte you give up. The current kernels stage each element's field into shared
   (`matvec_cfgs`, poisson.rs:117: gradient shared `(nn+epb·nn)·8`=1664 B/block at p=3,epb=12;
   operator `(nn+2·epb·nn)·8`=3200 B/block ⇒ ~9 blocks/SM ⇒ ~29 KB ⇒ driver rounds the carveout to
   32 KB shared, leaving 96 KB L1). Yet the **measured L1 hit is 2.66%** — the shared staging gives
   the global stream no L1 reuse, so we pay the L1 capacity AND get nothing back from L1. The only
   real temporal reuse is (a) WITHIN an element (each node read n1× in the contraction — belongs in
   registers, not necessarily shared) and (b) the **neighbor/face traces** shared by adjacent elements
   (the SIPG term) — and *that* reuse is exactly what a larger L1 would cache. So freeing shared (via
   registers/shuffle, Phase B) both removes the barrier AND returns SRAM to L1 for the face traffic.

---

## 4. Hardware facts (Tesla V100 whitepaper WP-08608, GV100 = Titan V)

- 80 SMs; **4 warp schedulers/SM** (1 inst/cycle each ⇒ **4 IPC/SM max**); **64 warps/SM** (2048
  threads); **65,536 32-bit regs/SM**; **unified 128 KB L1+shared/SM**, shared carveout selectable
  {0,8,16,32,64,96} KB — **shared and L1 trade off 1:1** (Volta merged them; whitepaper §"Combined
  L1 Data Cache and Shared Memory"). Knob: `cuFuncSetAttribute(...
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT ...)` to bias toward L1 when shared use is low.
- **2560 FP64 cores → ~7.0 TFLOP/s FP64** (8 FP64 lanes per SMSP ⇒ an FP64 warp issues over 4
  cycles; relevant to `math_pipe_throttle`).
- **HBM2 653 GB/s** (Titan V), L2 4.5 MB. Volta **independent thread scheduling** (enables
  fine-grained ILP/MLP, warp-shuffle algorithms).

**Machine balance = 7.0 TFLOP/s ÷ 653 GB/s ≈ 11 FLOP/byte (~91 FLOP/FP64-word).** Our matvec AI is
**~0.3–0.7 FLOP/byte** — two orders below balance ⇒ it is *correctly* memory-bound and *should* run
at the HBM2 BW limit. It runs at <20%. The gap is the entire opportunity and is NOT about FLOPs.

**256² CAN saturate (evidence, not assumption):** Little's law in-flight bytes to saturate =
BW × latency ≈ 653 GB/s × ~500 ns ≈ **0.33 MB**. A fused 256² matvec moves **16 MB ≈ 50× that**,
from 65,536 elements. The data volume is ~50× the saturation floor — **failure to saturate is a
software/threading defect, not a hardware or problem-size limit.** (If at any phase you conclude it
*cannot* saturate, you must refute this arithmetic with measured evidence.)

---

## 4b. MANDATORY per-kernel audit lens — read EVERY hot kernel as a veteran GPU engineer

Before (and during) the phases below, audit **each kernel's logic line-by-line** with the eyes of an
engineer who has written GPU kernels for a decade — but who **does NOT assume the Rust/CUDA source
compiles to the efficient PTX/SASS that the equivalent C++ would.** Rust safety idioms inject control
flow that a C++ kernel never has. Verify the *generated* code, don't trust the source's intent:
`cuobjdump --dump-sass target/release/<bin>` (or ncu's **Source page** + the **Branch Efficiency** and
**Warp Execution Efficiency** / **Stall Barrier** metrics) to see divergence, predication, bounds
checks, and serialization empirically. Treat every conditional as guilty until proven free.

**Anti-patterns to hunt (with confirmed examples in this codebase):**
1. **Thread-0 / single-thread serial LOOPS — the worst offender.** `upwind_lift` (logconf.rs:551):
   `if m == 0 { while t<4 { while a<n1 { … } } }` — **ONE node-thread does all 4 faces × n1 nodes
   serially** while the other 15 idle at `sync_threads`. This is why it measured **409 µs/call** (the
   most expensive per-call kernel in the profile, 5.3% of GPU time over just 60 calls). **FIX:**
   node-parallel gather — each thread owns its node and accumulates the faces it lies on into
   registers (exactly the `operator_nc` node-gather rewrite pattern already proven this session). Est.
   ~10–16× on this kernel. **HIGH PRIORITY — concrete, separate from the matvec.** Also note the
   `if un < 0.0` upwind test inside is data-dependent warp divergence — keep it branchless (predicated
   `fac = (un<0) ? sw*un : 0`) when parallelizing.
2. **Whole-kernel 1-thread scalar work.** `cg_alpha`/`cg_beta` (poisson.rs:952,973) run entirely on
   `threadIdx==0`: pure dispatch-floor overhead. → fold into the reduction epilogue (Phase E; partly
   started by the fused `reduce_beta`).
3. **Rust-idiom bounds-checked stores everywhere.** `if let Some(o) = buf.get_mut(thread::index_1d())
   { *o = … }` is pervasive (logconf.rs:170-582, poisson_nc.rs:78-356, …): each is a bounds-compare +
   branch/predicate per store that a C++ kernel writes unguarded. Where the launch config guarantees
   in-range (it almost always does — grid sized to ndof), use `get_unchecked_mut` (some kernels
   already do, e.g. dot_partial's `*partial.get_unchecked_mut(...)`). Same for `buf[i]` reads (panic
   bounds-check branch) → `get_unchecked`. Sweep these; verify the branch disappears in SASS.
4. **Benign vs real `if tid==0` — do NOT "fix" the good ones.** `dot_partial`/`reduce_scalar`
   (poisson.rs:791,821) are *correct* parallel tree reductions; the trailing `if tid==0` is just the
   single-element result write. Leave them — except: the tree's last ≤32 steps (`while s>0 { if tid<s
   … }`) diverge within the final warp; a warp-shuffle epilogue (`__shfl_down` / `redux.sync`) removes
   that divergence + several `sync_threads`. Minor, free, do opportunistically.
5. **Data-dependent divergence** generally — branch where warp lanes disagree (upwind sign, BC kind
   `if base_kind[bc]==0`, half-face `if h==0` in poisson_nc) → prefer predication / branchless select,
   or sort work so a warp is uniform.
6. **Serial `while`/`for` inside a thread that could be unrolled or spread across threads** — check
   each loop: is the trip count small+constant (unroll) or is it doing per-thread what the block could
   do in parallel (redistribute)?

**This audit is not optional and applies to every kernel touched in Phases A–E, plus `upwind_lift` and
the conformation/limiter kernels as standalone targets.** Log each finding (kernel, pattern, fix,
before/after ncu) so the sweep is auditable, not eyeballed.

## 5. The redesign — phased, each phase validated + profiled before the next

Principle: fusion and the latency fix are the **same** change here — fusing the operator chain with
intermediates kept in registers/shared (a) cuts DRAM traffic 3× AND (b) lengthens each thread's
independent instruction stream (ILP) so latency hides. Naive concatenation-fusion would not; this
register-staged fusion does.

### Phase A — Fuse `gradient → operator` into one matvec kernel (volume term)  [highest Amdahl]
- New `#[kernel] fn operator_fused` (poisson.rs kernels module): load u (one coalesced transaction
  per element) → gradient contraction → metric/mass scale → divergence contraction → write `out=Au`.
  **No gx,gy global buffers.** Keep PR/PS intermediates in shared (as `operator` already stages) or
  registers. Traffic 48→16 MB.
- The **face/SIPG term**: `operator` currently gathers face contributions (poisson.rs ~249, race-free
  node-gather, closed-form normals for the affine mesh; `face_nbr` for neighbor traces). Two options,
  pick by measurement:
  - **A1 (simpler first):** keep faces as the existing pass but eliminate ONLY the volume gx,gy
    round-trip by fusing gradient into operator's volume part; faces still read neighbor `u` traces.
  - **A2:** fuse faces too (neighbor traces needed → still one extra neighbor `u` read, no gx,gy).
- Wire `apply`/`vcycle`/`pcg` to call `operator_fused` instead of `gradient`+`operator`. Sites:
  `matvec_cfgs` consumers at poisson.rs:1318,1412,1517,1615,2183,3072,3240,3413,3647 and the
  `gradient_dev` at 2954 (used by device-resident assembly). Search `(gcfg, ocfg)` / `.gradient(` /
  `.operator(` and the `*_dev` device-resident step in `resident.rs::step`.
- **Validate:** `ve-check` bit-exact (volume FMAs unchanged order). **Profile:** ncu the fused kernel
  — expect DRAM BW up, lg_throttle down.

### Phase B — Restructure for ILP/MLP (the latency fix proper)
Occupancy is maxed; raise *instruction-level* and *memory-level* parallelism so each warp has
independent work in flight:
- **Register sum-factorization, one element per warp:** each thread owns a column (or row) of the
  n1×n1 element and accumulates `n1` partial products in registers; issue the `n1` independent loads
  before consuming → MLP hides L2 latency. Drop `__syncthreads` in favor of **warp-shuffle**
  (`__shfl`) for the cross-thread tensor contraction (2 elements share a 32-lane warp at p=3; a warp
  = 1 element at p≥4... handle the p=3 packing explicitly). Volta independent thread scheduling makes
  this clean.
- **Spend registers on ILP, accept lower occupancy** (Volkov): more work/thread ⇒ more regs/thread ⇒
  fewer resident warps, which is FINE if eligible-warps/IPC rise. Sweep epb / regs and measure.
- **Use shared memory sparingly — prefer registers (per point §3.4).** Every shared byte is L1 you
  lose, and the current shared staging buys 2.66% L1 hit. Move the per-element contraction workspace
  to **registers + warp-shuffle** (no shared, no `__syncthreads`); reserve shared only for what truly
  needs cross-warp sharing within a block. Then **bias the carveout toward L1**
  (`PREFERRED_SHARED_MEMORY_CARVEOUT` low) so the **neighbor/face traces cache in L1** (the one access
  pattern with real cross-element reuse). Measure L1 hit rate before/after — expect it to rise from
  2.66% once the face term leans on a bigger L1 instead of manual shared staging. If a phase keeps
  shared, check the achieved carveout (ncu "shared memory configuration size") and that it isn't
  silently shrinking L1 below what the face traffic needs.
- **Validate** bit-exactness becomes order-sensitive here (shuffle reductions reassociate FP64) —
  decide §6 policy. **Profile:** the §0 IPC/BW criteria.

### Phase C — Apply the same to `operator_jacobi` (smoother) and the coarse levels
- `operator_jacobi` (poisson.rs:457) is matvec + diagonal scale; same treatment. It's the biggest
  single attributed chunk (30% in the mixed nsys sum) — high Amdahl.
- Coarse MG levels are small grids (low occupancy by nature, latency-bound, minor cost) — do NOT
  over-invest; a fused kernel that also serves small grids is enough.

### Phase D0 — `upwind_lift` thread-0 serialization (HIGH priority, standalone, do early)
Independent of the matvec — fix the §4b.1 disaster: `upwind_lift` (logconf.rs:551) runs all faces on
`m==0`. Rewrite node-parallel (each thread owns node `vl`, gathers the ≤2 faces it lies on into
registers, branchless upwind), mirroring the `operator_nc` node-gather. It's in the conformation RK3
(3 calls/step) at 409 µs/call → est. ~10–16×. Validate vs `ve-check` conformation path; profile.

### Phase D — Secondary: `conformation` kernel block size
- `conformation` (C=exp Ψ, `logconf.rs`) launches **16 threads/block (33% occupancy)** — pack several
  elements/block (mirror `matvec_cfgs` epb). Quick, low-risk, after the matvec work.

### Phase E — Restructure the SOLVER ITERATION (the next bottleneck after the matvec is BW-bound)
Once Phases A–C make each matvec BW-bound (~35 µs), the matvec stops dominating and the **per-iteration
reduction + CG-vector kernels become the bottleneck** — exactly the prior finding ([[perf-pass-while-graph]]:
"62% of GPU compute is tiny dot/CG-vector kernels at the ~1.5–2.4 µs dispatch floor; 1-thread
cg_alpha/cg_beta = 1.7 µs"). The MG-PCG inner loop runs a matvec **plus** 2–3 global dot-products
(`dot_partial`→`reduce_scalar`) and 3–4 axpy/`cg_xr` updates **per iteration**, hundreds of iters/step.
Each dot is a global reduction = a synchronization point; each tiny kernel sits at the dispatch floor.
Levers, in rough order of payoff (measure with the §0 gate each time):
1. **Cut the number of reductions/sync-points per iteration.** A **Chebyshev (polynomial) smoother**
   in the MG V-cycle is matvec-ONLY — no inner dot products, no reductions — replacing Jacobi
   (`operator_jacobi`, poisson.rs:457) and removing its reduction traffic. It needs an eigenvalue
   estimate (cheap, once per remesh). Strong candidate for the smoother.
2. **Fuse the CG vector ops + partial reductions** so an iteration launches a handful of fat kernels
   instead of ~10 dispatch-floor ones (`cg_alpha`/`cg_beta`/`cg_xr`/`dot_partial`/`reduce_scalar` —
   poisson.rs:951,972,1006,990; `reduce_scalar`/`reduce_beta`). The 1-thread scalar kernels
   (`cg_alpha`/`cg_beta`) are pure dispatch overhead — fold into the reduction epilogue. (This is the
   already-noted `cg_xr` fusion idea, now higher priority because the matvec no longer hides it.)
3. **Fewer global reductions per iteration** via a **pipelined / communication-avoiding CG** variant
   (single fused all-reduce per iter, overlap with matvec). On one GPU the win is from fewer
   sync-points/launches, not network — still real because each reduction is a serialized dispatch chain.
4. **Cut iteration COUNT** (the highest-leverage but algorithmic): a stronger smoother/preconditioner
   so the deflated singular pressure solve (the hardest) needs fewer V-cycles. Chebyshev (1) also
   helps here. Measure iters/solve (the `pcg_cond` device counter, poisson.rs:718) before/after.

These compose with — and only pay off after — the matvec threading fix, because today the matvec
dwarfs the reductions; flip that and the reductions dominate. Treat Phase E as the second half of the
same campaign, gated by re-profiling after Phases A–C.

---

## 6. Numerical-equivalence policy (decide explicitly, document the choice)
- Phase A (volume fusion) keeps per-node FMA order ⇒ aim **bit-exact** (`ve-check` rel ~1e-15).
- Phase B (warp-shuffle reductions) reassociates FP64 sums ⇒ **not** bit-exact. Acceptable IF: (a)
  `ve-check` passes within a documented tol (e.g. vel rel ≤1e-10, conf rel ≤1e-9 — the tolerances the
  existing checks already report), and (b) the Kolmogorov spectrum (α≈3.75 at 128²) is unchanged. If
  bit-exactness is required, keep a `--features bitexact` path using the non-shuffle reduction.

---

## 7. How to build, run, profile (exact commands — environment gotchas)

- **Build:** `cd /home/ian/src/gale/gale-gpu && cargo oxide build` (NOT `cargo-oxide`; `--features
  traj` for the trajectory bin). One crate-wide kernel bundle ⇒ **`#[kernel]` export names must be
  globally unique** (e.g. `operator_fused`, not a generic name).
- **Bare binary under ncu/nsys/compute-sanitizer:** MUST set `CUDA_OXIDE_TARGET=sm_70` (cuda-oxide
  JIT-links the cubin at runtime; default sm_120 ⇒ `DriverError(209) no kernel image`). `cargo oxide
  run` injects it automatically; the profilers run the bare binary so it must be set explicitly.
- **Per-step timing (subtract JIT):** run RVP_STEPS=20 and =70, subtract, ÷50. JIT ≈18 s one-time.
- **Util:** `nvidia-smi dmon -s u` (want SM high). Already 100% — the fix must keep it 100% while
  doing *less* work per step (faster), so track **per-step time + ncu BW**, not util alone.
- **ncu the solve kernels:** they're in conditional graphs ⇒ use `--graph-profiling graph` (see §2).
  ncu works here (`RmProfilingAdminOnly: 0`).
- **Validation:** `cargo oxide run --bin ve-check` (conforming VE bit-exactness);
  `resident_ve_check` (device-resident VE); the 128² spectrum via `traj-resident-kolmogorov` +
  `spectrum.py` (α≈3.75 unchanged).

---

## 8. Risks / watch-list
- **Face/SIPG coupling** is the hard part of full fusion (neighbor traces). Phase A1 de-risks by
  fusing only the volume round-trip first.
- **Warp-shuffle at p=3** (16 nodes, 2 elements/warp) needs careful lane mapping; p≥4 (25 nodes) does
  not divide 32 cleanly — design the contraction for the general n1, validate at p=3 first.
- **Register pressure** could spill (watch ncu "registers per thread" + local-memory traffic).
- **`while_graph` interaction:** the fused kernel must remain capturable inside the conditional graph
  (no host sync, no illegal ops). Validate `GpuResidentVe` still graph-captures.
- **NC path divergence:** `GpuPoissonNc`/`gradient_nc` (poisson_nc.rs) is a separate operator (AMR).
  This plan targets the conforming `GpuPoissonMg` (what `GpuResidentVe` uses). Mirror to NC later only
  if the AMR path is revived for production.
- **Don't regress the committed MG-NC path** (`c4259ec`) or `ve-check`.

---

## 9. Suggested order of execution (first session after context clear)
0. **Standing discipline:** apply the §4b per-kernel audit (SASS/ncu source view) to every kernel you
   touch — log findings (kernel, pattern, fix, before/after).
1. Re-confirm baseline: `RVP_N=256 RVP_TOL=1e-6` per-step (subtract-JIT method) + the §2 ncu numbers.
   (Sanity that nothing drifted.)
2. **Phase D0 first** (`upwind_lift` thread-0 → node-parallel): standalone, low-risk, ~10–16× on a
   confirmed-409 µs kernel, and a warm-up on the node-gather pattern + the audit lens. → `ve-check` →
   profile.
3. Phase A1 (`operator_fused`, volume-only round-trip elimination) → `ve-check` bit-exact → ncu BW.
4. Phase B (register/shuffle MLP, minimize shared → free L1) on the fused kernel → validate (§6) →
   ncu IPC/BW/L1-hit (the §0 gate).
5. Phase C (smoother) → validate → profile.
6. Re-profile: confirm the matvec is now BW-bound and that the per-iteration reduction/CG-vector
   kernels are the new top cost (they should be, per [[perf-pass-while-graph]]) → THEN Phase E
   (Chebyshev smoother + fuse/cut reductions) gated by that profile.
7. Report before/after per-step + BW%; update memory ([[gale-gpu-performance-pass]],
   [[perf-pass-while-graph]]) and this doc's status.
8. Phase D (conformation block size) + remaining Phase E levers if time.

**North star:** 256² matvec from ~18% BW / ~280 ms/step (tol 1e-6) → bandwidth-bound, <100 ms/step,
with ncu showing eligible-warps/IPC/BW all up. If a phase plateaus below BW, profile and state the new
binding resource with whitepaper-grounded arithmetic before stopping.
