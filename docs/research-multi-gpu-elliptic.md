# Multi-GPU for gale's elliptic MG-PCG solver — research reference

Verified deep-research pass (5 angles, 25 sources, 25 claims 3-vote-verified, 24 confirmed / 1
refuted; Reisner/Olson/Moulton, Ghysels/Vanroose, Carson, Kronbichler, NVIDIA HPGMG). Target: 2×
Titan V (sm_70), **PCIe P2P, no NVLink**. Method-decision reference per [[research-before-architecture]];
builds on the existing validated multi-GPU *advection* path (`gale-gpu/src/distributed.rs`:
`DomainDecomposition` + `memcpy_peer_async` halo).

## TL;DR verdict — do it, but as a *thin finest-level* distribution, at *large* sizes

Multi-GPU pays for gale's elliptic solve **only where there's enough work to saturate a GPU — the
finest level(s) at large DoF counts** — and the complexity is justified only if you confine
distribution there. The literature is unanimous on the two load-bearing decisions:

1. **Distribute ONLY the finest level(s).** The matrix-free DG matvec is bandwidth-bound and
   occupancy-saturated only at large problem sizes (A100: ~4M DoF for the matvec, **>1B DoF for the
   smoother**; Titan V saturates somewhat smaller but the shape holds). This is exactly your
   large-size instinct — and it's the *only* regime where a P2P halo exchange overlaps a substantial
   matvec instead of adding latency to an under-occupied kernel.
2. **Do NOT distribute the coarse levels — replicate them.** The parallel-multigrid coarse-grid
   bottleneck is fundamental: work-per-processor falls *exponentially* descending the V-cycle, so the
   coarse levels are latency-limited (NVIDIA: "not enough work to make efficient use of all the
   parallel cores"). gale h-coarsens to ~1 element — un-splittable across 2 GPUs anyway. Keep the
   coarse solve **replicated/redundant on each GPU** (or agglomerated onto one), so there's *zero*
   cross-GPU coarse communication.

**Expected strong scaling: ~1.3–1.7×, not 2×.** The distributed finest level can approach ~2×
*there*, but the replicated coarse levels + the per-iteration P2P halo + the global reduction drag
the whole solve down. Your large-size point pushes toward the **upper** end of that band (the bigger
the finest level, the more its bandwidth-bound matvec dominates and hides the fixed comm cost) — but
the replicated coarse work and reduction keep it under the ideal 2×.

This is an honest "yes, but modest." It's *not* the regime where multi-GPU shines (that's
weak-scaling huge problems across many NVLink GPUs); on a 2-GPU PCIe box the win is real but capped.
The strongest case is **memory capacity**: at ~1024² p4 (~26M DoF) the MG working set (~7–8 GB)
already fills one 12 GB Titan V, and beyond that the problem *doesn't fit on one card* — there,
2-GPU isn't faster, it's *required*.

## §1 — Coarse-grid handling: replicate, don't distribute

- **The bottleneck is fundamental** (Reisner/Olson/Moulton SIAM JSC 2018; SciELO 2022; NVIDIA HPGMG):
  "the cost of message passing may start to prevail over the cost of computations" as work-per-unit
  falls exponentially down the cycle. This is *gale's already-profiled regime* (small coarse matvecs
  are latency-bound; mixed precision failed for the same reason).
- **Recommendation:** keep the partition **consistent across all MG levels** (a coarse parent stays
  on the same GPU as its children — our h-coarsening makes this natural, the 4 children of a coarse
  element are contiguous), and **replicate the coarse solve redundantly on both GPUs** so it needs no
  cross-GPU communication at all. Truncating the V-cycle (stop coarsening while ≥ N elements/GPU) is
  a valid alternative but unnecessary if the coarse solve is just replicated.
- **Refuted (do not adopt):** predictive-model-guided *incremental agglomeration* as the strategy
  (arXiv:1803.02481, voted 0-3). Prefer the simple replicated/single-GPU coarse solve.
- (CPU-offload of coarse levels gave 7–30% in a 2014 Kepler HPGMG benchmark — directional only, a
  different hardware/axis; not transplantable to Titan V.)

## §2 — Halo exchange in the matvec (reuse the advection pattern)

- IP-DG "do not magically guarantee high performance: they require non-local memory access due to
  coupling between neighbouring cells" (arXiv:2510.00998). The enabler for hiding the halo is
  **separating cell- and facet-operations**: post the P2P face-trace copy, **compute interior
  elements while the copy is in flight**, then apply the SIPG penalty/flux at the partition boundary
  once traces arrive.
- **Exchange face traces only** (boundary `gx/gy/u` at partition edges), not volumes — surface/volume
  ratio is `~2/√N` in 2D (≈3% at 64², ≈0.2% at 1024²), so the halo vanishes at scale.
- **Bit-consistency pitfall:** the boundary SIPG penalty/flux MUST be applied *after* halo arrival, or
  the distributed operator diverges from the monolithic one. gale already proved exactly this pattern
  for explicit advection (`distributed.rs`, validated bit-for-bit) — the elliptic matvec reuses it.

## §3 — Global reductions: plain CG + hand-rolled P2P add-reduce

- Every CG iteration needs a global all-reduce for the dot-products **and** the deflation mean (for
  the singular pressure). On **2 GPUs a single small all-reduce is cheap** — hand-roll it as a P2P
  copy + add; **NCCL is not worth its setup overhead for 2 ranks**.
- **Pipelined CG** (Ghysels-Vanroose: 2 reductions → 1, overlapped with the SpMV) and **s-step CG**
  (Carson: O(s) fewer syncs) are correct *mechanisms* but their benefit is **conditioned on the
  reduction being the dominant cost at large processor counts** — explicitly *not* gale's 2-GPU
  regime (the SpMV-to-reduction ratio is unfavorable for hiding), and they carry **real
  finite-precision stability costs** (multi-term recurrences; ill-conditioned s-step bases). **Defer
  them**; start with plain CG. Treat pipelining as a deferred option *only if* the all-reduce ever
  measures as dominant.
- **Optimization:** fuse the deflation-mean reduction into the *same* peer reduction as the CG
  dot-products to avoid a second synchronization point per iteration (open question, worth doing).

## §4 — Does it pay? (the honest answer)

- **Latency-bound regime confirmed:** small/coarse levels are under-occupied; reduction-hiding needs
  `SpMV cost ≥ reduction latency` — only true at the large finest level. So distribution pays *only*
  there, and the global solve lands ~1.3–1.7×.
- **Cartesian caveat (relevant to gale):** Kronbichler & Kormann found the matrix-free DG operator is
  within 10% of peak bandwidth *except* the Cartesian/axis-aligned case, "where the cost of gather
  operations and communication are more substantial" — i.e. gale's affine-metric mesh is precisely
  where comms are a *larger* fraction. Tempers the optimism slightly.

## §5 — P2P vs NCCL

Hand-rolled **P2P (`cuMemcpyPeerAsync` over PCIe)** for both the halo and the 2-rank add-reduce;
overlap with compute via streams/events. NCCL's collective machinery isn't worth it for 2 ranks.
gale's existing advection halo path is the substrate.

## §6 — Partitioning

For the structured axis-aligned mesh, simple **block / SFC (Morton/Hilbert)** partitioning suffices
(METIS graph partitioning is overkill). Keep the partition **consistent across all MG levels**.
SFC is also the load-balance substrate if AMR is added. Halo size grows with `p` (more face nodes
per boundary face) — measure where it starts to erode the finest-level benefit.

## §7 — Recommendation + first increment

**Architecture:** distribute the finest level(s) only (P2P face-trace halo matvec + smoother);
replicate the coarse solve on each GPU (zero coarse comm); plain CG with a hand-rolled P2P
add-reduce (fuse the deflation mean into it); partition consistent across all levels; target large
finest-level sizes.

**Decisive measurement FIRST (the key open question):** micro-benchmark the **PCIe P2P round-trip
latency** (small-message `cuMemcpyPeerAsync` + the 2-rank add-reduce) between the two Titan Vs, and
find the **finest-level DoF count where the matvec time ≥ that latency** (the crossover). That single
number tells us whether — and at what size — multi-GPU pays on *this* hardware, before building the
distributed solver. Cheap, decisive, and it's exactly what no source measured for this config.

**First implementation increment:** distribute only the finest-level smoother + matvec via the
existing P2P halo path; keep everything else (coarse solve, all other levels) replicated and the
outer CG structure intact; validate the distributed operator bit-for-bit vs the single-GPU one
(reuse the advection validation pattern), then measure strong scaling at the crossover size and up.

**Pitfalls:** (1) apply the boundary SIPG penalty *after* halo arrival (bit-consistency); (2) keep
the partition consistent across MG levels (coarse parent with its children); (3) don't expect
precision tricks to rescue the coarse levels — mixed precision already failed for the same
latency-bound reason.

## Open questions to resolve empirically

1. **The PCIe latency + crossover DoF** (the §7 measurement) — *the* gating number for this hardware.
2. Can the deflation mean be **fused** into the CG-dot peer reduction (one sync/iter not two)?
3. **Replicate-on-both** vs agglomerate-on-one vs CPU-offload for the 1×1 coarse grid — measure.
4. How the SIPG halo size grows with `p`, and where it erodes the finest-level benefit.

## Sources (primary unless noted)

- Reisner, Olson & Moulton, *Scaling structured multigrid to 500K+ cores* / coarse-grid model, SIAM JSC / arXiv:1803.02481 (2018)
- Multilevel interior-penalty DG on GPUs (A100 saturation thresholds), arXiv:2405.18982 (2024)
- Kronbichler & Kormann, *Matrix-free DG operator evaluation*, arXiv:1711.03590 (Cartesian gather caveat)
- arXiv:2510.00998 (2025, preprint) — cell/facet separation for IP-DG comm overlap
- Ghysels & Vanroose, *Pipelined CG*, Parallel Computing 2014, DOI 10.1016/j.parco.2013.06.001
- Cornelis/Cools/Vanroose, *deep-pipelined p(l)-CG*, arXiv:1801.04728; stability arXiv:1804.02962, 1902.03100
- Carson, *s-step CG*, SIAM, DOI 10.1137/16M1107942
- NVIDIA HPGMG GPU blog (coarse levels → CPU/latency-optimized); P2P/NCCL practitioner sources

## Caveats

Strong source quality, but a real **relevance gap**: nearly every source is at far higher processor
counts (500K+ cores, where the reduction genuinely dominates) or is CPU/MPI / a single-axis GPU-vs-CPU
study — **none measures gale's exact config** (2 PCIe Titan Vs, no NVLink, matrix-free SIPG, deflated
singular-Neumann). The pipelined/s-step claims are true *as mechanisms* but their benefit is scoped to
large scale (weak applicability here). The DoF-saturation thresholds (4M/1B) are A100-specific. The
**~1.3–1.7× strong-scaling estimate and the P2P-over-NCCL call are engineering inferences, not direct
quotes** (medium confidence) — which is exactly why the §7 PCIe-latency/crossover measurement comes
*before* the build.
