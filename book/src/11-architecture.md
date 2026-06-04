# Architecture: Rust, the GPU, and Multi-GPU

> 🎓 **Reviewer — chapter verdict:** The strongest of the four assigned chapters. The "GPU-native vs GPU-accelerated" framing is genuinely well-sold (the FFI-type-mismatch example is the kind of concrete pain a newcomer immediately believes), and "one block per element, one thread per node" is the cleanest possible statement of why DG and GPUs are made for each other. The gather-formulation → determinism → bit-for-bit-checkability chain is the best paragraph in the chapter and earns its length. Two things to fix: the cuda-oxide bug list is one bullet too long and tips from "honest posture" into "look how many bugs I fixed" (trim to two), and the device-bundle section is borderline in-the-weeds — keep it, but cut it to the punchline (see mark). Otherwise: ship it.

The previous chapters were about *what* gale computes — fluxes, penalty terms,
projection steps, mortars. This one is about *where* and *how* that computation
runs: in Rust, on NVIDIA GPUs, written in Rust **all the way down to the device
kernel**, and spread across more than one GPU. The recurring question of the book
has been "what could blow up here, and what are we doing about it?" In the previous
chapters the answer was a numerical-stability story. Here the analogue is a
*correctness* story — a young toolchain and hand-written GPU kernels are exactly
the kind of thing that silently computes the wrong answer, and the architecture is
shaped around making sure they do not.

## Why Rust, and why GPU-*native*

Almost every production GPU fluid solver is **GPU-accelerated**: the simulation
logic lives in C++ (or Python, or Julia) and calls *out* to device code written in
CUDA C, or to a library like cuBLAS or a code-generation layer that emits CUDA C and
compiles it with `nvcc`. The host language and the device language are different
languages, joined by a foreign-function-interface (FFI) boundary and a separate
compiler.

gale is **GPU-native** instead. The device kernels are written in ordinary Rust and
compiled directly to PTX (NVIDIA's GPU assembly) by the
[`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) toolchain — there is no CUDA C,
no `nvcc`, no separate kernel language. A kernel is a Rust function wearing a
`#[kernel]` attribute; the host wrapper that launches it is Rust in the same crate.

Why go to this trouble?

- **One language, one mental model.** The CPU reference operator and the GPU kernel
  are both Rust, often reading the same `Mesh2d`, the same reference-element
  matrices, the same `Neighbor` connectivity. Porting an operator to the GPU is a
  re-expression, not a translation into a second language with its own gotchas.
- **Type safety across the boundary that usually has none.** The classic CUDA bug
  is an FFI mismatch — the host thinks an argument is an `int`, the kernel reads a
  `float`, and nothing complains until the numbers come out wrong. In gale the
  kernel signature *is* a Rust function signature; a launch that passes the wrong
  types does not compile.
- **No FFI marshalling layer to maintain.** There is no hand-written `extern "C"`
  shim, no header that can drift out of sync with the kernel. The host launch
  wrapper and the device function are checked together by one compiler.

The cost is honest and worth stating plainly: **cuda-oxide is young.** At the time
of writing it is a few months old — high-velocity, single-maintainer-dominated,
tagged at v0.1.0. gale does not treat that as a reason to hedge with a fallback
CUDA-C path; it **commits to the toolchain and contributes fixes upstream.** Real
examples gale hit and resolved while porting its operators:

- a **typed-pointer NVVM-IR** bug — on the pre-Blackwell GPUs gale targets (the
  Volta-class Titan V, `sm_70`), the path that emits NVVM IR text to call libdevice
  math functions printed a typed pointer (`double*`) into a slot declared as the
  erased `i8*`, without the bitcast that typed-pointer mode requires, so the kernel
  would not compile. This blocked every kernel needing `sqrt`/`exp`/`log` —
  i.e. compressible Euler and the log-conformation update.
- a **default-target** bug where unknown/sentinel compile targets defaulted to
  opaque pointers (`ptr`), which pre-Blackwell libNVVM cannot parse, breaking the 3D
  hex operators.
- **libdevice-math gaps**, such as `atan2` not being in the device-intrinsic table
  — worked around in gale (and flagged upstream) by computing the symmetric-2×2
  eigenvector directly instead of via `atan2`.

> 🎓 **Reviewer (cut):** Three bullets here, each a real codegen bug, and by the third the reader has stopped learning anything new — the rhetorical work ("a young toolchain bites, and gale bit back") is done by bullet two. The third (`atan2` / libdevice gaps) is the most in-the-weeds and the least illustrative; cut it, or fold it into the prose as a parenthetical. The danger of a list like this is that it reads as a trophy case rather than the "honest cost" you frame it as one paragraph up. Two vivid examples sell the posture; three start to protest it.

The point is not the detail (that lives in `docs/cuda-oxide-codegen-notes.md`); it
is the posture. gale carries a fork of cuda-oxide with these fixes, validates them
on the exact hardware, and treats each one as a candidate upstream contribution.
A GPU-native code on a young toolchain *is* partly a toolchain project — and
gale embraces that rather than hiding from it.

## Why DG maps so well to GPUs

Chapter 3 argued for the discontinuous Galerkin spectral element method (DG-SEM) on
*numerical* grounds: high order, exponential accuracy on smooth flow, the ability to
use few large elements. It turns out the same property that makes DG numerically
attractive — that it is **element-local** — is exactly what makes it map cleanly
onto a GPU. This is the payoff promised back in Chapter 3.

A GPU is a machine for running thousands of threads, grouped into **blocks**, where
threads in a block can cooperate through fast **shared memory** and a barrier
(`sync_threads`). It rewards work that is (a) massively parallel, (b) decomposable
into independent chunks, and (c) arithmetic-heavy relative to how much memory it
touches. DG-SEM is all three:

- **One GPU block per element, one thread per node.** A DG element computes its own
  contribution to the right-hand side from its own nodal data plus a read-only peek
  at its neighbors' face values. That is a self-contained chunk of work — precisely a
  block. Inside the block, each collocation node is a thread. For a degree-4 element
  in 2D that is \\\( (p+1)^2 = 25 \\\) threads; in 3D, \\\( (p+1)^3 = 125 \\\). The mesh
  has thousands of elements, so the GPU is saturated.

> 🎓 **Reviewer (deepen):** "one block per element, one thread per node" is the right hook, and the thread-count arithmetic is exactly the concrete detail a newcomer needs. Worth one extra sentence of practitioner honesty, though: 25 threads (2D, p=4) is *less than one warp* (32), so a naive one-thread-per-node mapping leaves a 2D block underutilizing its warp — which is precisely why the high-arithmetic-intensity, shared-memory-reuse argument in the next bullets is doing the real work, and why the 3D case (125 threads ≈ 4 warps) sits more comfortably. You don't have to dwell on it, but acknowledging that the mapping is "obvious but not automatically efficient" makes the shared-memory paragraph land as the *fix* rather than just a third nice property. A reader who has touched CUDA will trust you more for naming the warp-size wrinkle out loud.
- **The per-element work is dense small tensor contractions.** The volume term of a
  DG operator is the sum-factorized differentiation from Chapter 3 — a handful of
  small matrix–vector products against the reference differentiation matrix. These
  are dense, regular, and have **high arithmetic intensity**: lots of
  multiply-adds per byte loaded. That is the regime where a GPU runs near its
  floating-point peak rather than starving for memory bandwidth.
- **Shared memory holds the element's nodal data.** The kernel stages the element's
  solution and the small reference matrix into shared memory, `sync_threads`, and
  then every thread reuses that staged data many times in the contraction. Fast
  on-chip reuse instead of repeated trips to global memory.

The decisive design choice is the **"gather" formulation**. Each block computes
*its own* element's right-hand side. It reads its neighbors' data, but it only ever
*reads* it — the neighbor contributions are gathered in, never scattered out. No
block writes into another block's element. That matters enormously on a GPU: there
are **no cross-block write races**, so no atomics and no locks on the hot path, and
the result is deterministic regardless of how the blocks happen to be scheduled.
You can see this directly in gale's advection kernel: the block owns element
`e = blockIdx.x`, loads its own nodal flux into shared arrays, reads neighbor face
values through `u[face_nbr[idx]]` (a read), accumulates the numerical flux into a
*shared* face buffer local to the block, and finally writes only its own element's
output slot. Determinism here is not a nicety; it is what makes the bit-for-bit CPU
comparison of Chapter 12 even *possible* — a racy kernel would give slightly
different answers run to run and could never be checked against an oracle to
\\\( 10^{-14} \\\).

> 🎓 **Reviewer:** This is the best paragraph in the chapter. The gather-vs-scatter choice → no cross-block write races → no atomics → run-to-run determinism → *that is what makes the bit-for-bit oracle test possible* is a genuine, non-obvious insight, and you've chained it cleanly. Most GPU-CFD writeups mention "gather formulation" and move on; tying it forward to the validation story is the thing that will make a careful reader sit up. Leave it exactly as is. (One micro-note: the floating-point reason it's deterministic is that gather fixes the *order of summation* per output node — scatter-with-atomics does not. You're implying this; a four-word aside "(the summation order is fixed)" would nail it for the FP-pedantic reader, but it's optional.)

Data layout follows from this. Nodal values are stored element-major
(`element * n_nodes + node`), so the threads of a block read a contiguous run of
global memory — **coalesced** access, the fast pattern. The small reference
matrices (differentiation, filtering) are the same for every element and can live in
constant or shared memory, loaded once and reused across all blocks.

## The device-bundle model: one constraint that shapes the code

There is one cuda-oxide rule that visibly shapes how gale's GPU code is organized,
and it is worth understanding because it explains some otherwise-odd naming.

cuda-oxide compiles **every** `#[kernel]` in a crate into a *single* device bundle,
keyed by the crate name. Every `#[cuda_module]` in that crate loads that same
crate-wide bundle. The consequence: **kernel export names must be globally unique
across the whole crate.** A non-generic `#[kernel]` exports under its bare function
name, with no module namespacing to disambiguate it.

So in `gale-gpu` you find the 2D advection kernel named `advect2d_rhs` and its 3D
hex sibling `advect3d_rhs` — not both `advect_rhs` in different modules, which would
collide in the shared bundle. The multi-GPU variants get yet another suffix
(`advect2d_mg_rhs`). And **shared linear-algebra primitives are defined once**,
crate-wide, rather than copied into each operator module, because two definitions of
the same export name would clash.

This is why `gale-gpu` is organized as one crate with many operator modules under a
single bundle, with a disciplined export-naming convention, rather than as a swarm of
tiny per-operator crates. The constraint is a cuda-oxide fact, and the module layout
is gale's response to it.

> 🎓 **Reviewer:** Verdict on the question I was asked to rule on — *keep it, but trim it.* This is in-the-weeds, but it's the *good* kind of weeds: it's a real-world consequence of building on a young toolchain that a blog reader hasn't seen elsewhere, and "why are these kernels named `advect2d_rhs` instead of `advect_rhs`?" is exactly the sort of small mystery that makes a curious reader feel let into the machine room. What it doesn't need is three paragraphs. The payload is two sentences: (1) cuda-oxide puts every `#[kernel]` in one crate-wide bundle, so export names must be globally unique; (2) hence the `2d`/`3d`/`mg` suffixes and the single crate with many modules. The middle paragraph re-explains the collision twice. Cut it to roughly half its length and it goes from "in the weeds" to "delightful footnote you chose to read."

## The hybrid host/GPU pattern

Not every part of a flow solve belongs on the GPU. gale uses a deliberately
**hybrid** pattern, paying off the principle stated in Chapters 6 and 7: *put the
expensive work on the GPU and keep the cheap work simple.*

In an incompressible step the dominant cost is the **elliptic solves** — the
pressure-Poisson projection and the viscous Helmholtz solve of Chapter 6, run with
the matrix-free Krylov solvers of Chapter 7. Those iterate the SIPG operator many
times per timestep over the whole mesh; they are the bottleneck, and they run
**entirely on-device**. The matrix-free operator-apply, the CG/PCG iteration, the
multigrid preconditioner — all of it stays on the GPU across the iterations, so the
solution vector never round-trips to the host inside the Krylov loop.

The *cheap*, element-local **assembly** — building the right-hand side, the
divergence and gradient corrections of the dual-splitting scheme — reuses the
validated host code. These are O(one pass over the DOFs) operations, negligible next
to dozens of Krylov iterations, and re-deriving them as kernels would buy little
speed while adding surface area to validate.

The engineering reason this is the right split is **host↔device traffic**. The slow
thing in GPU programming is moving data across the PCIe bus between CPU and GPU. By
keeping the iterative solver resident on the device and only crossing the boundary
for the cheap setup at the start and end of a step, gale minimizes that traffic in
the inner loop — the loop that runs millions of times over a simulation.

## Multi-GPU: first-class, not bolted on

gale's headline application — large viscoelastic particle-laden suspensions — is
big. Resolving many particles in a domain is exactly the regime where one GPU's
memory and throughput run out, so **multi-GPU is a first-class concern**, present in
the design from the start rather than retrofitted.

The strategy is **domain decomposition with peer-to-peer (P2P) halo exchange.** The
mesh is partitioned so each GPU owns a block of elements. Each device holds its
partition's state in a combined `[local | halo]` buffer: the local elements, plus a
*halo* region holding the neighbor face-traces that live on *other* devices. Before
each operator apply, the cross-partition traces are copied straight from one GPU into
another's halo region by `cuMemcpyPeerAsync` over the PCIe link (with peer access
enabled) — no detour through host memory.

The elegant part is that **the kernel does not change.** Recall the gather
formulation reads neighbors through `u[face_nbr[idx]]`. With the halo in place,
`face_nbr` simply points either into local state or into the halo slot; the *same*
advection kernel runs unmodified on each device. The decomposition is exactly the one
validated CPU-side against the monolithic operator (Chapter 12), so the multi-GPU
result is provably the single-GPU result, split. gale runs **real two-device
advection in both 2D and 3D** on its 2× Titan V box today.

> 🎓 **Reviewer (flag):** "provably the single-GPU result, split" is *almost* true and I'd tighten it to be exactly true, because a sharp reader will poke it. It's bit-for-bit identical only if the partitioned operator does its per-node summations in the same order as the monolithic one — for a pure gather of face traces it does, so the claim holds *here*. But the word "provably" is carrying a floating-point assumption you haven't stated, and the moment a reduction (a CG dot-product, a global residual norm) crosses the partition boundary, partial sums reassociate and bit-for-bit becomes "agrees to round-off." Since this section is specifically about *advection* (a gather, no global reduction) you're fine — but I'd swap "provably the single-GPU result" for "bit-for-bit the single-GPU advection result" so the claim is scoped to the case where it's actually exact, and you don't write a check the multi-GPU *Krylov* solver can't cash later.

## How gale does it

- **Crate split: `gale` (host) vs `gale-gpu` (device).** The pure-host `gale`
  library implements every operator on the CPU and is `std`-only; it never depends
  on `gale-gpu` or on cuda-oxide, so `cargo test` is a pure-CPU oracle that anyone
  can run. The `gale-gpu` crate carries the `#[cuda_module]` device code, depends on
  `gale` for the mesh/operator types, and must be built with the cuda-oxide backend
  (`cargo oxide`).
- **The `#[cuda_module]` pattern.** Each operator module wraps its `#[kernel]`s in a
  `#[cuda_module]` and exposes a plain Rust host wrapper (e.g. `advection_rhs`) that
  flattens the mesh metrics, uploads buffers, launches one block per element, and
  gathers the result. A fork fix anchors the embedded device artifact across the
  rlib boundary so these wrappers are reusable *library* components, not one-off
  binaries.
- **`distributed.rs`** holds the multi-GPU wrappers (`multigpu_advection_2d` /
  `_3d`), driven by the framework's `Device` / `DomainDecomposition` abstraction
  (Chapter 12) and the P2P primitives from cuda-oxide's host crate.
- **The cuda-oxide fork.** gale pins a fork carrying its codegen fixes
  (typed-pointer NVVM-IR, target-default, the math-intrinsic workarounds), each one
  scoped as an upstream contribution. The toolchain status and contribution map live
  in `docs/cuda-oxide-repo-status.md` and `docs/cuda-oxide-codegen-notes.md`.

With the *where* and *how* of execution established, the next chapter turns to the
*assembly* layer: how a whole simulation — CPU or GPU — is declared, composed, and
checked for correctness.
