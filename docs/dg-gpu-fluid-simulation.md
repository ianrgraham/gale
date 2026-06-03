# Discontinuous Galerkin Methods for GPU Fluid Simulation

**A research reference for the `gale` project**

> **Scope.** This document surveys discontinuous Galerkin (DG) methods for fluid
> dynamics on GPUs, oriented toward `gale`'s goals: a native-Rust + CUDA
> proof-of-concept for **viscoelastic / incompressible** flows on **multi-GPU**
> hardware (2× Titan V, Volta, FP64-capable), with **first-class immersed
> boundary** support.
>
> **Confidence convention.** Statements tagged **[V]** are backed by an
> adversarially-verified research pass (each survived 3-0 refutation voting); see
> [References](#references). Statements tagged **[E]** are engineering judgment /
> general domain knowledge filling residual gaps. **Two** research passes feed
> this doc: the first covered DG theory, GPU performance, viscoelasticity, and
> IBM; a second pass upgraded the Rust toolchain (§9), multi-GPU (§6), data layout
> (§5.3), and software comparison (§8) from [E] to [V]. Treat **[E]** items as
> informed starting points, and **re-verify the Rust-toolchain findings** — they
> are dated ≈May 2026 and move fast.
>
> *Last updated: 2026-06-01.*

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Why DG, and why DG on GPUs](#2-why-dg-and-why-dg-on-gpus)
3. [DG formulation primer](#3-dg-formulation-primer)
4. [Incompressible & viscoelastic flow](#4-incompressible--viscoelastic-flow)
5. [Mapping DG to the GPU](#5-mapping-dg-to-the-gpu)
6. [Multi-GPU & domain decomposition](#6-multi-gpu--domain-decomposition)
7. [Immersed boundary method + DG](#7-immersed-boundary-method--dg)
8. [Software landscape to learn from](#8-software-landscape-to-learn-from)
9. [The Rust GPU landscape](#9-the-rust-gpu-landscape)
10. [Recommended roadmap for gale](#10-recommended-roadmap-for-gale)
11. [Open questions](#11-open-questions)
12. [References](#references)

---

## 1. Executive summary

- **DG is structurally a good fit for GPUs.** The DG operator is overwhelmingly
  *element-local*, with only weak penalty-based coupling between neighboring
  elements through numerical fluxes on shared faces. That locality is exactly the
  memory-access pattern massively parallel hardware rewards, and it has produced
  large measured speedups since the earliest GPU DG work (40–60× over a serial
  CPU core on a 2009 GTX 280; 30× time-to-solution over CPU at scale to thousands
  of V100s). **[V]**

- **Matrix-free is the only sane choice at high order.** Evaluating the operator
  via *sum factorization* (exploiting tensor-product bases on quad/hex elements)
  is compute-bound with high arithmetic intensity, recast as small dense
  matrix–matrix products. Storing the operator matrix instead makes evaluation
  memory-bound at ~2 flops/byte — a poor fit for bandwidth-limited accelerators.
  **[V]**

- **The hot kernels are small batched GEMMs**, and hand/code-generated CUDA can
  beat vendor BLAS substantially (SeisSol's generated kernels beat cuBLAS batched
  GEMM by 2.5× on average on V100). This matters for `gale`: writing our own
  Rust-emitted kernels is not just acceptable, it can be *faster* than calling a
  library. **[V]**

- **For incompressible NS**, a proven recipe is a **semi-implicit** scheme:
  explicit treatment of the nonlinear advection term, implicit split-Stokes
  operators, with the pressure/elliptic solve done by **conjugate gradient +
  GPU-accelerated multigrid preconditioner**. **[V]**

- **For viscoelastic constitutive models** (Oldroyd-B, FENE-P), evolve the
  **log-conformation representation (LCR)** — the matrix logarithm of the
  conformation tensor — to alleviate the high-Weissenberg-number problem (HWNP).
  It alleviates but does *not* eliminate the HWNP. **[V]**

- **For compressible/hyperbolic robustness**, use **split-form / entropy-stable
  nodal DG-SEM** with summation-by-parts operators to defeat aliasing-driven
  blow-up. **[V]**

- **Immersed boundaries** combine with DG either via **sharp-interface cut cells**
  (needs small-cut-cell time-step mitigation) or **volume-penalization / forcing**
  (simpler, more robust to implement). **[V]**

- **Closest reference codebase: libParanumal** (CEED / Virginia Tech, OCCA-based)
  — high-order DG flow solvers for heterogeneous GPU/CPU, covering exactly the
  discretizations we care about. Study it first. **[V]** **PyFR** is the cleanest
  template for the *host-driver + runtime-generated-kernel* pattern `gale` should
  adopt (Python + Mako DSL + GiMMiK zero-eliding matmul; CUDA-Aware MPI). **[V]**

- **Toolchain: committed to cuda-oxide — and validated on the hardware.** A
  Milestone-1 probe on the 2× Titan V box confirmed FP64, shared memory, multi-GPU
  P2P, and FP64 libdevice math (`exp`/`ln`/`powf`) **all work on sm_70** (see
  [`milestone-1-probe-results.md`](./milestone-1-probe-results.md)). Libdevice math
  needs a 3-part recipe (pin PR #101's backend; use the `ltoir` loader; avoid
  `Option<&mut>` accessors) until two upstream fixes land — which are gale's
  contribution targets. `gale` backs cuda-oxide and **upstreams gaps** rather than
  switching away. **[V]**

- **Inter-GPU communication:** the proven DG pattern is **CUDA-Aware MPI /
  GPUDirect RDMA** (PyFR passes device pointers straight to MPI; Kirby & Mavriplis
  measured +24% over non-GPUDirect on Volta V100s). Note the Titan V has **no
  NVLink** — inter-GPU traffic crosses PCIe, so overlap communication with
  interior-element compute. **[V]**

- **Reality of the bottleneck:** the *assembled* matrix-free DG operator is
  **memory-bandwidth-bound** (well-tuned code runs within ~10% of peak bandwidth),
  even though the *isolated* sum-factorization kernel is compute-bound. Optimize
  for bytes-moved: SoA layout, kernel fusion, on-chip reuse. **[V]**

---

## 2. Why DG, and why DG on GPUs

Discontinuous Galerkin combines two traditions: the high-order accuracy and
spectral convergence of finite-element/spectral methods, and the local
conservation and upwind-stabilized flux handling of finite-volume methods. Each
element carries its own polynomial solution; elements communicate *only* through
numerical fluxes evaluated on shared faces.

The key consequence for hardware: **the majority of the DG operator is applied
independently per element, with weak penalty coupling between elements.** This
produces strong memory-access locality, which is the central reason DG runs well
on commodity GPUs. The seminal demonstration (Klöckner, Warburton, Bridge,
Hesthaven, *J. Comput. Phys.* 2009) measured **40–60× over a single serial CPU
core and >200 GFLOP/s** net application throughput on a single Nvidia GTX 280.
**[V]**

This is a *structural* property, not a benchmark artifact, so it transfers to
newer hardware and to `gale`'s Titan Vs — though the *absolute* multiplier will
differ. **Caveat:** most headline speedups are quoted against *serial / single
core* CPU baselines. Against a 64-core EPYC 7702P running a well-vectorized
multicore solver, the realistic GPU advantage is much smaller (single-digit to
low-double-digit ×, not 40–60×). Set expectations accordingly. **[V, caveat]**

At scale, the Kirby & Mavriplis result (*"30× Speedup on 345 Billion Unknowns"*,
SC20) demonstrated DG for compressible Euler via OCCA reaching **30× time-to-
solution over CPU-only up to 1,536 V100s** and strong scaling to **6,144 V100s**.
Note this is compressible Euler on Cartesian meshes — not viscoelastic or
incompressible — so it bounds the *infrastructure* potential, not our exact
problem. **[V, caveat]**

---

## 3. DG formulation primer

### 3.1 Nodal vs. modal

- **Nodal DG** represents the solution by its values at a set of interpolation
  nodes inside each element (e.g. Legendre–Gauss–Lobatto, LGL). Operators
  (differentiation, mass, lift) are precomputed dense matrices acting on nodal
  vectors. This is the Hesthaven–Warburton "Nodal DG" book formulation and the
  most common starting point. **[E]**
- **Modal DG** represents the solution by coefficients of an orthogonal
  polynomial basis (e.g. Legendre/Jacobi, or the orthonormal Dubiner basis on
  simplices). ADER-DG codes like SeisSol are modal. **[E]**

The two are related by a Vandermonde transform; many codes move between them.

### 3.2 DG-SEM (spectral element flavor)

**DG-SEM** is nodal DG on quad/hex elements where the interpolation and quadrature
nodes coincide at LGL points. This *collocation* makes the mass matrix diagonal
and enables **sum factorization** — the tensor-product structure that gives DG its
GPU-friendly arithmetic intensity (see §5). This is the variant most GPU DG
performance work builds on, and the recommended target for `gale`. **[E, with V
support for sum factorization]**

### 3.3 The DG operator structure

For a conservation law `∂ₜu + ∇·f(u) = 0`, the per-element semi-discrete form is:

```
M (du/dt) = -S·f(u)  +  L·(f* - f)|_faces
            └ volume ┘   └─ surface/flux ─┘
```

- `M` — mass matrix (diagonal in DG-SEM with LGL collocation)
- `S` — stiffness/differentiation operator (volume term)
- `L` — lift operator mapping face data to element interior
- `f*` — the **numerical flux** on faces (the only inter-element coupling)

### 3.4 Numerical fluxes

- **Hyperbolic / advective terms:** approximate Riemann solvers — Rusanov/local
  Lax–Friedrichs (cheap, robust, dissipative), HLL/HLLC, Roe. Start with Rusanov.
  **[E]**
- **Diffusive / viscous terms:** the **Local DG (LDG)** flux or the **interior
  penalty (IP)** method. libParanumal's incompressible solver uses interior-
  penalty DG (or continuous FEM) for the elliptic operators. **[V for
  libParanumal's use of IP-DG]**

### 3.5 Robustness: aliasing and split form

Classical nodal DG-SEM for nonlinear compressible flow suffers
robustness/stability failures rooted in **aliasing of the nonlinear flux terms**.
Building a robust scheme requires a particular **summation-by-parts (SBP)**
differentiation matrix together with a **split-form** discretization of the
advective fluxes (the entropy-stable / split-form DGSEM consensus from
Kopriva, Winters, Gassner et al.). If `gale` ever touches compressible flow, do
not skip this. **[V]**

### 3.6 Basis functions & quadrature

- **Lagrange (nodal) bases** at LGL points — standard, what libParanumal uses.
  **[V for libParanumal]**
- **Bernstein–Bézier bases** give derivative/lift operators with sparse special
  structure (≈ `d+1` nonzeros per row), enabling optimal-complexity,
  quadrature-free evaluation that *outperforms* nodal DG kernels in time-explicit
  GPU solvers — **but only at high polynomial order**; at low order, nodal kernels
  are competitive or better. A later-stage optimization, not a starting point.
  **[V, caveat]**

### 3.7 Time integration

- **Explicit:** strong-stability-preserving Runge–Kutta (**SSP-RK3**) is the
  standard explicit DG integrator; low-storage RK (LSERK4) is common on GPUs for
  its memory footprint. Subject to a CFL limit that *tightens like ~1/p²* with
  order `p`. **[E]**
- **IMEX / semi-implicit:** treat stiff terms (diffusion, acoustics, the
  incompressible pressure constraint) implicitly and advection explicitly. This is
  the proven path for incompressible NS (§4.1). **[V]**

---

## 4. Incompressible & viscoelastic flow

### 4.1 Incompressible Navier–Stokes with DG

The verified, GPU-proven recipe (Karakuş, Chalmers, Świrydowicz & Warburton,
*JCP* 2019; libParanumal lineage):

- **Semi-implicit time stepping:** explicit nonlinear/advection term, implicit
  split Stokes operators (a velocity-Helmholtz solve + a pressure-Poisson solve).
- **Pressure system:** conjugate gradient (**CG**) with a **fully GPU-accelerated
  multigrid preconditioner**.
- Dominant kernels tuned (fine-grain parallelism, bandwidth) close to their
  empirically predicted roofline.

**[V]** — This is the single most directly applicable result for `gale`'s
incompressible target. The pressure-Poisson solve is the hard part and the
performance bottleneck; the multigrid-preconditioned CG is what makes it tractable
on GPU.

> **→ See [`implicit-solver-strategy.md`](./implicit-solver-strategy.md)** for the
> full, verified plan: the recommended **dual-splitting** time scheme (circumvents
> LBB → equal-order interpolation), the **matrix-free p-multigrid + Chebyshev-Schwarz**
> pressure-Poisson solver, **log/SRCR** viscoelastic stabilization, the failure-mode
> watchdogs to instrument, and the scalar-elliptic-first build order. Key risk it
> surfaces: assembled-AMG multi-GPU scaling was *refuted* → prefer matrix-free on the
> 2× Titan V.

### 4.2 Viscoelastic constitutive models & the HWNP

Viscoelastic flow adds a polymeric stress tensor evolved by a constitutive law:

- **Oldroyd-B** — linear dumbbell model, simplest; unbounded extension (can blow
  up in extensional flow).
- **FENE-P** — finitely-extensible nonlinear elastic, bounded, more physical.

Both are advected tensor transport equations coupled to momentum. The notorious
**high-Weissenberg-number problem (HWNP)**: at high elasticity, the standard
formulation loses positive-definiteness of the conformation tensor and the
simulation blows up.

**Fix: the log-conformation representation (LCR)** (Fattal & Kupferman, *JNNFM*
2004/2005). Evolve the **matrix logarithm** of the conformation tensor:

- Preserves positive-definiteness by construction.
- Linearizes the exponential stress profiles that defeat polynomial bases.
- Enables stable simulation at Weissenberg numbers unreachable by the standard
  formulation; now the dominant stabilization across FEM/FVM/SPH/LBM/spectral and
  OpenFOAM/rheoTool.

**Critical caveat:** LCR *alleviates* but does **not solve** the HWNP. Convergence
is still not obtained in localized stress-singularity regions, and mesh
convergence remains challenging. **[V, caveat]**

> **Important gap.** The verified literature does **not** contain a unified GPU DG
> solver for *incompressible viscoelastic* flow (DG + log-conformation
> Oldroyd-B/FENE-P) at scale. `gale` will likely have to **combine** the
> incompressible-DG recipe (§4.1) and the LCR-viscoelastic literature itself.
> This is the project's genuine research contribution — and its main risk. **[V
> for the gap]**

---

## 5. Mapping DG to the GPU

### 5.1 Matrix-free + sum factorization

The central performance principle (Kronbichler & Kormann, *ACM TOMS*):

- **Matrix-free** operator evaluation via **sum factorization** is *fully
  compute-bound* with high flop/byte.
- Sum factorization reduces per-cell cost from `O(p^(2d))` to `O(d·p^(d+1))` and
  expresses the work as **matrix–matrix products that vectorize as fused
  multiply-add (FMA)**.
- A *stored* matrix–vector product runs at only ~2 flops per matrix element —
  memory-bound, and poor on bandwidth-limited GPUs.
- **Real-hardware caveat:** the full operator often reaches only ~half peak,
  because input/output vector loads remain partly memory-bound.

**[V]** — Conclusion for `gale`: never assemble global matrices for the explicit
operator. Keep everything matrix-free, tensor-product, and on quad/hex elements
so sum factorization applies. (On simplices the tensor structure is lost and the
benefit shrinks. **[V, caveat]**)

### 5.2 The kernels: small batched GEMM

DG operator application is, concretely, **many small dense matrix multiplications**
(one element's nodal vector times an operator matrix), batched across all
elements. This is the compute core.

**Code-generated CUDA beats vendor BLAS here.** SeisSol's generated small batched
GEMM kernels (YATeTo DSL + GemmForge) outperform **cuBLAS batched GEMM by 2.5× on
average** on V100, via better memory utilization; a 2024 follow-up (ChainForge,
fused GEMMs) adds a further ~60%. **[V]**

Implication for `gale`: emitting our own specialized kernels from Rust (with
compile-time-known small matrix dimensions `p`, exploiting shared memory and
register blocking) is the *high-performance* path, not a fallback. The matrix
dimensions are tiny and fixed per polynomial order — ideal for code generation /
const generics.

### 5.3 Data layout, on-chip memory, roofline

> **Verified framing [V]:** at the *full-operator* level, matrix-free high-order
> DG is **memory-bandwidth-bound**, not compute-bound. Kronbichler & Kormann
> found isolated sum-factorization kernels reach ~50–60% of arithmetic peak, but
> *full* operator evaluation reaches only about *half that* — limited by loading
> input/output vectors, ghost/halo exchange, variable coefficients, and geometry
> — and a well-optimized implementation runs **within ~10% of available memory
> bandwidth**. (Caveat: that study is CPU/Intel/MPI; the memory-bound *conclusion*
> is architecture-agnostic and a reliable design guide, but the exact percentages
> don't transfer to GPU.) This is *why* the layout choices below matter: minimize
> bytes moved, maximize reuse. The §5.1 "compute-bound" claim is about the
> *isolated kernel*; the *assembled operator* is bandwidth-bound — both are true.

The concrete layout guidance below is standard GPU-DG engineering practice (**[E]**
except where the bandwidth-bound principle above makes it **[V]**-grounded):

- **SoA over AoS** for field storage so that consecutive threads read consecutive
  memory (coalesced global loads). Layout fields as `[field][element][node]` or
  blocked variants; benchmark `[element][node][field]` vs `[field][node][element]`
  for your access pattern.
- **One thread-block per element (or per few elements)**; stage the element's
  nodal data and the (small, shared) operator matrices into **shared memory**,
  then do the GEMM from shared/registers. The operator matrices are reused across
  all elements → keep them resident.
- **Register blocking** for the innermost sum-factorization loops.
- **Roofline:** at low order the operator is closer to memory-bound; at high order
  it moves toward compute-bound. Profile against the Titan V's FP64 peak (~6.9
  TFLOP/s FP64; ~653 GB/s HBM2) — note Titan V has *strong* FP64, unusual for a
  consumer-class card, which suits double-precision DG.
- **Fuse kernels** where possible (volume + surface + lift) to avoid round-trips
  to global memory; ChainForge-style fusion gave measurable wins. **[partly V]**

---

## 6. Multi-GPU & domain decomposition

> The second research pass added **direct evidence** here. The
> established inter-GPU pattern in real DG/FR codes is **CUDA-Aware MPI /
> GPUDirect RDMA**: PyFR's v2.0.3 backend passes **GPU device pointers directly to
> MPI routines** (via `mpi4py`), exploiting GPUDirect RDMA on NVIDIA (and the HIP
> analogue on AMD), with a backend-independent message format that even allows
> different ranks to run different backends. **[V]** Kirby & Mavriplis measured a
> **24% speedup from CUDA-Aware MPI vs non-GPUDirect** communication on 32 V100s,
> on top of 30× over CPU and strong scaling to 6,144 V100s. Crucially, **V100 is
> the same Volta generation as gale's Titan V**, so this pattern transfers
> directly. **[V]** Partition/load-balancing specifics (METIS, space-filling
> curves) below remain **[E]**.

DG's locality makes domain decomposition natural — partition elements across GPUs;
the *only* inter-GPU data is **face/trace data on partition boundaries**.

Recommended structure for a 2× Titan V box:

- **Partition** the element mesh (e.g. by space-filling curve, or METIS for
  unstructured) into one subdomain per GPU, balancing element counts.
- **Halo exchange of face traces:** before the surface/flux kernel, exchange the
  boundary-face nodal values with the neighboring partition. The volume of data is
  `O(surface)` while compute is `O(volume)` — favorable surface-to-volume ratio at
  high order.
- **Overlap compute and communication:** compute interior-element volume terms
  while the face-trace exchange is in flight, then do boundary elements. This
  hides latency and is the standard DG multi-GPU trick.
- **Two GPUs in one node:** **CUDA-Aware MPI / GPUDirect RDMA** is the
  battle-tested path used by PyFR and Kirby & Mavriplis and gave a measured **+24%
  over non-GPUDirect** on Volta V100s **[V]**. For a single 2-GPU node, **CUDA
  peer-to-peer** (`cudaMemcpyPeer`) with explicit streams is the simplest
  lower-overhead alternative; **NCCL** is the option for collectives. Note the
  Titan V has **no NVLink**, so inter-GPU traffic crosses **PCIe** — keep the
  trace-exchange volume small (favorable at high order) and overlap aggressively.
- **Load balancing:** with two identical Titan Vs, equal element counts suffice
  *unless* immersed-boundary cut cells concentrate work — then weight the
  partition by per-element cost. **[E]**

> ⚠️ Note on cuda-oxide: multi-GPU / P2P support is **not documented** in its
> README (§9) **[V]** — validate that the chosen Rust CUDA layer exposes
> multi-context / P2P / stream / CUDA-Aware-MPI APIs *before* committing to it for
> multi-GPU. The Rust CUDA project's `cust` host crate is the safer bet for this
> plumbing today.

---

## 7. Immersed boundary method + DG

`gale` wants first-class IBM. Two verified families:

### 7.1 Sharp-interface cut cells **[V]**

Represent the (possibly moving) solid by cutting it out of the background mesh;
elements partially covered by the solid become **cut cells**.

- BoSSS (Computers & Fluids 2017) builds a higher-order IBM for moving rigid
  bodies on a DG discretization of incompressible NS with sharp-interface cut
  cells (rigid motion via Newton's equations; hierarchical moment fitting for
  forces).
- Xiao, Febrianto, Zhang & Cirak (arXiv:1902.10232) formulate immersed high-order
  DG for compressible NS on non-boundary-fitted unstructured/simplicial meshes via
  an implicit signed-distance function.

**The key challenge:** tiny "sliver" cut cells impose an **excessive
stable-time-step restriction** on explicit DG. Mitigations:
1. Replace cut-cell basis functions with **extrapolated basis functions from the
   nearest largest element**, or
2. **Scale cut-cell basis functions by the solid-covered fraction**.

**Pros:** sharp, high-order-accurate boundary representation, accurate forces.
**Cons:** geometrically intricate (quadrature on cut cells, moment fitting), and
the small-cell time-step problem is real. **[V]**

### 7.2 Volume-penalization / forcing **[V]**

Add a forcing/penalty term that drives the velocity toward the solid's velocity
inside the body — no mesh cutting.

- A high-order IBM for fluid–structure interaction combining **volume
  penalization** with a **high-order nodal DG solver** (arXiv:2512.05733, Dec
  2025), corroborated by JCP / Computers & Fluids volume-penalization-for-high-
  order-DG papers.

**Pros:** dramatically simpler to implement (just a source term), no cut-cell
geometry, GPU-friendly (stays element-local). **Cons:** lower-order boundary
accuracy near the interface; penalty parameter introduces stiffness (may need
implicit/IMEX treatment) and a smeared interface.

### 7.3 Recommendation for gale

**Start with volume penalization.** It keeps the discretization element-local and
GPU-friendly, avoids cut-cell quadrature and the small-cell time-step trap, and
gets a moving body working fast. Graduate to sharp cut cells later if/when
boundary-force accuracy demands it. **[E, grounded in V trade-offs]**

---

## 8. Software landscape to learn from

| Project | Method | GPU strategy | Why study it for `gale` |
|---|---|---|---|
| **libParanumal** | DG/SEM: upwind DG (compressible NS), penalty-flux DG (Boltzmann), IP-DG / CFEM (incompressible) on tri/quad/tet/hex, Lagrange bases **[V]** | OCCA (portable JIT kernels), matrix-free, multigrid-preconditioned CG | **The closest reference architecture.** CEED/DOE, Tim Warburton's group. Covers our exact discretizations incl. incompressible. Read its kernels. **[V]** |
| **PyFR** | Flux Reconstruction (FR, a DG superset) for advection–diffusion | Python host + **Mako-based DSL** (PyFR-Mako) generating pointwise kernels at runtime; **GiMMiK** auto-tunes fully-unrolled, zero-eliding matmul kernels vs cuBLAS and picks the fastest per op (libxsmm on CPU); CUDA/OpenCL/HIP/Metal; CUDA-Aware MPI **[V]** | **The canonical pattern to mirror** — high-level host driver + runtime-generated low-level kernels. GiMMiK's zero-elision gave ~10–63× over cuBLAS *for sparse DG operator matrices*. Exactly the Rust-host + emitted-kernel split `gale` should use. **[V]** |
| **Trixi.jl** | Nodal DG-SEM, entropy-stable/split-form, adaptive | **CPU-only in-tree** (multithreading + MPI); GPU only via separate **TrixiCUDA.jl** (WIP, GSoC 2023, semidiscretization kernels only) **[V]** | Systems-level language doing DG with strong split-form/entropy-stability — good *design* reference, but its GPU path is **comparatively immature** vs PyFR/libParanumal. **[V]** |
| **deal.II** | Matrix-free FEM/DG, sum factorization | CUDA / Kokkos matrix-free | Canonical matrix-free sum-factorization implementation (Kronbichler). Read for the operator-evaluation algorithms. **[V for the matrix-free principle]** |
| **MFEM** | High-order FEM/DG | Partial-assembly matrix-free, GPU via RAJA/Kokkos/CUDA | Another mature matrix-free/partial-assembly reference (also CEED). **[E]** |
| **SeisSol** | Modal ADER-DG (seismology) | Code-generated small batched GEMM (YATeTo/GemmForge), beats cuBLAS 2.5× | The reference for **how to generate fast small-GEMM kernels** — directly relevant to emitting kernels from Rust. **[V]** |
| **FLEXI / Fluxo** | DG-SEM, split-form | GPU ports ongoing | Split-form/entropy-stable DGSEM reference codes. **[E]** |
| **NUMA / Nektar++** | Spectral/hp element, DG & CG | GPU work varies | Mature spectral-element ecosystems for breadth. **[E]** |

**Lessons distilled for a Rust implementation:**
1. Adopt the **host-driver + generated-kernel split** (PyFR, libParanumal/OCCA,
   SeisSol) **[V]**. Rust is the host; emit/compile specialized PTX per polynomial
   order. PyFR's GiMMiK shows the highest-value version: **generate fully-unrolled
   kernels that elide multiply-by-zero** in the (sparse) operator matrices and
   autotune them against cuBLAS — for `gale`, CubeCL's `comptime` or Rust const
   generics are the natural way to express this. **[V]**
2. Stay **matrix-free** and **tensor-product** (deal.II, libParanumal).
3. Use **multigrid-preconditioned CG** for the implicit pressure solve
   (libParanumal). **[V]**
4. Build in **split-form/entropy-stability** from the start if compressible is on
   the roadmap (Trixi.jl, FLEXI). **[V for the need]**

---

## 9. The Rust GPU landscape

> This section was **upgraded from [E] to [V]** by a dedicated second research
> pass (24 sources, 25 claims verified 3-0). The toolchain findings are
> **time-sensitive** — cuda-oxide is v0.1.0 (May 2026) and the Rust CUDA revival
> is explicitly "bumpy/unstable" — so re-verify against the live repos before
> committing. **Bottom line: cuda-oxide (gale's current scaffold) is the most
> elegant pure-Rust→PTX path but the least mature, and two features gale needs —
> FP64 and multi-GPU/P2P — are *not documented*; the revived Rust CUDA project is
> the more production-ready alternative today.**

### 9.1 The three serious options

**cuda-oxide (NVlabs)** — *what gale is scaffolded on.* A **custom `rustc`
codegen backend** compiling idiomatic Rust directly to PTX via
`Rust → MIR → Pliron IR → LLVM IR → PTX` (the final IR→PTX step delegates to
LLVM's NVPTX backend / `llc` after emitting a `.ll` — exactly the `gale.ll`
[`nvptx64-nvidia-cuda`] and `gale.ptx` artifacts already in the repo). It is a
*compiler*, not merely a host-side CUDA wrapper, and also ships host crates
(`cuda-core`, `cuda-async`). It documents rich device-side abstractions: shared
memory (`SharedArray`/`DynamicSharedArray`), scoped atomics, barriers
(`thread::sync_threads` → `bar.sync`), warp intrinsics (`shuffle_xor_f32`,
`warp_reduce_sum`), TMA, MMA/TMEM, cluster programming with DSMEM, and host-side
streams/concurrent execution. **[V]**

> ⚠️ **Critical caveats for gale [V]:**
> - **v0.1.0 alpha (≈May 2026)** — README explicitly warns of bugs, incomplete
>   features, and API breakage; a known-unsound `index_2d(stride)` case exists.
> - **TMA, MMA/TMEM, and thread-block clusters require Hopper/Blackwell** (sm_90 /
>   sm_100) and are **unusable on the Titan V (Volta, sm_70)** — so cuda-oxide's
>   marquee features don't apply to gale's hardware anyway.
> - **FP64 and multi-GPU / peer-to-peer are *not documented* in the README** —
>   both are central to gale (double-precision physics; 2-GPU box). This is
>   absence of evidence, not evidence of absence — **probe both in Milestone 1.**

**Rust CUDA project (`rustc_codegen_nvvm` + `cust` host crate)** — *the more
production-ready alternative.* Compiles Rust to **NVVM IR** (NVIDIA's LLVM-based
CUDA frontend, legacy LLVM 7 dialect) → PTX, via
`MIR → SSA codegen → NVVM IR → PTX → PTX opts → final PTX` (built with
`cuda_builder`). Exposes the full memory hierarchy in device code — registers,
shared (~48 KB), constant (64 KB total), global — with manual placement via
`#[cuda_std::address_space(constant/global)]` (directly useful for resident
operator matrices and shared-memory blocking). **Revived Jan 2025** after 3+ years
dormant, maintained by a small team (C. Legnitto, J. Ortega), advanced to
`nightly-2025-03-02` with CUDA 12.x CI; explicitly "active-but-unstable" and
seeking maintainers. **[V]** Different programming model than cuda-oxide, but a
longer track record and host-side ecosystem. **Strong fallback if cuda-oxide
blocks on FP64/multi-GPU.**

**CubeCL (Tracel / Burn ecosystem)** — *the JIT/portability option.* GPU compute
from a single `#[cube]` Rust source targeting CUDA, ROCm/HIP, Metal & Vulkan (via
wgpu), WebGPU, and CPU SIMD. **JIT-compiles only the launched kernel variants**
and provides a **`comptime` mechanism that rewrites the compiler IR at first
compile** for instruction specialization, loop unrolling, and **shape
specialization** — plus automatic vectorization and autotuning. **[V]** This
comptime-specialization model is the closest Rust analogue to PyFR's runtime
kernel generation (§8) and is genuinely well-suited to DG's order-parameterized
operator kernels. Alpha (expect breaking changes) but **proven in production by
Burn**, which drove its design. **[V]**

### 9.2 Toolchain decision: cuda-oxide (committed)

**`gale` commits to cuda-oxide.** This is a deliberate direction choice: it is the
purest realization of "native Rust all the way to PTX" — no DSL, no FFI, no
foreign IR dialect — and the project treats advancing that vision (including
**upstreaming features `gale` needs**) as part of its mission. Rust-CUDA and C/C++
CUDA are kept as *reference* for what missing features should look like, **not as
fallbacks to switch to.**

The other Rust GPU stacks, for reference only:

| Criterion | **cuda-oxide (chosen)** | Rust CUDA (`cust`) — *reference* | CubeCL — *reference* |
|---|---|---|---|
| Maturity | v0.1.0 alpha **[V]** | active-but-unstable, longer history **[V]** | alpha, production-used by Burn **[V]** |
| Codegen path | Rust→PTX (LLVM NVPTX) **[V]** | Rust→NVVM IR (LLVM 7)→PTX **[V]** | JIT, multi-backend **[V]** |
| FP64 on Volta | verify in M1 (see 9.3) **[V]** | supported (general LLVM/NVVM) **[E]** | backend-dependent **[E]** |
| Multi-GPU / P2P | verify in M1 (see 9.3) **[V]** | via `cust` host APIs **[E]** | via runtime **[E]** |
| Order-specialized kernels | const generics **[E]** | const generics **[E]** | `comptime` specialization **[V]** |
| What it teaches gale | — | host-side multi-GPU API shape; NVVM intrinsic coverage | runtime JIT + comptime specialization à la PyFR |

### 9.3 cuda-oxide gap analysis → upstream-contribution roadmap

Since the plan is to *back* cuda-oxide, the productive framing is: what does `gale`
need that C/C++ CUDA (and Rust-CUDA) provide, that cuda-oxide may not yet? Each gap
is a candidate contribution. A **live repo snapshot with issue/PR analysis and a
prioritized contribution roadmap** is maintained in
[`cuda-oxide-repo-status.md`](./cuda-oxide-repo-status.md) — read it alongside this
table.

> ✅ **Answered empirically (2026-06-01) — probe run on the 2× Titan V box; see
> [`milestone-1-probe-results.md`](./milestone-1-probe-results.md). Nothing is
> hard-blocked on sm_70:**
> - FP64 arithmetic, shared memory + barriers, and **multi-GPU P2P (dev0↔dev1 over
>   PCIe) all PASS**. The P2P primitives (`cuda_core::peer::*`, `memcpy_dtod_async`)
>   exist and work — multi-GPU is **not** greenfield/blocked.
> - **FP64 libdevice math (`exp`/`ln`/`powf`) also PASSES** (worst rel err 3e-15)
>   via a 3-part recipe: pin **PR #101**'s backend (fixes opaque→typed NVVM IR,
>   issue #98); use the file-based `ltoir` loader (`load_kernel_module` +
>   `cuda_launch!`), not the embedded `#[cuda_module]` path; and avoid
>   `Option<&mut>` slice accessors (use `get_unchecked_mut` — a residual #101
>   missing-bitcast bug). Pure-arith/shared-mem kernels have none of these
>   constraints. gale's contribution targets: fix #101's bitcast gap + validate
>   #101 on sm_70.

| Capability gale needs | C/C++ CUDA | cuda-oxide status (≈May 2026) | Action |
|---|---|---|---|
| **FP64 device arithmetic** | native | **not documented** in README **[V]** | Write an FP64 probe kernel M1; if codegen is wrong/missing, fix the NVPTX lowering upstream |
| **Multi-GPU / multi-context** | CUDA driver API | **not documented** **[V]** | Probe `CudaContext` per device; contribute multi-context host support if absent |
| **Peer-to-peer (`cudaMemcpyPeer`)** | native | **not documented** **[V]** | Needed for the 2-GPU trace exchange (§6); likely a host-crate addition |
| **CUDA-Aware MPI / NCCL interop** | via libs | unknown | Need device-pointer-to-MPI handoff (the PyFR pattern, §6); may be host-side glue |
| **Shared memory + barriers** | native | **documented** (`SharedArray`, `sync_threads`) **[V]** | ✅ available — use directly |
| **Warp intrinsics** (shuffle/reduce) | native | **documented** (`shuffle_*`, `warp_reduce`) **[V]** | ✅ available |
| **Atomics** | native | **documented** (scoped atomics) **[V]** | ✅ available |
| **Streams / async** | native | **documented** (`cuda-async`, streams) **[V]** | ✅ available — basis for compute/comm overlap |
| TMA / MMA / TMEM / clusters | Hopper+ | documented but **needs sm_90+** **[V]** | N/A on Volta sm_70 — ignore for gale |

So the genuinely on-chip kernel primitives `gale` needs (shared memory, warp ops,
atomics, streams) **appear already present** — the open questions are concentrated
in **FP64 codegen** and the **host-side multi-GPU surface** (contexts, P2P,
MPI/NCCL handoff). Those two are the highest-leverage probe-then-contribute
targets.

- **Hand-written PTX / `.cu` via FFI** remains a temporary escape hatch for any
  single kernel cuda-oxide can't yet express — but prefer fixing cuda-oxide so the
  whole codebase stays native Rust. The repo already emits PTX, so the diagnostic
  path (inspect `gale.ll`/`gale.ptx`) is open. **[E]**

**Structuring a DG-on-GPU codebase in Rust [E]:**
- Use **const generics** for polynomial order `p` and dimension `d` so operator
  sizes are compile-time constants → the compiler/codegen can fully unroll the
  small GEMMs (mirrors the SeisSol code-generation win, but at Rust compile time).
- Keep a clean **host/device boundary**: host owns mesh, partitioning, time loop,
  MPI/P2P; device owns volume/surface/lift/flux kernels and the CG/multigrid
  iterations.
- **CPU/GPU dispatch flexibility** (a `CLAUDE.md` requirement): define the operator
  evaluation behind a trait with CPU (rayon) and GPU (cuda-oxide) backends so the
  same DG core runs on the EPYC or the Titan Vs. Validate correctness on CPU,
  scale on GPU.
- Precompute reference-element operator matrices on the host (CPU), upload once,
  keep resident in device constant/shared memory.

---

## 10. Recommended roadmap for gale

A pragmatic, de-risked path. Each milestone is independently testable.

### Milestone 0 — Foundations (CPU first)
- Reference element: **nodal DG-SEM on quads (2D)**, LGL nodes, tensor-product
  basis. Polynomial order `p` as a const generic (start `p=3`/`p=4`).
- Precompute `D` (differentiation), `M` (diagonal mass), `L` (lift) operators.
- **CPU reference solver** for a scalar **linear advection** equation with a
  **Rusanov flux** and **SSP-RK3** time stepping. This is the correctness oracle.
- Verify spectral convergence against a smooth manufactured solution.

### Milestone 1 — First GPU kernels
- Port the advection operator to GPU via cuda-oxide: volume + surface/lift +
  flux kernels, matrix-free, one block per element, operators in shared memory
  (cuda-oxide's `SharedArray` is documented **[V]**).
- **cuda-oxide gap probes (§9.3): ✅ done** — `cargo oxide run --bin probe-sm70`
  confirmed FP64, shared memory, and multi-GPU P2P all work on the Titan Vs. The
  one blocker is libdevice math (issue #98 / PR #101); until #101 is pinned,
  **keep the advection flux/norm math libdevice-free or use the #101 fork.** See
  [`milestone-1-probe-results.md`](./milestone-1-probe-results.md).
- Validate bit-for-bit-ish against the CPU oracle; profile against roofline.

### Milestone 2 — Nonlinear & systems
- Move to a **nonlinear system**: 2D compressible Euler *or* go straight to the
  incompressible target. If touching nonlinear hyperbolic, adopt **split-form /
  SBP** operators now to avoid aliasing blow-up. **[V]**

### Milestone 3 — Incompressible Navier–Stokes
- Implement the **semi-implicit** scheme: explicit advection, implicit split
  Stokes. **[V]**
- Build the **pressure-Poisson solve as multigrid-preconditioned CG** on GPU —
  budget the most effort here; it is the bottleneck and the hardest kernel set.
  **[V]**
- Validate: Taylor–Green vortex, lid-driven cavity, flow past cylinder.

### Milestone 4 — Immersed boundary
- Add **volume-penalization IBM** (forcing term) for a fixed then moving rigid
  body. **[V]** Simpler than cut cells and stays element-local. **[E]**
- Validate forces against flow-past-cylinder benchmarks.

### Milestone 5 — Viscoelasticity
- Add a polymeric stress field with **Oldroyd-B in the log-conformation
  representation**. **[V]** Then **FENE-P**.
- Validate against known viscoelastic benchmarks (e.g. 4:1 contraction,
  flow past cylinder at increasing Wi). Expect HWNP-related convergence limits
  near stress singularities. **[V, caveat]**

### Milestone 6 — Multi-GPU
- Partition elements across the 2 Titan Vs; **face-trace halo exchange** with
  **compute/communication overlap** (interior elements while traces are in
  flight). **[E]**
- Use CUDA **P2P** (single node) first; consider NCCL/CUDA-Aware MPI if scaling
  beyond the box. Confirm the Rust layer exposes the needed APIs. **[E]**

### Milestone 7 — Python entry point
- Expose the configured solver via **pyo3** (a `CLAUDE.md` goal) for setup,
  driving, and analysis from Python.

### Starting choices, summarized
| Decision | First choice | Rationale |
|---|---|---|
| DG variant | **Nodal DG-SEM on quads/hexes** | Diagonal mass + sum factorization; GPU-friendly **[V]** |
| Element shape | **Quad (2D) → hex (3D)** | Tensor-product → sum factorization applies **[V]** |
| Advective flux | **Rusanov / local Lax–Friedrichs** | Cheap, robust, simple to implement **[E]** |
| Viscous flux | **Interior penalty (IP-DG)** | Used by libParanumal incompressible solver **[V]** |
| Time integrator (explicit) | **SSP-RK3** (LSERK4 for memory) | Standard, stable DG explicit integrator **[E]** |
| Incompressible time scheme | **Semi-implicit (explicit advection + implicit split Stokes)** | GPU-proven **[V]** |
| Pressure solve | **CG + GPU multigrid preconditioner** | GPU-proven bottleneck solver **[V]** |
| Viscoelastic model | **Oldroyd-B then FENE-P, log-conformation** | Alleviates HWNP **[V]** |
| IBM | **Volume penalization** first, cut cells later | Simpler, element-local, no small-cell CFL trap **[V/E]** |
| Robustness (if compressible) | **Split-form / entropy-stable SBP** | Defeats aliasing blow-up **[V]** |

---

## 11. Open questions

Two research passes resolved much of the toolchain/multi-GPU/software detail (now
folded into §§5,6,8,9). These remain genuinely open and material to `gale`:

1. ~~cuda-oxide FP64 + multi-GPU on Volta~~ **— RESOLVED (2026-06-01).** Probe on
   the Titan Vs proved FP64, shared memory, and multi-GPU P2P all work on sm_70.
   Remaining sub-task: **land/pin cuda-oxide PR #101** so libdevice math
   (`sqrt`/`exp`/`log`) compiles on sm_70 (issue #98). See
   [`milestone-1-probe-results.md`](./milestone-1-probe-results.md).
2. **Unified incompressible-viscoelastic GPU DG:** Still no published work
   demonstrating DG + log-conformation Oldroyd-B/FENE-P for incompressible flow at
   scale on GPU. `gale` must combine the two literatures — the real research risk.
3. **2-GPU interconnect choice (no NVLink):** For PCIe-connected Titan Vs, is
   CUDA-Aware MPI (GPUDirect), NCCL, or direct `cudaMemcpyPeer` fastest for
   face/halo trace exchange, and how much does interior/boundary
   compute–communication overlap actually recover? Settle empirically in
   Milestone 6.
4. **Concrete DG layout numbers:** Published AoS-vs-SoA benchmarks and
   shared-memory/register-blocking + kernel-fusion (volume+surface+lift)
   granularity for matrix-free sum-factorization kernels *on NVIDIA GPUs* remain
   thin — benchmark directly on Volta.
5. **IBM trade-off:** Quantify the small-cut-cell time-step penalty vs.
   volume-penalization accuracy loss for `gale`'s explicit GPU integrator before
   committing long-term.

---

## References

All sources below were retrieved and (except where noted) contributed claims that
passed 3-0 adversarial verification in the research pass.

**Theory & formulation**
- Kopriva, Winters, Gassner et al., *Construction of Modern Robust Nodal DG
  Spectral Element Methods for the Compressible Navier–Stokes Equations* —
  arXiv:2005.02317. (aliasing, SBP, split-form) **[V]**
- Fattal & Kupferman, *Constitutive laws for the matrix-logarithm of the
  conformation tensor*, JNNFM 126:23–37 (2005) —
  doi:10.1016/j.jnnfm.2004.12.003 (S0377025705000042); see also
  arXiv:2112.06829. (log-conformation / HWNP) **[V]**

**GPU performance & benchmarks**
- Klöckner, Warburton, Bridge, Hesthaven, *Nodal DG Methods on Graphics
  Processors*, J. Comput. Phys. (2009) — arXiv:0901.1024. (locality; 40–60×;
  >200 GFLOP/s on GTX 280) **[V]**
- Kirby & Mavriplis, *GPU-Accelerated DG Methods: 30× Speedup on 345 Billion
  Unknowns*, SC20 — arXiv:2006.15698. (OCCA; scaling to 6,144 V100s) **[V]**
- Karakuş, Chalmers, Świrydowicz & Warburton, *A GPU-accelerated high-order DG
  method for incompressible NS*, J. Comput. Phys. 390:380–404 (2019) —
  arXiv:1801.00246. (semi-implicit; multigrid-CG pressure solve) **[V]**
- Kronbichler & Kormann, *Fast matrix-free evaluation of DG operators (deal.II)*,
  ACM TOMS — arXiv:1711.10885. (sum factorization; compute-bound; ~2 flops/byte
  for stored matrices) **[V]**
- Dorozhinskii & Bader, *SeisSol on GPUs* (code-generated batched GEMM), HPC Asia
  2021 — doi:10.1145/3432261.3436753. (generated kernels beat cuBLAS 2.5×) **[V]**
- Chan & Warburton, *GPU-Accelerated Bernstein–Bézier DG Methods for Wave
  Problems*, SIAM J. Sci. Comput. 39(2) (2017) — arXiv:1512.06025.
  (Bernstein basis; high-order advantage) **[V]**

**Immersed boundary**
- *A high-order immersed boundary (cut-cell) method for moving bodies with DG for
  incompressible NS* (BoSSS), Computers & Fluids (2017) — S0045793017301706.
  **[V]**
- Xiao, Febrianto, Zhang & Cirak, *Immersed high-order DG for compressible NS on
  non-boundary-fitted meshes* — arXiv:1902.10232. (cut cells; small-cell
  time-step mitigations) **[V]**
- *A high-order immersed boundary method for FSI via volume penalization + nodal
  DG* — arXiv:2512.05733 (2025). **[V]**

**Software & architecture** *(all [V] from the second research pass)*
- libParanumal — https://github.com/paranumal/libparanumal (canonical) and
  https://github.com/CEED/libParanumal (mirror). **[V]**
- Witherden, Farrington & Vincent, *PyFR: … Flux Reconstruction on Streaming
  Architectures* — Comput. Phys. Commun. (2014), S0010465514002549;
  arXiv:1312.1638; and *PyFR v2.0.3* (CPC 2025), tarikdzanic.org/docs/cpc_pyfr2.pdf
  — Mako DSL, GiMMiK, CUDA-Aware MPI. **[V]**
- GiMMiK — https://github.com/PyFR/GiMMiK; Wozniak et al., *GiMMiK* CPC 202:12
  (2016), S0010465515004506 — zero-eliding matmul, ~10–63× vs cuBLAS for sparse.
  **[V]**
- Trixi.jl — https://github.com/trixi-framework/Trixi.jl; GPU via
  https://github.com/trixi-gpu/TrixiCUDA.jl (WIP). **[V]**
- Kronbichler & Kormann, *Fast matrix-free … (full operator is bandwidth-bound)* —
  arXiv:1711.03590 (ACM TOMS 2019); 2025 follow-up arXiv:2509.10226. **[V]**
- Vincent et al., *Heterogeneous computing in PyFR* — arXiv:1409.0405. **[V]**

**Rust GPU tooling** *([V] as of ≈May 2026 — fast-moving; re-verify)*
- cuda-oxide — https://github.com/NVlabs/cuda-oxide and
  https://nvlabs.github.io/cuda-oxide/ (incl. architecture-overview). Custom rustc
  backend, Rust→MIR→Pliron→LLVM→PTX; v0.1.0 alpha; TMA/MMA need Hopper/Blackwell;
  FP64/multi-GPU undocumented. **[V]**
- Rust CUDA project — https://rust-gpu.github.io/blog/2025/05/27/rust-cuda-update/
  and .../2025/08/11/rust-cuda-update/; https://rust-gpu.github.io/Rust-CUDA/;
  https://docs.rs/cuda_std; NVVM IR spec https://docs.nvidia.com/cuda/nvvm-ir-spec.
  `rustc_codegen_nvvm` → NVVM IR (LLVM 7) → PTX; `cust` host crate. **[V]**
- CubeCL — https://github.com/tracel-ai/cubecl; https://docs.rs/cubecl;
  crates `cubecl-cuda`, `cubecl-wgpu`. JIT + `comptime` specialization;
  production-used by Burn. **[V]**

---

*Generated from a fact-checked deep-research pass (5 search angles, 21 sources
fetched, 91 claims extracted, 25 verified 3-0). Findings tagged **[V]** are
verified; **[E]** are engineering synthesis filling gaps the verified set did not
cover. Re-verify performance figures and toolchain status periodically — method
formulation is stable, but GPU numbers and Rust GPU tooling age fast.*
