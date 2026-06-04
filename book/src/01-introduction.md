# What gale Is, and What It Solves

## The one-sentence version

gale is a **GPU-native, high-order fluid solver** that simulates **incompressible and
viscoelastic flow around immersed objects**, written in native Rust with CUDA device
kernels, built on the **discontinuous Galerkin spectral element method (DG-SEM)**.

## The problem it is aimed at

The headline target is **viscoelastic, particle-laden, microfluidic suspensions** — for
example, polymer solutions carrying suspended particles through small channels. That one
phrase packs in every hard thing this book is about:

- **viscoelastic** — the fluid is not simple water; it has "memory," like a dilute
  polymer solution, and is described by an extra evolving *stress* (Chapter 8);
- **particle-laden** — there are solid (or deformable) objects immersed in the fluid,
  which we represent without meshing their surfaces (Chapter 9);
- **microfluidic** — small length scales and slow speeds, so **inertia is weak**
  (low Reynolds number) but **elasticity can be strong** (high Weissenberg number). This
  is the regime where viscoelastic effects dominate and instabilities are subtle;
- **suspension** — ultimately many particles, which demands **adaptivity** to resolve
  the flow near each one without paying for fine resolution everywhere (Chapter 10).

This is a genuinely hard corner of computational fluid dynamics, and it is *why* gale
makes the specific choices it does. Keep this target in mind: when a later chapter
explains why we picked a method, the answer is almost always "because of low-Re,
high-Wi, particle-laden flow."

## What is actually built (the capability map)

gale is a research code, so it is useful to be precise about what exists *today* and
runs **on the GPU**, validated against a CPU reference to machine precision. Each item
below maps to a chapter.

**Spatial discretization — DG-SEM (Ch. 3–5)**
- Nodal tensor-product spectral elements on **quadrilaterals (2D) and hexahedra (3D)**.
- **Hyperbolic operators:** linear advection, Burgers, compressible Euler; both the
  standard weak form and an **entropy-stable split form** for high-Reynolds robustness.
- **Elliptic operators:** the symmetric interior-penalty (SIPG) Poisson and Helmholtz
  operators, the workhorses of incompressible flow.

**Incompressible & viscoelastic flow (Ch. 6–8)**
- **Unsteady Stokes and Navier–Stokes** by a BDF1 dual-splitting (projection) scheme.
- **GPU linear solvers:** matrix-free conjugate gradient, p-multigrid-preconditioned CG,
  and deflated CG for the singular pressure system.
- **Viscoelasticity:** the Oldroyd-B constitutive model in both the **direct** form and
  the **log-conformation** form (the high-Weissenberg-robust representation), including a
  full **on-device 3×3 symmetric eigensolver** for the 3D log-conformation update.

**Geometry, boundaries, adaptivity (Ch. 9–10)**
- **Immersed boundaries** via the Brinkman **volume-penalization** method (2D and 3D),
  with a hydrodynamic-drag diagnostic.
- **Adaptive mesh refinement** in 2D: 2:1 non-conforming meshes with conservative,
  symmetry-preserving **mortar** coupling, a smoothness indicator, and dynamic
  refine/coarsen — and the GPU flow solver runs on these adaptive meshes.

**Architecture (Ch. 11–12)**
- Everything above runs **on the GPU**, including **multi-GPU** execution with
  peer-to-peer halo exchange across two devices.
- A **HOOMD-style simulation framework** (`Simulation`, fields, integrators, hooks) so a
  full simulation is assembled the same way whether it runs on CPU or GPU.

**The capstone.** All of these compose: an **immersed rigid particle in viscoelastic
flow, end-to-end on the GPU, in both 2D and 3D** — the project's headline capability,
realized and validated. That single test exercises the elliptic solvers, the
dual-splitting flow, the log-conformation polymer model, and the immersed-boundary
penalization together.

## The design philosophy

A few principles run through the whole codebase. They will make more sense after the
relevant chapters, but it helps to state them up front.

1. **GPU-native, not GPU-accelerated.** The device kernels are written in Rust (via the
   `cuda-oxide` toolchain) and compiled to PTX, rather than calling out to a C++/CUDA
   library. Multi-GPU is a first-class concern, not an afterthought (Chapter 11).

2. **The CPU is the oracle.** The pure-host `gale` library implements every operator on
   the CPU; the GPU crate (`gale-gpu`) re-implements them as device kernels and is
   checked **bit-for-bit** (to ~10⁻¹⁴ relative) against the CPU result. This is how we
   trust the GPU code at all (Chapter 12). It also means the CPU library is a clean,
   `std`-only reference you can read to understand the math without GPU noise.

3. **Put the expensive work on the GPU, keep the cheap work simple.** The flow solvers
   use a *hybrid* pattern: the costly elliptic solves (the bottleneck) run entirely
   on-device, while cheap, element-local assembly (divergence, gradient correction,
   right-hand-side construction) reuses the validated host code (Chapter 7).

4. **High order, because the application demands it.** Spectral elements give
   exponential accuracy on smooth flow, which lets us use far fewer, larger elements —
   crucial when you eventually want many particles in a domain (Chapter 3).

5. **Validate everything, and be honest about what is not done.** Every kernel has a
   `*-check` binary comparing it to the oracle. Features that are planned but not built
   (3D adaptivity, true two-phase flow, hp-adaptivity) are tracked, not hidden
   (Chapter 13).

## How to use the rest of this book

The natural path is Part II → III → IV, which mirrors how a simulation is built up:
*discretize space*, *solve the flow*, *add geometry and adaptivity*. But every chapter
stands on its own with a "what / why / how gale does it" structure, so you can also dip
into whichever capability you are curious about.

Next, Chapter 2 lays out the physics — the equations we are actually solving, and the
dimensionless numbers (Reynolds, Weissenberg, Deborah) that define gale's regime and
explain its priorities.
