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
runs **on the GPU**, validated against a CPU reference. The four broad capabilities, each
unpacked in its own chapter:

**Spatial discretization — DG-SEM (Ch. 3–5).** We can discretize space with high-order
nodal spectral elements, with both hyperbolic and elliptic operators.

**Incompressible & viscoelastic flow (Ch. 6–8).** We can solve incompressible
Navier–Stokes and the Oldroyd-B viscoelastic model on the GPU.

**Geometry, boundaries, adaptivity (Ch. 9–10).** We can embed solids without meshing
them, and refine the mesh adaptively around them.

**Architecture (Ch. 11–12).** We can run all of it multi-GPU, validated against a CPU
oracle.

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
   checked to **match the CPU oracle to ~10⁻¹⁴ relative** (round-off, not bit-identical —
   the kernels reorder reductions and use fused multiply-adds). This is how we trust the
   GPU code at all (Chapter 12). It also means the CPU library is a clean, `std`-only
   reference you can read to understand the math without GPU noise.

3. **Put the expensive work on the GPU, keep the cheap work simple.** The flow solvers
   use a *hybrid* pattern: the costly elliptic solves (the bottleneck) run entirely
   on-device, while cheap, element-local assembly (divergence, gradient correction,
   right-hand-side construction) reuses the validated host code (Chapter 7).

4. **High order, because the application demands it.** Spectral elements give
   exponential accuracy on smooth flow, which lets us use far fewer, larger elements —
   crucial when you eventually want many particles in a domain (Chapter 3). But that
   exponential convergence holds *only where the solution is smooth*, and the
   immersed-boundary penalization of principle #1's application deliberately introduces a
   non-smooth, smeared interface right where you care most. So gale simultaneously bets on
   smoothness (high-order elements) and breaks it (penalized solids): high order is free
   accuracy on the bulk flow, but you pay for it with brutal sensitivity at sharp
   features — so you keep the features rare and refine around them. That tension is
   exactly why AMR shows up.

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
