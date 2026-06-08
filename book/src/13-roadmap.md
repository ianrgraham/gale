# Roadmap and Open Problems

This book has described what gale *is*. This chapter is honest about what it is *not yet*
— the capabilities that are designed, scoped, or partially built but not finished. For a
research code this is not an embarrassment; it is the map of where the interesting work
lives. Each item below says what exists today and what closing the gap would take.

## The build-order, and where we are on it

gale follows the build order in `docs/implicit-solver-strategy.md §7`, which is itself a
ladder of physics:

1. **Scalar elliptic GPU solve** — *done* (Chapters 5, 7).
2. **Unsteady Stokes** — *done* (Chapter 6).
3. **Incompressible Navier–Stokes** — *done*, low-Re with explicit convection (Ch. 6).
4. **Viscoelastic** — *done*, Oldroyd-B + log-conformation, 2D and 3D (Chapter 8).
5. **(Upgrade) H(div)-HDG** for exact pressure-robustness — *not started*; only pursue
   if the divergence diagnostics from steps 2–3 ever justify the extra complexity.

On top of that ladder, immersed boundaries (Chapter 9) and 2D adaptivity (Chapter 10)
are built and GPU-resident, and the whole stack exists in both 2D and 3D. The headline
capstone — an immersed particle in viscoelastic flow on the GPU — runs.

So the *core* is in place. What remains is breadth, performance, accessibility, and a few
genuinely hard pieces.

## Parked and planned work

### 3D non-conforming adaptivity *(substantial port)*
2D adaptivity is complete (Chapter 10), and the GPU flow solver runs on 2:1
non-conforming meshes. **3D** non-conforming adaptivity is not built: it needs `Mesh3d`
octree non-conforming connectivity (currently 3D meshes are conforming only) and the 3D
hex mortar operators, *then* a GPU 3D non-conforming operator. This is the most direct
extension of finished work — the 2D path is the template — but it is a substantial port.

### p- and hp-adaptivity *(substantial port)*
gale has `h`-adaptivity (subdivide elements) but a single, uniform polynomial degree per
mesh. **p-adaptivity** (per-element degree) and **hp-adaptivity** (both) are not built.
This is the one axis where the reference literature (the Nayak–Mavriplis interior-penalty
hp-IBM paper) is ahead of gale, and it matters for exactly gale's problems: it is the
clean way to recover the accuracy that volume-penalized immersed boundaries (Chapter 9)
and steep viscoelastic stress layers (Chapter 8) lose, by raising the order *only* where
it is needed. The design — generalize the mortar to a rectangular degree-projection
$ P(p_\text{from} \to p_\text{to}) $, carry per-element degree in the mesh, and add a
"by-order batched" GPU launch — is written up in `docs/hp-adaptivity-and-ibm-gaps.md`.
The 2D non-conforming GPU mortar plumbing built in Chapter 10 is the prerequisite, and it
now exists, so this is unblocked.

### Toward suspensions: deformable particles and two-phase flow *(open problem)*
Today's immersed boundaries are *rigid* (or prescribed-motion) bodies. The headline
target — particle-laden suspensions — ultimately wants:

- **two-way coupled rigid particles** (the fluid moves the particle, the particle moves
  the fluid), then **many** of them, with adaptivity tracking each;
- **deformable particles / capsules** — an elastic membrane immersed in the fluid, a
  fluid–structure interaction problem;
- and, distinct from all of the above, **true two-phase flow** — two fluids with an
  evolving interface (e.g. a viscoelastic drop in a Newtonian matrix), which is *not* an
  immersed-boundary problem but an interface-capturing one.

`docs/mesh-and-adaptivity-strategy.md §8–9` lays out the staged path and the method
choices. These are the largest remaining physics extensions, and the reason the
discretization was built to be high-order and adaptive from the start.

### Constitutive models beyond Oldroyd-B *(contained)*
Oldroyd-B allows *infinite* polymer extension, which is unphysical and a source of
high-Wi pathology. **FENE-P** (finitely-extensible nonlinear elastic, Peterlin closure)
bounds the stretch and is the natural next model. The log-conformation machinery of
Chapter 8 carries over directly; this is a contained addition.

### Higher-order time integration *(contained)*
The dual-splitting scheme is currently **BDF1** (first-order in time). Higher-order BDF /
stiffly-stable schemes, and possibly IMEX treatments of the coupling, are a known upgrade
(Chapter 6). Worth doing once spatial accuracy and stability are no longer the binding
constraint.

### Temporal adaptivity: local time-stepping and local implicitness *(researched, parked)*
gale takes one global time step everywhere. A refined small cell, or a stiff
high-stress region, can force that global step to be tiny — the temporal analogue of the
spatial waste that AMR exists to remove. The fix is **temporal adaptivity**, and a
dedicated, adversarially-verified research pass (`docs/implicit-solver-strategy.md §3b`)
mapped the design space and reached a deliberately cautious conclusion: it is *designed*
but not built, and the order in which to build it is not the obvious one.

The pass turned on a single distinction — *why* is a cell's step small?

- **A small cell with a fast local speed is CFL-limited** → the cure is **local
  time-stepping (LTS)**: let different cells subcycle at different rates.
- **A high-stress, slowly-relaxing cell is stiffness-limited** (relaxation $ \sim 1/\mathrm{Wi} $,
  the high-Weissenberg problem of Chapter 8) → the cure is **local implicitness / IMEX**:
  treat *only* the stiff cells implicitly, the rest explicitly.

These are different mechanisms with different cures, and the order matters for gale
specifically, for two architectural reasons. First, **our spine is implicit**: the
incompressible solve is a dual-splitting projection with a *global* pressure-Poisson
every step (Chapters 6–7), which couples the whole domain regardless of any local time
step. So LTS can only ever accelerate the **explicit substeps — advection and
conformation-stress transport** — never the flow solve, and its payoff is Amdahl-bounded
by that global solve. Second, **gale's hard cells are stiff, not CFL-limited**:
high-Weissenberg stress layers are stiffness-plus-under-resolution. The prescribed cure
for them is therefore **spatial AMR + local implicit/IMEX, not smaller explicit steps** —
which makes locally-implicit/IMEX treatment of the stiff coupling the genuinely valuable
near-term temporal upgrade, ranking *above* LTS proper.

If LTS is ever built, the research pins down the only viable form: **cluster-based,
power-of-2, level-batched** subcycling (the SeisSol design — per-element *asynchronous*
LTS is GPU-hostile), with **2:1 temporal grading** between neighbors (the time analogue
of the 2:1 spatial balance of Chapter 10, so time-refinement tracks $h$-refinement). One
sharp trap is recorded: for a projection/dual-splitting solver, classical
**Berger–Colella refluxing is the *wrong* correctness mechanism** at coarse↔fine time
interfaces — the conservation fix that works for explicit hyperbolic AMR does not carry
over. Two genuine gaps remain open even in the literature: the multirate
method-of-lines order conditions for a fast-explicit-transport / slow-implicit-pressure
split, and the LTS × immersed-boundary interaction.

The one concrete thing to preserve now, ahead of any implementation, is a **per-cell
time-level** in the time-integration abstraction, so this is addable without a rewrite.

### Performance *(open problem)*
gale has been built **correctness-first**: every kernel is validated bit-for-bit against
the CPU oracle (Chapter 12), but it has not had a systematic performance pass. A roofline
study of the elliptic solver (the flagged bottleneck), occupancy and shared-memory
tuning, and multi-GPU scaling beyond two devices are all open. The architecture
(Chapter 11) was designed to make this tractable, but the work is not done.

### Load balancing for adaptive multi-GPU *(open problem)*
The current domain decomposition is block-based. Adaptive meshes that refine and coarsen
during a run need **space-filling-curve (Hilbert/Morton) load balancing** to stay
balanced across GPUs — noted in the code as a drop-in replacement, not yet implemented.

### A Python entry point *(contained)*
gale is Rust-first, but the long-term intent (per `CLAUDE.md`) is a **pyo3** interface so
that simulations can be assembled and driven from Python, bridging to the wider analysis
ecosystem. Not started.

## The cuda-oxide relationship

gale is built on the `cuda-oxide` toolchain (Chapter 11), which is young. Rather than
work around its gaps, gale **fixes them upstream**: the typed-pointer NVVM-IR support for
pre-Blackwell GPUs, the cross-crate device-artifact linking that makes `gale-gpu` a
reusable library, the libdevice mapping for `atan2`/`sinh`/`cbrt`/… and the NaN-constant
codegen that unblocks `f64::signum` — all originated as gale needs and live in the fork.
This is a deliberate posture: the toolchain is part of the project, and improving it is
in scope (see `docs/cuda-oxide-repo-status.md`).

## How to think about the gaps

None of the above changes what you have learned in this book — the theory is the same
whether a feature ships today or next quarter. The methods gale has chosen (high-order
DG-SEM, projection methods, interior penalty, log-conformation, volume penalization,
mortar adaptivity) were chosen precisely *because* they extend cleanly to the harder
problems above. The roadmap is not a list of regrets; it is the payoff structure that
the architecture was designed to unlock.
