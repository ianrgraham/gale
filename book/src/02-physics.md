# The Physics: Flow, Viscoelasticity, and Suspensions

Before we discretize anything, we have to be clear about *what* we are discretizing.
This chapter sets up the equations gale solves and — more importantly — the physical
*regime* it lives in. That regime is the lens for every method choice in the rest of the
book: when a later chapter says "we do it this way," the honest answer is almost always
"because of low-Reynolds, high-Weissenberg, particle-laden flow." So let us earn that
phrase.

We build up in layers: the bedrock conservation laws, the incompressible Navier–Stokes
equations, the dimensionless numbers that pin down our regime, what "viscoelastic" adds,
and finally the objects we want to put *inside* the fluid — particles, and the line
between "immersed solid" and "true two-phase flow" that organizes the whole project.

## Conservation of mass and momentum

Fluid dynamics is bookkeeping. Two quantities are conserved — mass and momentum — and
the governing equations are just the statement that they cannot appear or vanish, only
move around and be pushed.

**Mass.** For a fluid of density \\( \rho \\) moving with velocity \\( \mathbf{u} \\), the
mass in any fixed region changes only by flux through its boundary:

\\[
\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\,\mathbf{u}) = 0.
\\]

**Momentum.** Newton's second law, written per unit volume: the rate of change of
momentum equals the forces acting. The forces are the *surface stresses* the surrounding
fluid exerts (gathered into a stress tensor \\( \boldsymbol{\sigma} \\)) plus any body
force \\( \mathbf{f} \\) (gravity, an immersed-boundary penalty force, a polymer-stress
divergence):

\\[
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
= \nabla\cdot\boldsymbol{\sigma} + \mathbf{f}.
\\]

The term \\( (\mathbf{u}\cdot\nabla)\mathbf{u} \\) is *advection* — the fluid carrying its
own momentum along with it. It is nonlinear, and it is the source of most of the
interesting (and most of the unstable) behavior in high-Reynolds flow. Keep an eye on
it; later chapters spend real effort taming it.

## The stress tensor, and what makes a fluid "Newtonian"

Everything physical about *which* fluid we are simulating is hidden in the stress tensor
\\( \boldsymbol{\sigma} \\). Splitting it into an isotropic pressure part and a
deviatoric (shape-changing) part:

\\[
\boldsymbol{\sigma} = -p\,\mathbf{I} + \boldsymbol{\tau}.
\\]

The \\( -p\,\mathbf{I} \\) term is pressure: it pushes equally in all directions and only
resists *volume* change. The deviatoric part \\( \boldsymbol{\tau} \\) is where the
material's personality lives. For a **Newtonian** fluid — water, air, glycerin — stress
is simply proportional to the local rate of strain:

\\[
\boldsymbol{\tau} = \mu\left(\nabla\mathbf{u} + \nabla\mathbf{u}^{\mathsf T}\right),
\\]

where \\( \mu \\) is the (dynamic) viscosity. The combination
\\( \nabla\mathbf{u} + \nabla\mathbf{u}^{\mathsf T} \\) is twice the symmetric
rate-of-strain tensor \\( \mathbf{D} \\); using the symmetric (physical) gradient here,
rather than just \\( \nabla\mathbf{u} \\), turns out to matter for a subtle stability
property called *pressure-robustness* that Chapter 6 returns to. The key feature of a
Newtonian fluid is that it has **no memory**: the stress *right now* depends only on the
deformation rate *right now*. Stop shearing it and the stress vanishes instantly. Much
of this chapter is about what happens when that stops being true.

## The incompressible Navier–Stokes equations

In the microfluidic world gale targets, flow speeds are far below the speed of sound, so
density variations are negligible: the fluid is **incompressible**. Mass conservation
then collapses to the statement that the velocity field is divergence-free, and with
constant \\( \rho \\) and \\( \mu \\) the momentum equation becomes the incompressible
Navier–Stokes equations:

\\[
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
= -\nabla p + \mu\,\nabla^2\mathbf{u} + \mathbf{f},
\qquad
\nabla\cdot\mathbf{u} = 0.
\\]

This is the workhorse system. The first equation evolves momentum; the
\\( \mu\,\nabla^2\mathbf{u} \\) term is viscous diffusion (it smooths velocity and is the
stiff, elliptic part that wants implicit treatment); \\( \mathbf{f} \\) is where polymers
and immersed bodies will enter as body forces.

### Incompressibility is a constraint, not an evolution equation

Notice what the second equation is *not*. There is no \\( \partial p/\partial t \\)
anywhere. Pressure has no evolution equation of its own. This is the single most
important structural fact about incompressible flow, and it shapes the entire solver.

The right way to read \\( \nabla\cdot\mathbf{u}=0 \\) is as a **constraint** the velocity
must satisfy at every instant — like the rigidity constraint on a pendulum's arm. And
just as a constraint force keeps the pendulum's length fixed, **pressure is the
Lagrange multiplier that enforces incompressibility.** At each moment, pressure
instantaneously adjusts itself to exactly whatever field is needed to project the
momentum forward into the space of divergence-free velocities — no more, no less. It
carries no dynamics; it is the bookkeeper that keeps the constraint satisfied.

This is *why* solving incompressible flow is hard and why so much of the machinery in
Part III exists. We cannot just march pressure forward in time. We have to solve, every
step, an elliptic equation (a Poisson problem) for the pressure that makes the velocity
divergence-free. That elliptic solve is the computational bottleneck of the whole
enterprise, and Chapter 6 is devoted to the projection / dual-splitting trick that turns
the coupled velocity–pressure problem into a sequence of solvable pieces.

## The dimensionless numbers that define gale's regime

A simulation does not really care about the dimensional values of \\( \rho \\), \\( \mu \\),
or the channel width — it cares about their *ratios*. Two flows with the same
dimensionless numbers behave the same. A handful of these numbers pin down gale's
regime.

**Reynolds number** — inertia versus viscosity:

\\[
Re = \frac{UL}{\nu} = \frac{\rho\,U L}{\mu},
\\]

with \\( U \\) a characteristic speed, \\( L \\) a length, and \\( \nu = \mu/\rho \\) the
kinematic viscosity. High \\( Re \\) means inertia dominates: turbulence, sharp wakes, the
nonlinear advection term running the show. Low \\( Re \\) means viscosity dominates: flow
is smooth, slow, reversible, "creeping." **Microfluidics is firmly low-\\( Re \\)** —
small \\( L \\), modest \\( U \\) — often \\( Re \ll 1 \\). The advection term is *weak*, which
sounds like a simplification (and for stability of advection, it is). The price is that
the stiff viscous and pressure parts now completely dominate, so the implicit elliptic
solve is everything.

For viscoelastic fluids we need two more numbers, because the fluid now has an internal
clock — a relaxation time \\( \lambda \\), the time the polymer takes to "forget" a
deformation.

**Weissenberg number** — elastic stretching versus relaxation:

\\[
Wi = \lambda\,\dot\gamma,
\\]

where \\( \dot\gamma \\) is a characteristic shear (strain) rate. \\( Wi \\) measures how
hard the flow is stretching the polymers relative to how fast they relax back. Low
\\( Wi \\): polymers relax faster than the flow deforms them, elasticity is a mild
perturbation. High \\( Wi \\): the flow stretches polymers faster than they recover, and
elastic stresses build up enormously. This is where viscoelastic flows do their
spectacular, counterintuitive things.

**Deborah number** — elastic memory versus the flow's own timescale:

\\[
De = \frac{\lambda}{t_{\text{flow}}},
\\]

the ratio of the relaxation time to a characteristic time of the flow itself. The two
numbers are closely related — \\( Wi \\) leans on a *rate*, \\( De \\) on a *timescale* — and
in steady shear they often coincide; the name to remember is that *both* measure how
strongly the fluid's memory matters.

### Why this particular corner is where the interesting physics live

Here is the crucial combination. Microfluidics gives us **low \\( Re \\)** (inertia is
weak) — but nothing stops \\( Wi \\) from being **large**, because \\( Wi \\) depends on the
polymer relaxation time, not on inertia. You can have a slow, creeping, perfectly
laminar-looking flow that is nonetheless violently elastic.

That regime — low \\( Re \\), high \\( Wi \\) — is exactly where the famous *purely elastic
instabilities* and *elastic turbulence* appear: chaotic, mixing flows driven entirely by
polymer stress, with no inertia in sight. It is genuinely useful (mixing at small scales
is otherwise very hard) and genuinely hard to simulate. The numerical difficulty even
has a name, the **High-Weissenberg-Number Problem**, and defeating it is one of the
recurring stability stories of this book (Chapter 8). gale aims squarely at this corner;
that is the whole point.

## What "viscoelastic" adds, physically

A Newtonian fluid has no memory. A **viscoelastic** fluid does. The canonical example,
and gale's target, is a *dilute polymer solution*: a Newtonian solvent (water) carrying a
small concentration of long, flexible polymer chains.

In a quiescent fluid those chains are coiled up in a relaxed, high-entropy blob. When the
flow shears or stretches the fluid, it stretches the chains, and like tiny springs they
store elastic energy and pull back. Crucially, this pull-back is not instantaneous: the
chains relax over the timescale \\( \lambda \\). So the stress in the fluid *now* depends
on the *history* of deformation — the fluid remembers. That memory is what makes
"viscoelastic" simultaneously viscous (it dissipates, like a liquid) and elastic (it
springs back, like a solid).

Mechanically, the polymers contribute an extra stress \\( \boldsymbol{\tau}_p \\) that
*evolves in time* and gets added to the momentum balance:

\\[
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
= -\nabla p + \eta_s\,\nabla^2\mathbf{u} + \nabla\cdot\boldsymbol{\tau}_p + \mathbf{f},
\qquad \nabla\cdot\mathbf{u}=0,
\\]

where \\( \eta_s \\) is the *solvent* viscosity and the polymer feeds back into the flow
only through the divergence of its stress, \\( \nabla\cdot\boldsymbol{\tau}_p \\). The
total zero-shear viscosity is \\( \eta_0 = \eta_s + \eta_p \\), with \\( \eta_p \\) the
polymer contribution.

The open question is: what determines \\( \boldsymbol{\tau}_p \\)? We need a *constitutive
model* — an evolution law for the polymer stress. The standard, physically grounded way
to track it is not the stress tensor directly but a **conformation tensor**
\\( \mathbf{C} \\): roughly, a statistical measure of how stretched and oriented the polymer
chains are (its eigenvalues are stretch-squared along each principal direction, so for an
unstretched fluid \\( \mathbf{C} = \mathbf{I} \\)). The conformation tensor obeys an
advection-stretch-relaxation equation, and the stress is read off from it, e.g. for the
Oldroyd-B model \\( \boldsymbol{\tau}_p = (\eta_p/\lambda)(\mathbf{C}-\mathbf{I}) \\). We
develop this fully in Chapter 8 — including *why* \\( \mathbf{C} \\) must stay
symmetric-positive-definite, why the obvious discretization fails to keep it so at high
\\( Wi \\), and the log-conformation cure. For now, the high-level idea is enough: **a
fluid with memory, whose memory is carried by an extra evolving tensor field.**

## Particle-laden suspensions: objects inside the fluid

gale's headline target is not bare polymer solution — it is a *suspension*: solid (or
eventually deformable) particles carried by the fluid. Think of cells flowing through a
microchannel, or rigid beads in a polymer solution.

The naive way to handle a particle is to mesh around its surface — to build the
computational grid so that element faces line up with the particle boundary
("body-fitted"). For a *fixed* shape this is the accuracy gold standard, and gale
supports it. But for *moving* particles it is a nightmare: every time the particle moves
you must deform or rebuild the mesh, which is slow, fragile, destroys conservation when
you transfer the solution between meshes, and — fatally for a GPU code — wrecks the fixed
partitioning that makes multi-GPU execution efficient. And it simply *cannot* represent
topology changes: particles touching, large rotations, many bodies.

So gale represents particles **without meshing their surfaces**, using the **immersed
boundary** idea: keep a fixed background mesh, and let the particle live *inside* it as a
field — a mask marking which points are "solid," plus a forcing term that makes the fluid
there move with the body. The mesh never changes; the body is just data on it. This is
Chapter 9, and the specific flavor gale implements is *volume penalization* (a Brinkman
penalty force that drives the velocity toward the body's velocity inside the solid
region). It keeps the discretization element-local — exactly the property the GPU
rewards — and topology changes become free.

The catch, and gale is honest about it (see Chapter 13): immersing a body *smears* its
interface across a few background cells, which degrades the near-wall stress — precisely
where viscoelastic stress is largest and most delicate. This is a real tension for moving
viscoelastic walls, and the partial cure is to refine the mesh in a band around each
body. Which brings us to the last piece.

**Many particles demand adaptivity.** A suspension is not one particle; it is many, each
needing fine resolution near its surface and in its wake, while the bulk of the domain
can stay coarse. Resolving everything everywhere is hopelessly expensive. The answer is
**adaptive mesh refinement** (AMR): let the grid refine itself where the solution is
under-resolved and coarsen where it is smooth. DG turns out to be unusually friendly to
AMR — because elements already only talk through their faces, a hanging-node interface is
handled by the same flux machinery as any other face. That is Chapter 10.

## Two-phase flow versus immersed solids — an important distinction

There is a conceptual fork here that is easy to get wrong, and gale's design (following
the mesh-and-adaptivity strategy) is built around getting it right. The question is:
**what kind of material is the immersed thing?**

The litmus test is simple: *does the material elastically return to a reference shape?*

- **Yes — it is a SOLID.** A rigid bead, an elastic capsule, a red blood cell membrane.
  It stores elastic energy relative to a rest configuration, so the natural description
  tracks its *material deformation* (a Lagrangian view), and the natural coupling is an
  **immersed-solid method** — for rigid/forced bodies, the volume-penalization IBM gale
  uses now.

- **No — it is a FLUID.** A droplet of one liquid suspended in another. It has no rest
  shape; it flows indefinitely and its memory fully relaxes. What you track is the
  *interface* between the two fluids (an Eulerian view), and this is a genuinely
  different problem called **two-phase flow**.

These are *not* interchangeable methods for the same problem; they are the right tools
for *different* materials. You should not push a fluid droplet through an immersed-solid
method, and you should not track a flowing material as if it had a reference
configuration. (Real soft matter — a capsule, which is an elastic *membrane* enclosing a
distinct *interior fluid* — straddles both families, which is exactly what makes the full
suspension problem hard.)

**Where gale stands today, honestly.** gale targets the **immersed-solid (IBM) path**:
rigid bodies via volume penalization, validated in 2D and 3D, including the capstone of a
rigid particle in viscoelastic flow on the GPU. **True two-phase flow and deformable
particles are a planned future direction, not built.** The intended route for two-phase
is a *phase-field* (Cahn–Hilliard diffuse-interface) model, which pairs naturally with
high-order DG — it is a smooth PDE, needs no sharp interface reconstruction, handles
droplets merging and breaking automatically, and is conservative. Deformable particles
point toward immersed-finite-element / Peskin-style coupling. These are coherent with the
rest of the architecture and are sketched in Chapter 10 and tracked in Chapter 13, but
the book is careful to label them as the road ahead, not the road travelled.

## The throughline

Hold onto one picture from this chapter. gale's world is **low Reynolds number** (so the
advection term is mild but the implicit pressure/viscous solve dominates everything),
**potentially high Weissenberg number** (so elastic polymer stress is strong, evolving,
and numerically dangerous — the High-Weissenberg-Number Problem), and **particle-laden**
(so we need immersed boundaries and adaptive refinement, not body-fitted remeshing).
Every method in the chapters that follow — the discontinuous Galerkin discretization, the
numerical fluxes, the interior-penalty Laplacian, the projection solver, the
log-conformation representation, volume penalization, conservative mortars — is a response
to some part of that sentence, and very often a response to the same underlying worry:
*what could blow up here, and what are we doing about it?*

With the physics fixed, the next part of the book turns to *how we represent fields and
operators in space at all* — and the answer, and the reasons for it, is the discontinuous
Galerkin spectral element method of Chapter 3.
