# Incompressible Navier–Stokes

By this point in the book we can discretize the two halves of a fluid equation
separately: Chapter 4 built the *hyperbolic* operators (advection, the transport of
stuff by the flow) and Chapter 5 built the *elliptic* operators (diffusion, the
interior-penalty Laplacian). The incompressible Navier–Stokes equations need both at
once — and they add a third ingredient that is neither hyperbolic nor elliptic, and
that turns out to be the whole difficulty: the **incompressibility constraint**.

This chapter is about how gale gets from

\\[
  \partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}
  = -\nabla p + \nu\,\nabla^2 \mathbf{u} + \mathbf{f},
  \qquad
  \nabla\cdot\mathbf{u} = 0
\\]

to code you can step in time. The short version: we *refuse* to solve the coupled system
directly, and instead split each timestep into a sequence of the SPD elliptic solves we
already know how to do well on the GPU. Understanding *why* that split is the right move
— and where it is paid for — is the point of the chapter.

## The challenge: incompressibility is a constraint, not an equation

Look at the two equations above and count what they are for. The momentum equation is an
evolution equation for \\( \mathbf{u} \\): it tells you \\( \partial_t \mathbf{u} \\), so in
principle you could march velocity forward in time. But there is no evolution equation
for pressure \\( p \\) anywhere. Pressure has no \\( \partial_t p \\) term. It is not a thing
that *evolves*; it is a thing that is *whatever it has to be* so that the second equation,
\\( \nabla\cdot\mathbf{u}=0 \\), stays true.

That is the conceptual heart of incompressible flow. The constraint
\\( \nabla\cdot\mathbf{u}=0 \\) says the velocity field must be divergence-free at every
instant — fluid is neither created nor destroyed in any cell. Pressure is the
**Lagrange multiplier** that enforces it. Think of the classic analogy from mechanics: a
bead constrained to a wire. The constraint (stay on the wire) is maintained by a
constraint force (the normal force from the wire), and that force is *not* something you
prescribe — it is exactly the force needed to keep the bead on the wire, no more, no
less. Pressure is the fluid's normal force, and \\( -\nabla p \\) is how it pushes back the
instant the flow tries to compress.

This has a sharp consequence for time-stepping. If you tried to write the system as one
big linear solve for \\( (\mathbf{u}, p) \\) together, you would get a matrix with the
block structure

\\[
  \begin{pmatrix} A & B^\top \\ B & 0 \end{pmatrix}
  \begin{pmatrix} \mathbf{u} \\ p \end{pmatrix}
  = \begin{pmatrix} \mathbf{g} \\ 0 \end{pmatrix},
\\]

where \\( A \\) is the velocity (viscous/mass) block and \\( B \\) is the divergence. The
zero block on the diagonal — there is no pressure-pressure coupling — makes this a
**saddle-point problem**: indefinite, not positive-definite, and far harder to solve than
an ordinary elliptic system. Indefinite means eigenvalues of both signs, so conjugate
gradient — which assumes a positive-definite energy to descend — simply does not apply;
the whole zoo of saddle-point solvers (Uzawa, MINRES, block preconditioners, Schur
complements) exists precisely because of that zero block. You cannot just time-march
velocity and read off pressure;
velocity and pressure are knotted together at every step. Worse, a naïve discretization
of a saddle-point system is *unstable* unless the velocity and pressure spaces satisfy a
compatibility condition (the **LBB / inf-sup** condition), which rules out the convenient
choice of using the same polynomial space for both. The constraint is the villain of this
chapter, and everything that follows is about defeating it without ever assembling that
indefinite matrix.

## The projection idea: split the timestep

The escape route is **operator splitting in time**, in the family of *projection* or
*fractional-step* methods that goes back to Chorin and, in the high-order form gale uses,
to Karniadakis, Israeli & Orszag's **stiffly-stable "dual-splitting"** scheme. The idea is
to handle the three physical effects — forcing/advection, the incompressibility
constraint, and viscous diffusion — *one at a time* within each step, rather than all at
once.

Concretely, to advance from \\( \mathbf{u}^n \\) to \\( \mathbf{u}^{n+1} \\) over a step
\\( \Delta t \\), gale does three stages.

**Stage 1 — explicit predictor.** Take the terms that are cheap and non-stiff explicitly.
For pure Stokes flow that is just the forcing; for Navier–Stokes it also includes the
nonlinear advection:

\\[
  \hat{\mathbf{u}} = \mathbf{u}^n + \Delta t\,\big(\mathbf{f} - (\mathbf{u}^n\cdot\nabla)\mathbf{u}^n\big).
\\]

This \\( \hat{\mathbf{u}} \\) is a provisional velocity. It is *wrong* in two ways: it does
not satisfy the incompressibility constraint, and it has not yet felt the viscous
diffusion of this step. The next two stages fix those, in order.

**Stage 2 — pressure projection.** This is where the constraint gets enforced. We want to
remove from \\( \hat{\mathbf{u}} \\) exactly the part that violates
\\( \nabla\cdot\mathbf{u}=0 \\). The tool is the **Helmholtz–Hodge decomposition**: any
vector field splits uniquely into a divergence-free part plus a gradient,
\\( \hat{\mathbf{u}} = \hat{\hat{\mathbf{u}}} + \Delta t\,\nabla p \\), with
\\( \nabla\cdot\hat{\hat{\mathbf{u}}} = 0 \\). Taking the divergence of that relation and
using divergence-free-ness gives a **pressure-Poisson equation**:

\\[
  \nabla^2 p = \frac{1}{\Delta t}\,\nabla\cdot\hat{\mathbf{u}},
\\]

written here with \\( +\nabla^2 \\); the code negates both sides and solves the
positive-definite \\( -\nabla^2 \\), which is what makes the implemented operator SPD (and
the right sign for CG) downstream. We then **project** the predictor onto the
divergence-free space by subtracting the gradient:

\\[
  \hat{\hat{\mathbf{u}}} = \hat{\mathbf{u}} - \Delta t\,\nabla p.
\\]

This is exactly the "subtract off the part that compresses" step. Geometrically it is an
orthogonal projection onto the space of divergence-free fields; pressure is the potential
whose gradient is the discarded piece. The split is orthogonal in the \\( L^2 \\) inner
product — a divergence-free field and a gradient are \\( L^2 \\)-orthogonal (integrate by
parts and the divergence-free part kills the coupling) — so the projection returns the
*closest* divergence-free field to the predictor: it removes the constraint violation and
nothing else, which is the formal reason it does not contaminate the velocity we cared
about, and the same orthogonality that makes the operator SPD downstream.

**Stage 3 — implicit viscous solve.** Finally apply the viscous diffusion, *implicitly*
(we will see why in a moment). Discretizing \\( \partial_t\mathbf{u} = \nu\nabla^2\mathbf{u} \\)
with backward Euler (BDF1) and using \\( \hat{\hat{\mathbf{u}}} \\) as the data gives, for
each velocity component, a **Helmholtz** problem:

\\[
  \Big(\tfrac{1}{\nu\,\Delta t}\,M + A\Big)\,\mathbf{u}^{n+1}
  = \tfrac{1}{\nu\,\Delta t}\,M\,\hat{\hat{\mathbf{u}}},
\\]

where \\( M \\) is the mass matrix and \\( A \\) is the SIPG stiffness (the
interior-penalty Laplacian, sign-flipped so it is positive). Writing
\\( \lambda = 1/(\nu\Delta t) \\), the operator is \\( \lambda M + A \\) — the
**Helmholtz operator** from Chapter 5, the Laplacian plus a positive reaction term.
Solving it for both components gives the new, divergence-free, properly-diffused velocity.

That is the whole scheme. One hard coupled problem has become three easy stages: an
explicit update (free), a Poisson solve, and a Helmholtz solve.

## Why split: turning a saddle point into SPD elliptic solves

Here is the central design payoff, and it is worth stating plainly because it justifies an
enormous amount of the rest of the codebase.

The coupled system was an indefinite saddle-point problem. After splitting, the *only*
linear systems we ever solve are the **pressure-Poisson** \\( \nabla^2 p = \dots \\) and the
**viscous-Helmholtz** \\( (\lambda M + A)\mathbf{u} = \dots \\). Both of these are
**symmetric positive-definite (SPD)** — they are precisely the operators Chapter 5 built
the SIPG discretization for, and SPD is precisely the property Chapter 7's
**conjugate-gradient** solver needs. The diagonal GLL mass matrix (Chapter 5) makes the
reaction term \\( \lambda M \\) trivially cheap. So the splitting converts the one part of
incompressible flow that does *not* fit our GPU toolbox into two parts that fit it
perfectly.

There is a second, structural payoff. By decoupling velocity and pressure in time, the
dual-splitting scheme **sidesteps the coupled LBB/inf-sup constraint** — not by making the
condition vanish, but by replacing it with a well-posed SIPG pressure-Poisson solve. We
never assemble the coupled saddle-point matrix, so we are never at the mercy of *its*
stability condition; what we owe instead is that the discrete pressure Laplacian be
non-singular (modulo the constant nullspace, handled by deflation) and free of spurious
pressure modes. For DG that is not automatic, and the reason gale is safe is that it builds
the pressure operator as a genuine SIPG Laplacian (Chapter 5) rather than as
\\( B M^{-1} B^\top \\) from the equal-order operators — and *that* is the design choice that
earns the convenience of **equal-order interpolation**, the same polynomial degree for
velocity and pressure, which makes a clean nodal DG-SEM implementation possible. The
code-level prize is concrete: same nodes, same operators, same kernels for \\( u \\),
\\( v \\), \\( w \\), *and* \\( p \\) — one apply, one mass matrix, one p-multigrid setup
reused across all four solves (Chapter 7).
This is exactly the route the verified viscoelastic-DG literature takes (the SRCR-DG
solver builds on the same Karniadakis–Israeli–Orszag splitting), and it is the
"recommended spine" of gale's solver-strategy document.

## Why the viscous step must be implicit

Stage 3 solves a linear system every step, which is more expensive than an explicit
update. Why pay for it? Because **diffusion is stiff**, and an explicit viscous update has
a punishing stability limit.

If you treated \\( \nu\nabla^2\mathbf{u} \\) explicitly, von Neumann stability analysis
gives a timestep restriction of the form

\\[
  \Delta t \;\lesssim\; \frac{h^2}{\nu},
\\]

and on a high-order spectral element this is far worse than it looks. The spectral radius
of the second-derivative (Laplacian) operator on a GLL grid grows like
\\( \rho \sim p^4/h^2 \\) — steeper than a naive nodal CFL estimate, because it is set by the
*operator norm*, not by the minimum node spacing. The true explicit diffusive limit is
therefore \\( \Delta t \lesssim h^2/(\nu\,p^4) \\) — quadratic in element size, *quartic* in
polynomial degree. For gale's target regime that is a disaster: microfluidics means small
elements and small length scales, and high-order means large \\( p \\). Every one of those
pushes the explicit viscous timestep toward zero. You would spend thousands of steps
resolving viscous decay you do not care about.

Treating the viscous term implicitly removes the limit completely: backward Euler is
*unconditionally* stable for diffusion, so \\( \Delta t \\) is set by accuracy and by the
explicit terms, not by stiffness. This is the recurring stability lesson of the book in
its sharpest form — **identify the stiff term and make it implicit.** Here the stiff term
is diffusion, and the price of taming it is exactly the Helmholtz solve of Stage 3, which
(no coincidence) is SPD and GPU-friendly.

## Why convection can stay explicit

The nonlinear advection \\( (\mathbf{u}\cdot\nabla)\mathbf{u} \\) is the term that makes a
fully-implicit Navier–Stokes step nonlinear and miserable. gale keeps it **explicit**, and
in its target regime that is justified. At **low Reynolds number** — gale's microfluidic
home turf — inertia is weak and advection is *not* the stiff part of the problem. The
advective stability limit is a CFL condition \\( \Delta t \lesssim h/|\mathbf{u}| \\), which
at low speeds is mild and easily satisfied; the stiffness lives in the viscous and
(later) the polymer-relaxation terms, which we make implicit. Treating advection
explicitly keeps each step a *linear* solve and avoids any per-step nonlinear iteration —
exactly the "advance momentum → pressure → done" decoupling the dual-split scheme is
designed for.

gale offers two ways to discretize the convection term, mirroring Chapter 4's two volume
forms:

- **Nodal (collocation) form** — `ConvectionScheme::Nodal` — evaluates
  \\( (\mathbf{u}\cdot\nabla)\mathbf{u} \\) pointwise at the GLL nodes, element by element.
  It is cheap, simple, and exact on polynomials; it is the right default for smooth,
  low-Re flow.
- **Split-form DG** — `ConvectionScheme::SplitFormDg` — uses the kinetic-energy-preserving
  split form with a Rusanov interface flux (Chapter 4). This is the energy-stable path you
  reach for when the flow is under-resolved or the effective Reynolds number is higher,
  where the nodal form would aliasing-drive an instability.

Both are validated, on the same test, to reproduce the exact Taylor–Green vortex.

The cost of all this convenience is **splitting error** and **temporal order**. Doing the
physics one stage at a time, with BDF1 backward Euler and explicit extrapolation of the
non-stiff terms, makes the scheme **first-order accurate in time**. That is fine for
reaching a steady state or for the validation regimes gale targets now; when transient
accuracy matters you would extend the same structure to a higher-order stiffly-stable
variant. There are really *two* first-order errors stacked here: the BDF1 time-integration
error, \\( O(\Delta t) \\), which you would have even with no splitting; and the *splitting*
(commutator) error from solving the constraint and the viscous step in sequence rather than
together — also \\( O(\Delta t) \\), and the one that interacts with the pressure boundary
condition below. Going to BDF2/BDF3 with matching extrapolation order is a coefficient
change to the predictor and the Helmholtz reaction and cleanly fixes the time integrator;
but lifting the *splitting* error to high order also requires the consistent high-order
pressure Neumann condition (without it these schemes develop the well-known
\\( O(\sqrt{\nu\Delta t}) \\) numerical boundary layer in pressure near walls). The build
keeps BDF1; the higher orders are a documented, planned upgrade.

## The boundary-condition subtlety

Split schemes have a notorious wrinkle at the boundaries, and it is worth flagging because
it is a classic accuracy sink. The original momentum equation has boundary conditions for
*velocity* — typically Dirichlet, the no-slip wall. But Stage 2's pressure-Poisson is an
equation for *pressure*, which had no boundary condition of its own. The splitting forces
one on it: consistency with the projection requires a **Neumann** condition on pressure
(its normal derivative is tied to the momentum balance at the wall). The *consistent*
condition is inhomogeneous — projecting the momentum equation onto the wall normal gives
\\( \partial_n p = \mathbf{n}\cdot(\nu\nabla^2\mathbf{u} - (\mathbf{u}\cdot\nabla)\mathbf{u} +
\mathbf{f} - \partial_t\mathbf{u}) \\), generally nonzero, and getting this term right (in
its rotational form) is what restores high-order accuracy near walls. gale's BDF1 build
instead uses the cheap **homogeneous-Neumann** simplification on all boundary faces; the
consistent inhomogeneous high-order pressure BC is part of the same planned BDF2/BDF3
upgrade, and it is exactly this approximation that (together with BDF1) caps the current
scheme at first order.

That creates a second, subtler problem. With velocity Dirichlet everywhere and pressure
Neumann everywhere, the pressure is only determined **up to an additive constant** — add
any constant to \\( p \\) and \\( \nabla p \\) is unchanged, so the projection is unaffected.
The pressure-Poisson operator is therefore **singular**: it has the constant vector in its
nullspace. A plain conjugate-gradient iteration on a singular system drifts and can fail
to converge. gale handles this with a **deflated CG** that projects out the constant
component each iteration — the subject of Chapter 7, where we return to exactly this
singular pressure operator.

## How gale does it

The CPU reference lives in `src/dg/operators/stokes.rs`, in the `Stokes` struct. Its
public surface is the three stages above, packaged as one call:

- `Stokes::step` — one Stokes step: explicit forcing predictor, then
  `project_and_diffuse`.
- `Stokes::step_ns` / `step_ns_forced` — one Navier–Stokes step: the predictor also
  includes the explicit convection (nodal or split-form), and `step_ns_forced` takes a
  *precomputed* nodal body force, which is the hook viscoelastic coupling uses to inject
  the polymer-stress divergence \\( \nabla\cdot\boldsymbol\tau_p \\) (Chapter 8).
- `Stokes::project_and_diffuse` — the shared Stage 2 + Stage 3 engine: assemble the
  divergence \\( \nabla\cdot\hat{\mathbf{u}} \\), solve the deflated pressure-Poisson, apply
  the gradient correction, then solve the per-component viscous Helmholtz.

Internally the two operators are just `Poisson` instances from Chapter 5: the pressure
operator is `Poisson::with_bc(..., all-Neumann)` and the velocity operator is
`Poisson::with_reaction(..., λ)` carrying the \\( \lambda M \\) reaction term. The
divergence and the gradient correction are element-local `grad_x`/`grad_y` evaluations —
cheap, \\( O(N) \\) work.

The GPU solver, `GpuStokes` in `gale-gpu/src/flow.rs`, is the same algorithm with the
expensive part moved to the device. This is gale's **hybrid pattern** (Chapter 11): the
two elliptic solves — the bottleneck — run *entirely on the GPU* via `pressure_cg_solve`
(deflated pressure-Poisson) and `helmholtz_cg_solve` (viscous velocity), while the cheap,
element-local assembly (the explicit predictor, the divergence, the gradient correction,
the SIPG right-hand-side lifting) reuses the *validated host* `gale::dg` code unchanged.
There is no point porting \\( O(N) \\) assembly to the GPU when the dominant cost is the
iterative solves; and reusing the host assembly means the GPU stepper is, by construction,
bit-for-bit equal to the CPU oracle. On a non-conforming (AMR) mesh the same stepper
automatically routes the solves through the mortar-capable paths (`pressure_nc_cg_solve` /
`poisson_nc_cg_solve`, Chapter 10).

Both the CPU and GPU steppers are wrapped as integrators in the HOOMD-style framework of
Chapter 12 — `GpuStokesIntegrator` for pure Stokes and `GpuDualSplitting` for
Navier–Stokes with optional body force and convection scheme — so a full simulation is
assembled the same way regardless of which one drives it.

**Validation.** The scheme is checked against the **decaying Taylor–Green vortex**, an
exact analytic solution of Navier–Stokes on the unit square whose velocity decays like
\\( e^{-2\pi^2\nu t} \\). The tests confirm three things that map directly onto the theory:
the velocity error decreases under timestep refinement at the expected **first order in
time** (BDF1); both the nodal and the split-form convection paths reproduce the exact
vortex; and the whole dual-splitting solver — both elliptic solves through the mortar SIPG
operator — runs **stably and accurately on a non-conforming mesh**. The same Taylor–Green
case is what the GPU `GpuStokes` is held to, bit-for-bit against the CPU reference.

With the flow solver in hand, the obvious next question is *how* those two SPD elliptic
solves actually run on the GPU — matrix-free conjugate gradient, p-multigrid
preconditioning, and the deflation that tames the singular pressure operator. That is
Chapter 7.
