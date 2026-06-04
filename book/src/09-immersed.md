# Immersed Boundaries and Particles

> 🎓 **Reviewer — chapter verdict:** This is the strongest "stability story" chapter in the book — the explicit-vs-implicit penalization argument is correct, well-motivated, and lands the β→∞ punchline cleanly. The rigid→deformable→two-phase ladder is honestly framed and the "does it return to a reference shape?" litmus test is a genuinely good organizing idea. Two gaps keep it from being airtight: (1) the accuracy section conflates two distinct convergence axes (penalty error in η_b vs spatial error in h/p) under the single word "first-order," and (2) it omits the one thing every IBM practitioner gets burned by — the smeared mask is *not* a smooth field a high-order polynomial can represent, so you get Gibbs oscillations at the interface, and that is the real reason low η_b is double-edged. Fix those and this chapter is excellent.

So far we have solved fluid flow inside a fixed box. The headline application —
particle-laden suspensions — needs *objects in the flow*: a cylinder to flow past, a
disk to drag downstream, eventually a whole crowd of particles tumbling through a
channel. This chapter is about how a solid can live inside a fluid mesh **without being
meshed**, and — the recurring theme — how to do it without blowing the simulation up.

## The meshing problem

There are two ways to put a solid object into a fluid simulation, and they sit at
opposite ends of a trade-off.

The classical way is a **body-fitted** grid: you build the mesh so that element edges
lie *on* the object's surface. The boundary then falls exactly where the no-slip
condition is applied, you can cluster fine elements right in the boundary layer, and a
high-order method gets to show off — the surface is resolved to the full order \\( p \\).
For gale's hardest validation cases (the fixed-cylinder and 4:1-contraction
viscoelastic benchmarks) this is the accuracy gold standard, and gale supports static
body-fitted curved meshes for exactly those.

The trouble is *generating* such a mesh, and *re-generating* it. A single fixed obstacle
is a meshing chore; a moving obstacle means the mesh must deform or be rebuilt every step
(carrying a whole "mesh-motion solver," the geometric-conservation-law headache, and
mesh-quality decay); and *many moving particles* — the suspension target — is essentially
hopeless, because particles approach, rotate, and nearly touch, and no body-fitted mesh
survives that without tangling. Re-meshing also destroys gale's GPU performance model,
which assumes a static partition with persistent device buffers and peer-to-peer halos
(Chapter 11).

The alternative is to **stop fitting the mesh to the object** and instead keep a simple,
fixed background grid, then represent the object by *modifying the equations* wherever it
sits. This family of techniques is called the **immersed boundary method (IBM)**. You
give up exact, high-order resolution of the surface — the boundary now lives *inside*
elements, smeared over roughly a cell — but you gain enormous geometric freedom:
arbitrary shapes, arbitrary motion, arbitrarily many bodies, topology changes (particles
touching and separating) all for free, on a mesh that never changes. For moving bodies
and suspensions, that trade is the whole game.

> 🎓 **Reviewer:** Good, honest framing. One thing a newcomer should be told here, because it bites *immediately* on a high-order method: "smeared over roughly a cell" is not just a resolution penalty, it is a *smoothness* penalty. The mask χ jumps from 0 to 1 across the body surface, and a degree-\\( p \\) polynomial cannot represent a step without ringing (Gibbs). So the smeared boundary is the price you pay to keep the *field* representable at all — which is also why the `tanh` smoothing of the mask, mentioned later, is not cosmetic. Worth one sentence here so the reader expects the oscillation problem before it shows up.

> **The honest tension.** Body-fitted = accurate surface, painful geometry. Immersed =
> trivial geometry, smeared surface. gale uses *both*: static body-fitted meshes for
> fixed benchmark geometry, and immersed boundaries as the moving-body workhorse.

## gale's choice: Brinkman volume penalization

Of the immersed-boundary family, gale uses the simplest member that fits its
infrastructure: the **Brinkman volume-penalization** method.

The intuition is physical. Treat the solid not as a wall but as a **porous medium of
vanishing permeability** — a sponge so dense that fluid simply cannot move through it
relative to the solid. A porous medium exerts a drag force on the fluid proportional to
their relative velocity (Darcy's law). Make that drag enormous inside the body and zero
outside, and the fluid velocity is forced to match the solid's velocity wherever the
solid is — which is exactly the no-slip / rigid-body condition we wanted, now imposed as
a *body force* rather than a meshed boundary.

Concretely, we add a forcing term to the momentum equation,

\\[
  \mathbf{f}_\text{pen} = -\frac{\chi(\mathbf{x})}{\eta_b}\,
  \bigl(\mathbf{u} - \mathbf{u}_s\bigr),
\\]

with three ingredients:

- a **masking function** (indicator) \\( \chi(\mathbf{x}) \in [0,1] \\), equal to 1 inside
  the solid and 0 in the fluid (gale samples it onto the mesh nodes; a disk's mask is
  either a sharp Heaviside or a `tanh`-smoothed band of half-width `smooth`);
- the **solid velocity** \\( \mathbf{u}_s \\) — zero for a fixed body, or the rigid-motion
  field \\( \mathbf{u}_s(\mathbf{x}) = \mathbf{U} + \boldsymbol{\omega}\times(\mathbf{x}-\mathbf{x}_c) \\)
  for a translating/rotating particle;
- the **penalization (porosity) parameter** \\( \eta_b \\), with units of time. It is the
  porous medium's relaxation time: the smaller \\( \eta_b \\), the stiffer the drag and the
  more strictly the no-slip constraint is enforced. As \\( \eta_b \to 0 \\) the masked
  region behaves as a perfect rigid solid; the modelling error of a finite \\( \eta_b \\)
  scales like \\( \sqrt{\eta_b} \\) for a no-slip (Dirichlet) surface.

> 🎓 **Reviewer (deepen):** This is exactly the place to say *why low \\( \eta_b \\) is a double-edged sword*, because the reader will otherwise conclude "just take \\( \eta_b \\) as small as floating point allows." Driving \\( \eta_b \to 0 \\) shrinks the *modelling* error (the \\( \sqrt{\eta_b} \\) boundary-layer slip), but the penalized solution then develops an internal boundary layer of thickness \\( \sim\sqrt{\nu\,\eta_b} \\) right at the mask edge — and your mesh has to resolve *that* layer or the high-order element rings across it. So the two error sources pull in opposite directions: too-large \\( \eta_b \\) leaks flow through the body; too-small \\( \eta_b \\) creates a sub-cell layer the polynomial cannot see, and you get spurious oscillations and a noisy drag signal. The practitioner's rule of thumb is to balance the penalty layer against the local mesh size, not to minimize \\( \eta_b \\) blindly. The fact that gale's tests sit at \\( 10^{-3} \\)–\\( 10^{-4} \\) rather than \\( 10^{-8} \\) is precisely this balance — worth saying so out loud.

So far this is just an extra term in the momentum equation. The interesting part — the
part this chapter is really about — is *how you integrate it in time*.

## Why apply it implicitly (the stability lesson)

Look again at the forcing. Inside the body \\( \chi = 1 \\), so the drag coefficient is
\\( 1/\eta_b \\). If we want a near-rigid body we take \\( \eta_b \\) tiny — gale's tests use
\\( \eta_b = 10^{-3} \\) or \\( 10^{-4} \\) — which makes \\( 1/\eta_b \\) *enormous*. The
penalty term is, in the language of Chapter 6, **extremely stiff**.

Now suppose we treated it like any other source term and added it explicitly to the
right-hand side: \\( \mathbf{u}^{n+1} = \mathbf{u}^{n} - (\Delta t\,\chi/\eta_b)(\mathbf{u}^n-\mathbf{u}_s) \\).
That is forward Euler on the relaxation ODE \\( \dot{\mathbf{u}} = -(\chi/\eta_b)(\mathbf{u}-\mathbf{u}_s) \\),
and it is only stable when the step is smaller than the relaxation time,
\\( \Delta t \lesssim \eta_b \\). With \\( \eta_b = 10^{-4} \\) that forces a *catastrophic*
time step — and the stiffer (more rigid) we make the body, the smaller the step we must
take. Explicit penalization makes \\( \Delta t \to 0 \\) exactly in the limit we care about.
This is the same villain as in every other chapter: a physically reasonable scheme that
is unconditionally *unstable* for the regime of interest.

> 🎓 **Reviewer:** This argument is correct and it is the heart of the chapter — keep it exactly as is. One tiny precision so a careful reader doesn't trip: forward Euler on \\( \dot u = -(\chi/\eta_b)(u-u_s) \\) is stable for \\( \Delta t < 2\eta_b/\chi \\) (the factor of 2), so "\\( \Delta t \lesssim \eta_b \\)" is right in spirit and right in scaling. The scaling is the whole point, so I would not clutter the prose with the 2 — just flagging that the inequality is an order-of-magnitude statement, which the "\\( \lesssim \\)" already signals. No change needed.

The cure is to integrate the relaxation **implicitly** (backward Euler). The drag ODE
\\( \dot{\mathbf{u}} = -(\chi/\eta_b)(\mathbf{u}-\mathbf{u}_s) \\), solved implicitly over a
step, has a closed form — it is just a convex combination:

\\[
  \mathbf{u} \;\leftarrow\; \frac{\mathbf{u} + \beta\,\mathbf{u}_s}{1 + \beta},
  \qquad
  \beta = \chi\,\frac{\Delta t}{\eta_b}.
\\]

Read this off: outside the solid \\( \chi = 0 \Rightarrow \beta = 0 \\), and the velocity
is untouched. Inside, \\( \beta \\) is huge, and the update pulls \\( \mathbf{u} \\) almost
all the way to \\( \mathbf{u}_s \\) — the residual after one step is
\\( 1/(1+\beta) \approx \eta_b/\Delta t \\), small precisely *because* \\( \eta_b \\) is
small. Because the new velocity is a weighted average of two existing velocities, it can
never overshoot: the update is a contraction, **unconditionally stable** in \\( \beta \\),
for *any* \\( \eta_b \\) and *any* \\( \Delta t \\). The stiffer we make the body, the
*better* this behaves. That is the whole reason gale writes the penalization this way
rather than as an additive source.

> **Where it lives in the code.** Because the update is a relaxation \\( \mathbf{u}
> \leftarrow g(\mathbf{u}) \\) and *not* an additive right-hand-side contribution, gale
> does not put it in the equation's flux/source assembly. It lives in a **post-stage
> hook** — applied to the velocity field *after* each integrator stage — which is exactly
> the structural consequence of choosing the implicit form. The framework's
> `PenalizationHook` (and `Penalization3dHook` in 3D) is that home; it simply calls the
> validated relaxation on the velocity components after the dual-splitting step.

## The accuracy caveat (and the forward link)

Volume penalization buys stability and geometric freedom, but it has a known, documented
weakness: it **smears the boundary over roughly a cell** and is only about **first-order
accurate at the interface** — in fact the no-slip velocity error scales like
\\( \sqrt{\eta_b} \\), and the polymer-stress field does not even converge pointwise right
at the surface. High polynomial order \\( p \\) does *not* rescue this: spectral accuracy
needs a smooth field within the element, and the penalized solution is *not* smooth across
a sharp body surface. So high \\( p \\) still pays off in the bulk (the matrix flow, the
stress transport of Chapter 8), but near the body the interface error dominates.

> 🎓 **Reviewer (flag):** Two genuinely different error axes are being braided together under the word "first-order," and a sharp reader will catch the seam. There is the *penalty* error (how close the model is to true no-slip), which scales like \\( \sqrt{\eta_b} \\) as \\( \eta_b \to 0 \\); and there is the *discretization* error (how well the mesh resolves the smeared/penalized solution), which is the \\( O(h) \\) / "first-order at the interface" story as \\( h \to 0 \\). The paragraph slides from "\\( \sqrt{\eta_b} \\)" (penalty axis) to "first-order accurate at the interface" (spatial axis) to "the near-surface field is only half-order" (back to penalty axis) without naming the switch. The claims are individually defensible, but the reader can't tell whether you're refining \\( \eta_b \\) or \\( h \\). I'd add one clause: "two knobs limit accuracy here — the penalty \\( \eta_b \\) and the mesh \\( h \\) — and both bottom out at low order near the surface." Then the "high \\( p \\) doesn't rescue this" point is unambiguous: it's the *smoothness* of the penalized field that caps the order, on either axis.

The good news, and the reason this remains usable, is that **integrated quantities still
converge**: the net force and torque on the body are first-order accurate even though the
near-surface field is only half-order. For the rigid-particle proof-of-concept — where we
care about drag, lift, and tumbling rate, not the pointwise wall stress — that is enough.

The principled remedy is to **localize refinement to the boundary**: pile up \\( h \\)- and
\\( p \\)-resolution in a thin band around the surface so the smeared interface is at least
resolved as finely as possible. This is precisely the subject of the Nayak–Mavriplis
paper on hp-adaptive volume-penalty DG-SEM, and it is the natural way to make the IBM
*accurate*. Be honest, though: **gale does not yet have this.** gale has \\( h \\)-adaptivity
in 2D (Chapter 10), but no \\( p \\)-adaptivity and no IB-targeted refinement criterion —
both are documented gaps on the roadmap (Chapter 13). The accuracy boost from localized
hp-refinement is a planned capability, not a built one.

## Diagnostics: drag on the body

The penalization gives us the hydrodynamic force on the body essentially for free. The
fluid loses momentum to the body at exactly the rate the implicit relaxation removes it,
so the force the fluid exerts on the solid is

\\[
  \mathbf{F} = \int_\Omega \frac{\chi}{\eta_b}\,
  \bigl(\mathbf{u} - \mathbf{u}_s\bigr)\,dV.
\\]

Evaluated on the penalized velocity, this is consistent with the update by construction.
gale exposes it as `VolumePenalization::force` and, in the framework, as the
`PenalizationDrag` `Compute`. The tests confirm the physics you would demand: for a disk
in a body-force-driven Stokes channel the drag points **downstream** (\\( F_x > 0 \\)), the
lift is **machine-zero by symmetry** (\\( F_y \approx 0 \\)), and — being the Stokes regime —
the drag is **exactly linear in the drive** (doubling the forcing doubles the drag, ratio
2.000).

For a *freely suspended* particle there is a companion operation: an L2 projection of the
fluid velocity onto rigid motions over the masked region recovers the body's translation
and angular velocity. A disk in simple shear comes out rotating at exactly half the
ambient vorticity, and an ellipse reproduces the classical **Jeffery orbit**, with a
tumbling period matching the analytic \\( (\pi/\dot\gamma)(r + 1/r) \\) to better than 0.1%.

## Toward deformable particles and two-phase flow

What gale does today is one rung on a ladder. It is worth distinguishing the rungs, because
they get progressively harder and the later ones are gale's *future direction, not current
capability*.

**(a) Rigid / forced immersed bodies — built.** The body has a prescribed (or rigidly
evolving) velocity; we only force the fluid to match it. This is everything above: the
mask is the body, and the implicit relaxation enforces it. Rigid Jeffery-orbit dynamics
are in via the rigid-motion projection.

**(b) Deformable particles / capsules — planned.** Now the immersed object is an *elastic
membrane* (think a red blood cell or a polymer capsule) that deforms in response to the
flow and pushes back — genuine two-way **fluid–structure interaction**. This needs a
*Lagrangian* representation of the membrane carrying its own elastic constitutive law,
with forces spread to the fluid and the fluid velocity interpolated back to advect the
markers (a Peskin-style front-tracking coupling). It is harder because the interface
*moves with its own physics* and you must watch membrane stiffness (a new time-step
constraint), area/volume conservation, and locking. gale has only a *foundation* here
(`src/dg/membrane.rs`: a DG-basis spread/interpolate pair and a stretching-spring capsule
that relaxes stably) — not a validated deformable-particle solver.

> 🎓 **Reviewer:** Honestly scoped, and right to name the membrane-stiffness time-step constraint — that explicit elastic-force CFL (\\( \Delta t \lesssim \sqrt{m\,h/k} \\) in the stiff-membrane limit) is the classic Peskin-IBM killer and the reason people reach for implicit or semi-implicit membrane integrators. Worth one half-sentence that this stiffness constraint is the *same kind of villain* as the penalty stiffness you just defeated in section "Why apply it implicitly" — it ties the ladder back to the chapter's throughline and tells the reader the cure (implicit treatment) is already familiar. Don't expand beyond that; the rung is planned, not built, and the section correctly resists over-claiming.

**(c) True two-phase flow — planned/parked.** Here the "particle" is *itself a fluid* — a
droplet of one fluid in another (e.g. a viscoelastic drop in a Newtonian matrix) — with an
interface that advects, stretches, and can break or merge. This is *not* an immersed-solid
problem at all: there is no rest shape to return to, so the right tool is an
interface-tracking method (gale's design points at a phase-field / Cahn–Hilliard
formulation, which pairs naturally with high-order DG and handles topology changes
automatically). It is the hardest rung — two evolving constitutive fields plus a moving
interface plus interfacial-stress singularities — and it is firmly future work.

The litmus test that organizes these is simple: *does the material return to a reference
shape?* If yes, it is a solid and belongs to the immersed-solid methods (a, b). If no, it
flows indefinitely and is a fluid phase (c). gale's capsule target straddles the seam — an
elastic membrane (solid) enclosing an interior fluid distinct from the exterior
(two-phase) — which is why it is the genuinely ambitious end of the roadmap.

## How gale does it

- **`VolumePenalization` (2D) / `VolumePenalization3d`** (`src/dg/immersed/`): sample a
  solid's mask \\( \chi \\) and velocity \\( \mathbf{u}_s \\) onto the mesh nodes; `apply`
  performs the implicit Brinkman relaxation \\( \mathbf{u} \leftarrow (\mathbf{u}+\beta\mathbf{u}_s)/(1+\beta) \\)
  in place; `force` returns the penalization drag; `project_rigid` recovers a
  freely-suspended body's rigid velocity. The `Disk` / `RigidBody` (disk or ellipse)
  shapes provide sharp or `tanh`-smoothed indicators.
- **Framework hooks** (`src/sim/ibm.rs`): `PenalizationHook` / `Penalization3dHook`
  (`StateStageHook`s) apply the relaxation after each integrator stage — the structural
  home demanded by the implicit form — and `PenalizationDrag` (`Compute`) reports a force
  component.
- **On the GPU** (`gale-gpu/src/immersed.rs`): the `penalize` / `penalize3d` kernels are
  embarrassingly parallel (one thread per node, no special functions) and validated
  bit-for-bit against the CPU oracle; `GpuPenalizationHook` / `GpuPenalization3dHook` wire
  them into the GPU flow integrators.
- **The capstone** (Chapter 1): a penalized rigid disk (2D) and sphere (3D) immersed in a
  **log-conformation Oldroyd-B** channel flow (Chapter 8), end-to-end on the GPU — flow
  suppressed inside the body, drag downstream, and the conformation tensor \\( \mathbf{C} =
  \exp(\boldsymbol{\Psi}) \\) staying symmetric-positive-definite *everywhere, including at
  the immersed surface*. The research-flagged risk (stress pathology at the body) did not
  bite at moderate Weissenberg number — the log-conformation representation is exactly the
  stabilizer the literature prescribes for it.

The next chapter takes up the other half of the moving-body strategy: refining the mesh
*around* these features — the boundary layers, the smeared interface, the wakes — without
paying for fine resolution everywhere.
