# Hyperbolic Operators and Stability

[Chapter 3](03-dgsem.md) built the machinery: nodal spectral elements on quads, a
diagonal mass matrix from GLL collocation, and the discontinuous Galerkin weak form
that couples elements only through their shared faces. This chapter puts that machinery
to work on the *transport* half of fluid dynamics — and, more importantly, it is where
the book's throughline begins in earnest. Almost everything here is a stability story.
A high-order method that merely *interpolates* accurately will, on a real nonlinear
flow, happily march itself to `NaN`. The job of this chapter is to explain the handful
of ideas that stop it.

## The transport half of fluid dynamics

A **hyperbolic conservation law** has the form

$$
\partial_t u + \nabla \cdot \mathbf{F}(u) = 0,
$$

where $ u $ is a conserved quantity (or a vector of them) and $ \mathbf{F}(u) $ is
its flux. The defining feature is that information travels at finite speed along
characteristics: a disturbance here shows up *there* a little later, carried by the
flow. This is *transport*, and it is exactly the part of fluid motion that advection
embodies. gale's hyperbolic operator covers a small zoo of these laws, each a step up in
difficulty:

- **Linear advection**, $ \partial_t u + \mathbf{a}\cdot\nabla u = 0 $ — a fixed field
  $ u $ carried at constant velocity $ \mathbf{a} $. The flux $ \mathbf{F} = \mathbf{a}\,u $
  is *linear* in $ u $. This is the harmless one, and the correctness oracle.
- **Burgers**, $ \partial_t u + \partial_x(u^2/2) = 0 $ — the simplest *nonlinear*
  law. A scalar that advects itself; smooth data steepens into a shock in finite time.
  It is the minimal test bed for every nonlinear-stability trick below.
- **Compressible Euler** — the full inviscid gas-dynamics system, conserved variables
  $ (\rho, \rho u, \rho v, E) $. A genuine nonlinear system with multiple wave
  families (acoustic and entropy waves).
- **Incompressible convection** — the momentum-transport term $ (\mathbf{u}\cdot\nabla)\mathbf{u} $
  written as a conservation law with flux $ \mathbf{u}\otimes\mathbf{u} $. This is the
  piece that lives *inside* the incompressible Navier–Stokes solver
  ([Chapter 6](06-incompressible.md)); the viscous and pressure parts are handled
  separately by the elliptic operators of [Chapter 5](05-elliptic.md).

For a single element, the DG weak form from Chapter 3 reads, schematically,

$$
M\,\partial_t u = \underbrace{D^{\!\top}(W\,\mathbf{F})}_{\text{volume}} \;-\; \underbrace{\oint \mathbf{F}^*\!\cdot\mathbf{n}}_{\text{surface}},
\qquad \partial_t u = M^{-1}(\cdots),
$$

with $ M = \mathrm{diag}(Jw) $ diagonal (the gift of collocation, Chapter 3). The
volume term is element-local. The surface term is the *only* place neighbouring elements
talk to each other, and it is the whole reason DG runs well on a GPU: weak,
face-localized coupling. It is also where the stability story lives, because that surface
term contains a quantity we have not yet defined — the **numerical flux** $ \mathbf{F}^* $.

## The numerical flux: who decides the value on the face?

In DG the solution is **discontinuous** across element interfaces by construction. Stand
on a shared face and look both ways: the element on the left says the solution there is
$ u^- $, the element on the right says it is $ u^+ $, and in general $ u^-\ne u^+ $.
The flux $ \mathbf{F}\cdot\mathbf{n} $ leaving one element must equal the flux entering
its neighbour — but with two different states there are two candidate fluxes and we must
agree on *one*. That single agreed value is the numerical flux $ \mathbf{F}^* $.

This is precisely a **Riemann problem**: two constant states meeting at an interface,
asking what flows between them. We do not need the exact Riemann solution (that would be
expensive and, for a face evaluated millions of times per step, wasteful); we need a
cheap, consistent *approximate* answer. "Consistent" has a precise meaning: when the two
states agree, $ u^-=u^+=u $, the numerical flux must reduce to the true flux,
$ \mathbf{F}^*(u,u)=\mathbf{F}(u)\cdot\mathbf{n} $. Anything consistent is a *candidate*;
stability decides which candidate is *usable*.

### Why the obvious choice is unstable

The most natural numerical flux is the **central** (average) flux,

$$
\mathbf{F}^* = \tfrac{1}{2}\big(\mathbf{F}(u^-) + \mathbf{F}(u^+)\big)\cdot\mathbf{n}.
$$

It is consistent, symmetric, and looks unimpeachable. It is also, for a pure transport
problem, the wrong choice — because it adds **no dissipation at all**. Here is the
intuition. Multiply the semi-discrete scheme by the solution and sum over the mesh — this
measures how the total "energy" $ \tfrac12\int u^2 $ (or, for systems, a convex
*entropy*) evolves in time. The volume terms telescope cleanly. For **linear advection
with a central flux** the surface terms also telescope, leaving only a boundary
contribution: the discrete energy is *exactly conserved*. That sounds benign, but it is
precisely the danger — energy-neutral means there is *nothing* to remove energy from the
under-resolved modes at the interface. Any small ripple at the grid scale persists
undamped, and under time-stepping or the slightest nonlinearity that missing damping sink
is what lets it grow. On a linear problem the scheme merely sits at the edge of stability;
on a nonlinear one it is a slow-motion explosion.

### Upwinding and the Rusanov flux: dissipation buys stability

The cure is to *respect the direction information travels*. Transport is directional: if
the wind blows left-to-right, the value on the face should be informed more by the
upwind (left) state than the downwind one. This is **upwinding**, and it is the oldest
idea in computational transport. The practical, robust way to bake it in for any
conservation law is the **Rusanov flux**, also called the **local Lax–Friedrichs (LLF)**
flux:

$$
\boxed{\;\mathbf{F}^* = \tfrac{1}{2}\big(\mathbf{F}(u^-)+\mathbf{F}(u^+)\big)\cdot\mathbf{n}
\;-\; \tfrac{1}{2}\,\lambda\,\big(u^+ - u^-\big)\;}
$$

where $ \lambda = \max\big(|s(u^-,\mathbf{n})|,\,|s(u^+,\mathbf{n})|\big) $ is the
largest signal (wave) speed in the face-normal direction, evaluated from both sides. The
first term is just the central flux. The second term is the entire stability mechanism:
it is **dissipation proportional to the jump**. Where the solution is smooth and
continuous, $ [u]\to 0 $ and the extra term vanishes — so we keep full high-order
accuracy on resolved features. Where there is a jump — a shock, or an under-resolved
ripple — the term switches on and drains energy out of it, enough to make the discrete
energy/entropy balance *non-increasing*, and a bit to spare. The dissipation is what buys
stability. The price of that generosity is bluntness: for the **Euler system** the scalar
factor $ \lambda(u^+-u^-) $ damps *every* characteristic field by the *fastest* wave
speed, which is exactly why Rusanov is the most diffusive of the Riemann solvers.

Two things are worth pausing on. First, $ \lambda $ is genuinely *local*: it is the
local maximum wave speed, not a global constant, which is why this is the "local"
Lax–Friedrichs flux and why it adds the least dissipation that does the job. Second, the
choice of $ \lambda $ is the only physics the flux needs to know about the law — which
is exactly how gale factors it (below). Rusanov is the workhorse: cheap, robust, and
applicable to any law for which you can name a maximum wave speed. More accurate Riemann
solvers (HLL, HLLC, Roe) exist and add less dissipation, but Rusanov is the right default
and the one gale uses.

## Conservation and consistency, for free

The surface formulation gives two structural guarantees almost for free, and both matter.

**Discrete conservation.** Because the *same* single-valued $ \mathbf{F}^* $ is
subtracted from one element and added to its neighbour across their shared face (with
opposite outward normals), the face contributions **telescope** when you sum over the
whole mesh: every interior flux appears twice with opposite sign and cancels. What
survives is only the flux through the true domain boundary. The discrete scheme therefore
conserves the total of $ u $ exactly, up to what enters or leaves at the boundary — the
discrete echo of $ \partial_t\int u = -\oint \mathbf{F}\cdot\mathbf{n} $. This is the
finite-volume DNA inside DG, and it is not optional book-keeping: it is why shocks travel
at the right speed and why mass/momentum/energy do not silently leak.

**Consistency / free-stream preservation.** A uniform state must have exactly zero
residual — a constant should sit still. With a consistent flux it does, to round-off.
gale tests this directly: `euler_free_stream_preserved` checks that a uniform Euler state
gives a residual below $ 10^{-9} $, and the same holds across hanging nodes on a refined
mesh (`euler_free_stream_on_refined_mesh`). It is a humble test that catches an enormous
class of bugs.

## High order's hidden danger: aliasing instability

Upwinding tames the *interface*. But high-order DG on a *nonlinear* law has a second,
subtler enemy that lives entirely *inside* each element: **aliasing**.

Here is the mechanism. Inside an element the solution is a polynomial of degree $ p $.
A nonlinear flux multiplies fields together — for Burgers the flux is $ u^2/2 $, for
the incompressible convection it is $ u_i u_j $, for Euler it is ratios and products of
the conserved variables. The product of two degree-$ p $ polynomials has degree
$ 2p $. But our element can only *represent* degree $ p $. When we collocate the
nonlinear flux at the GLL nodes and then differentiate, the high-frequency content of the
true product (the part above degree $ p $) does not just disappear — it gets
misrepresented as, *aliased onto*, the low modes the grid *can* carry. The name is borrowed
straight from signal processing, where under-sampling a high frequency makes it masquerade
as a lower one. Energy that should have lived at unresolved scales is folded back into the
resolved ones. On a marginally resolved, high-Reynolds-number flow this aliased energy has
nowhere to dissipate; it piles up at the grid scale and the simulation blows up. This is
**aliasing instability**, and it is the reason naive high-order schemes for turbulent or
high-Re flow are notoriously fragile. It is not a bug in the flux or the boundary handling
— it is intrinsic to taking nonlinear products in a finite polynomial space.

gale supports two cures, and they are complementary.

### Cure 1: the split / entropy-stable volume form

The deep fix is to change *how the volume term discretizes the nonlinearity* so that the
scheme respects a **secondary conservation law** — kinetic energy or entropy —
discretely, not just the primary one. This is the **split-form** (or skew-symmetric, or
kinetic-energy-preserving, or entropy-stable) family, built on **summation-by-parts (SBP)**
operators and **Fisher–Carpenter flux-differencing**.

The intuition is worth getting right, because it is one of the most beautiful ideas in
the subject. Continuous nonlinear PDEs satisfy *two* conservation statements at once: a
primary one (mass, momentum, energy) and a secondary one (the kinetic energy or the
mathematical entropy is conserved by the inviscid dynamics and only ever *decreased* by
physical dissipation). A standard discretization conserves the primary quantity but lets
the secondary one drift — and that drift *is* the aliasing blow-up. The split form
rewrites the volume divergence not as a single derivative of a collocated flux, but as a
**flux-difference built from a special two-point flux** $ \tilde{\mathbf{F}}^{\#}(u_i,u_j) $
evaluated between every pair of nodes on a line. On the **reference** element the
one-dimensional form is

$$
(\nabla\cdot\mathbf{F})_i \;\approx\; 2\sum_{j} D_{ij}\,
\tilde{\mathbf{F}}^{\#}(u_i,u_j),
$$

with $ D $ the SBP/collocation derivative matrix and the metric terms entering as in
the standard map (in particular, the Jacobian is *not* a scalar $ 1/J $ lumped into the
sum — that shortcut only holds for an affine 1D map). The SBP property is most naturally
stated on $ Q = MD $, where $ Q + Q^\top = B $; the factor of 2 and the $ M^{-1} $
of the mass matrix are placed to be consistent with that identity.

If the two-point flux is **symmetric** and **consistent** (it reduces to the physical
flux when $ u_i=u_j $), the SBP property of the differentiation matrix makes the volume
term telescope *in the entropy variables too*. The result: the semi-discrete scheme
discretely conserves entropy/kinetic energy (an **entropy-conserving** scheme), and adding
the Rusanov interface dissipation on top turns "conserving" into "non-increasing" — an
**entropy-stable** scheme. Entropy can only go down, never up, so there is no mechanism
for aliased energy to accumulate. What this *proves* is a semi-discrete entropy bound: the
spatial operator cannot manufacture entropy. That is exactly what makes the scheme
survivable at high Reynolds number without blowing up — though it is worth being honest
about what the guarantee does *not* cover. It is semi-discrete, so a finite-$ \Delta t $
time integrator can still violate the bound (fully-discrete entropy stability needs
relaxation Runge–Kutta or similar); it keeps the solution *bounded*, not *accurate*, so an
under-resolved mesh can still produce nonphysical oscillations; and for systems it
presumes the discrete state stays where the entropy is convex (positive density and
pressure), which the split form alone does not enforce.

The choice of two-point flux is where the law's identity lives, and gale's
`ConservationLaw` trait makes this explicit via a `two_point_flux` method: each law
supplies its own entropy-conserving (or kinetic-energy-preserving) two-point flux, with
Euler's needing the numerically-careful logarithmic mean $ (a-b)/(\ln a - \ln b) $
(gale's `ln_mean`, with a series expansion near $ a=b $ to avoid catastrophic
cancellation).

The default `two_point_flux` is just the central average $ \tfrac12(\mathbf{F}(u_L)+\mathbf{F}(u_R)) $;
nonlinear laws override it with their entropy-conserving counterpart. gale verifies the
payoff directly. `euler_split_form_is_entropy_conservative` runs the split form with the
interface dissipation **off** and checks that the semi-discrete entropy rate
$ \sum Jw\,(\mathbf{w}\cdot\partial_t u) $ vanishes to round-off — a decisive test that
the Chandrashekar flux is exactly entropy-conserving. And
`split_form_burgers_is_entropy_stable_through_a_shock` steepens $ \sin(2\pi x) $ into a
shock on a deliberately under-resolved grid and confirms that $ \|u\|_2 $ is
non-increasing through the shock: the high-Re robustness guarantee, demonstrated. A nice
sanity check sits alongside them — for *linear* advection there is no aliasing, so the
split form and the weak form are algebraically identical
(`split_form_matches_weak_for_linear_advection`). This is exactly the kind of "the
abstraction collapses to the trivial case when it should" test that builds trust, and it
is also the cleanest possible demonstration of *why* aliasing is a purely nonlinear
disease.

There is a dissipation toggle worth naming. gale exposes `dissipation: bool` on the
operator: `true` adds the Rusanov jump term (entropy-*stable*); `false` uses a pure
central interface flux (entropy-*conserving*, the configuration the round-off test above
relies on). Entropy-conserving is the diagnostic; entropy-stable is what you run.

### Cure 2: modal filtering, the gentle last resort

The split form is the structural cure and the one to reach for first. But gale also ships
a cheaper, blunter instrument: a **modal filter** (`ModalFilter`), in the spectral-vanishing-viscosity
(SVV) family. Each element's nodal field is transformed into its **Legendre modal**
coefficients, every mode is multiplied by a damping factor, and it is transformed back.
The damping is chosen to be surgical:

$$
\sigma(k) = \begin{cases} 1, & k \le k_{\text{cut}} \\ \exp\!\big(-\alpha\,\eta^{2s}\big), & k > k_{\text{cut}}\end{cases},
\qquad \eta = \frac{k-k_{\text{cut}}}{p-k_{\text{cut}}}.
$$

Modes below the cutoff are preserved *exactly* (so resolved features keep full
accuracy — `preserves_modes_below_cutoff` verifies this), while the highest modes — where
aliased garbage collects — are attenuated. A typical mild setting is
$ k_{\text{cut}}=p-1,\ \alpha=36,\ s=8 $, which leaves everything but the top mode
untouched and damps that one (`damps_the_top_mode`). The filter is applied per element as
a tensor-product operator, so it stays cheap and GPU-friendly. Treat it as a *last resort*
and a complement to the split form, not a replacement: it adds dissipation by fiat rather
than by a conservation principle, so it can mask a genuinely under-resolved simulation if
you lean on it too hard.

## How gale does it

The hyperbolic machinery lives in `src/dg/operators/hyperbolic.rs`, organized around one
trait and one struct:

- **`ConservationLaw`** is the pluggable physics: `n_vars`, the physical `flux`, the
  `max_wave_speed` (for the Rusanov $ \lambda $), and the optional `two_point_flux` for
  the split form. The four implementors are `LinearAdvection`, `Burgers`, `Euler`, and
  `IncompressibleConvection`. New physics is a new `impl` — the operator code does not
  change.
- **`Hyperbolic<L>`** is the discretization over a `Mesh2d`, parameterized by the law
  `L`, a **`VolumeForm`** (`Weak` or `SplitForm`), and the `dissipation` flag. `rhs`
  dispatches to `rhs_weak` (collocated flux, Dxᵀ(W F) volume + Rusanov surface) or
  `rhs_split` (Fisher–Carpenter flux-differencing volume + strong-form surface). Both
  paths handle 2:1 non-conforming faces through the conservative mortar projection
  ([Chapter 10](10-amr.md)). Time advance is `step_ssp_rk3`, the standard
  strong-stability-preserving Runge–Kutta integrator for DG.

The same pattern carries to the GPU. `gale-gpu/src/operators/burgers.rs` ports the
split-form Burgers RHS — the Fisher–Carpenter volume with metric averaging plus the
strong-form central surface flux — to a single device kernel (one element per block, one
node per thread), and it is validated **bit-for-bit** against the CPU `Hyperbolic` split
form. As [Chapter 1](01-introduction.md) put it, the CPU is the oracle: the GPU earns
trust by reproducing the host result to round-off, not by being independently plausible.

The thread to carry into the next chapter: hyperbolic operators stay stable by *adding
dissipation in the right places* — at interfaces via upwinding, and against aliasing via
split forms or filtering. The elliptic operators of [Chapter 5](05-elliptic.md) face the
opposite difficulty. There the danger is not too little dissipation but a discrete
operator that fails to be positive-definite at all — and the fix is a penalty parameter
that has to be tuned with exactly the same care, and for exactly the same reason: keeping
the simulation stable.
