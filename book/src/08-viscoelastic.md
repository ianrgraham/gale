# Viscoelastic Flow and the High-Weissenberg Problem

This is the chapter the book has been pointing at. Everything so far — the
discontinuous Galerkin discretization (Part II), the dual-splitting velocity solve
(Chapter 6), the GPU elliptic solvers (Chapter 7) — was infrastructure. Now we add the
thing that makes gale's target application *hard*: the fluid has **memory**. And with
memory comes the recurring villain of this whole book, in its sharpest form yet — a
numerical instability so notorious it has its own name, the **High-Weissenberg-Number
Problem**, and a cure so elegant it is worth the entire chapter to motivate.

If you read only one chapter for the "stability throughline," read this one.

## Why polymers change everything

In Chapter 2 the stress of a Newtonian fluid was a purely *local, instantaneous* object.
The viscous stress at a point depends only on the rate of strain *right now, right
there*:

$$
\boldsymbol{\tau} = 2\mu\,\mathbf{D}, \qquad \mathbf{D} = \tfrac{1}{2}\big(\nabla\mathbf{u} + \nabla\mathbf{u}^{T}\big).
$$

Stop shearing the fluid and the stress vanishes the same instant. Water has no past.

A **dilute polymer solution** does not behave this way. Dissolve long-chain polymer
molecules into a solvent and each molecule is, mechanically, a tiny entropic spring. At
rest it is a random coil; the flow stretches and rotates it; and — crucially — it does
not snap back instantly. It relaxes over a characteristic **relaxation time** $ \lambda
$. During that relaxation it stores elastic energy and pushes back on the fluid. So the
stress at a point and instant depends on the entire recent *history* of deformation that
the fluid element carried with it. The fluid is **viscoelastic**: part viscous (it
dissipates), part elastic (it remembers and springs back).

This is not a small correction. It is the origin of every counterintuitive effect in the
field — rod-climbing (the Weissenberg effect), die swell, elastic turbulence at
vanishing Reynolds number, the purely-elastic instabilities that destabilize
microfluidic flows where inertia is negligible. It is exactly gale's regime: **low
Reynolds number, high Weissenberg number**, where elasticity, not inertia, runs the show
(Chapter 2).

Mathematically, memory forces a structural change to the model. We can no longer write
the polymer stress as an algebraic function of the present velocity gradient. Instead we
must carry an **extra evolving tensor field** that encodes the current microstructural
state of the polymers, governed by its **own partial differential equation**, coupled to
the momentum equation. The single-PDE Newtonian problem of Chapter 6 becomes a coupled
system: momentum **plus** a constitutive transport equation. That second PDE, and the
fact that its unknown must stay physically admissible, is the whole subject of this
chapter.

## The conformation tensor $ \mathbf{C} $

The extra field we evolve is the **conformation tensor** $ \mathbf{C} $. Intuitively it
is the (ensemble-averaged) shape of the polymer coils. Model each polymer as an
end-to-end vector $ \mathbf{q} $ connecting its two ends; then

$$
\mathbf{C} = \langle \mathbf{q}\,\mathbf{q}^{T}\rangle,
$$

the second moment of that vector, averaged over all the polymers in a fluid element. It
is a coarse-grained microstructure variable: it throws away the detailed configuration
and keeps just the mean stretch and orientation.

Geometrically, $ \mathbf{C} $ is literally an *ellipsoid*: its eigenvectors are the
principal axes of the average coil's orientation, and its eigenvalues are the mean-square
stretch along each axis. At rest the ellipsoid is a unit sphere ($ \mathbf{C} = \mathbf{I}
$); shear it and the sphere tilts and elongates into a cigar pointing roughly downstream.
This picture makes "SPD" stop being an abstract matrix property and become the obvious
statement that *an ellipsoid cannot have a negative or zero axis length* — which is exactly
the invariant the whole chapter is about protecting.

Three facts about $ \mathbf{C} $ carry the entire chapter, so we state them plainly:

1. **$ \mathbf{C} $ is symmetric.** It is an outer product averaged, $ \langle
   \mathbf{q}\mathbf{q}^{T}\rangle $, so $ C_{ij} = C_{ji} $ identically. In 2D it has
   three independent entries $ [C_{xx}, C_{xy}, C_{yy}] $; in 3D, six $ [C_{xx},
   C_{xy}, C_{xz}, C_{yy}, C_{yz}, C_{zz}] $. gale stores exactly these — see the
   `[Vec<f64>; 3]` and `[Vec<f64>; 6]` layouts below.

2. **$ \mathbf{C} $ is positive-definite (SPD).** For any direction $ \mathbf{a} $,
   $ \mathbf{a}^{T}\mathbf{C}\,\mathbf{a} = \langle (\mathbf{a}\cdot\mathbf{q})^{2}\rangle
   \ge 0 $: it is a mean square, and it cannot be negative. An eigenvalue of $
   \mathbf{C} $ is the mean-square stretch of the coils along an eigendirection. A
   *negative* eigenvalue would mean a negative mean-square length — physically
   meaningless. **$ \mathbf{C} $ lives strictly inside the cone of SPD matrices.**

3. **Equilibrium is $ \mathbf{C} = \mathbf{I} $.** With no flow, the coils relax to
   their isotropic equilibrium size (nondimensionalized to unit length), so $ \mathbf{C}
   \to \mathbf{I} $.

The polymer's contribution to the stress is an explicit, *algebraic* function of $
\mathbf{C} $ (for the Oldroyd-B model):

$$
\boldsymbol{\tau}_{p} = \frac{\eta_{p}}{\lambda}\,(\mathbf{C} - \mathbf{I}),
$$

where $ \eta_{p} $ is the polymer viscosity and $ \lambda $ the relaxation time.
Stretched coils ($ \mathbf{C} \succ \mathbf{I} $) push; the elastic stress is the
departure of the microstructure from equilibrium.

Hold onto fact 2. It is the linchpin. The map $ \mathbf{C} \mapsto \boldsymbol{\tau}_p
$ is harmless as written, but it amplifies any error in $ \mathbf{C} $: if a numerical
scheme lets $ \mathbf{C} $ drift out of the SPD cone, $ \boldsymbol{\tau}_p $ becomes
unphysical and, as we will see, the simulation detonates. **The SPD property of $
\mathbf{C} $ is the entire stability story of viscoelastic flow.**

## Oldroyd-B: the evolution of $ \mathbf{C} $

How does $ \mathbf{C} $ evolve? The simplest constitutive model that captures
polymer memory is **Oldroyd-B** — the kinetic-theory model of non-interacting Hookean
(linearly elastic, infinitely extensible) dumbbells. Its conformation evolves as

$$
\overset{\triangledown}{\mathbf{C}} \;=\; -\frac{1}{\lambda}\big(\mathbf{C} - \mathbf{I}\big).
$$

The right-hand side is **relaxation**: left alone, $ \mathbf{C} $ decays exponentially
back to equilibrium $ \mathbf{I} $ over time $ \lambda $ (gale validates exactly this
— a no-flow field relaxing as $ \mathbf{C}(t) = \mathbf{I} + (\mathbf{C}_0 -
\mathbf{I})e^{-t/\lambda} $). The left-hand side is where the physics — and the subtlety
— lives.

### Why a plain time derivative will not do

The symbol $ \overset{\triangledown}{(\cdot)} $ is the **upper-convected derivative**,
and using it rather than the ordinary derivative $ \partial_t \mathbf{C} $ is not a
stylistic choice — it is forced by physics. A constitutive law must be **objective**
(frame-indifferent): the stress a material develops cannot depend on how the observer is
translating or, especially, *rotating*. The material derivative $ \partial_t \mathbf{C}
+ (\mathbf{u}\cdot\nabla)\mathbf{C} $ follows a fluid element as it is carried by the
flow — but it does *not* account for the element being **stretched and rotated** by the
local velocity gradient. A tensor like $ \mathbf{C} $, which encodes oriented
microstructure, is reoriented by the flow even when nothing physical changes in the
material frame. The plain derivative would spuriously attribute that rotation to a change
of state.

The upper-convected derivative fixes this by adding the stretching/rotation terms:

$$
\overset{\triangledown}{\mathbf{C}} \;=\; \underbrace{\partial_t\mathbf{C} + (\mathbf{u}\cdot\nabla)\mathbf{C}}_{\text{transport with the element}} \;-\; \underbrace{(\nabla\mathbf{u})\,\mathbf{C} \;-\; \mathbf{C}\,(\nabla\mathbf{u})^{T}}_{\text{stretching/rotation by the flow}}.
$$

Here $ (\nabla\mathbf{u})_{ij} = \partial u_i/\partial x_j $ — gale calls this $
\mathbf{L} = \nabla\mathbf{u} $. This index ordering is what makes the stretching term
come out as $ \mathbf{L}\mathbf{C} + \mathbf{C}\mathbf{L}^{T} $ rather than its transpose;
the answer is convention-dependent, so a reader cross-checking another textbook should
confirm the ordering before comparing signs. The combination is the unique objective rate
that correctly transports a contravariant tensor with the deforming, rotating fluid
element.

And the word "objective" carries a precise one-line test: put two observers in relatively
rotating frames, hand them the same physical material, and they must compute the same
stress. The plain material derivative fails this — an observer spinning in his chair sees
$ \mathbf{C} $ rotating and would conclude the polymer is being deformed when it is doing
nothing of the sort. The upper-convected derivative is constructed to subtract off exactly
the part of $ \dot{\mathbf{C}} $ that is "just the frame turning," so what is left is real
deformation both observers agree on. That is why it is forced by physics, not chosen for
elegance.

Writing it out, the Oldroyd-B equation gale actually integrates is

$$
\partial_t\mathbf{C} + (\mathbf{u}\cdot\nabla)\mathbf{C} \;=\; \mathbf{L}\,\mathbf{C} + \mathbf{C}\,\mathbf{L}^{T} \;-\; \frac{1}{\lambda}\big(\mathbf{C} - \mathbf{I}\big).
$$

Three physically distinct pieces: **advection** $ (\mathbf{u}\cdot\nabla)\mathbf{C} $
carries the microstructure downstream; **stretching** $ \mathbf{L}\mathbf{C} +
\mathbf{C}\mathbf{L}^{T} $ (symmetric, so it preserves the symmetry of $ \mathbf{C} $)
deforms the coils; **relaxation** pulls them back to equilibrium. The balance between
stretching and relaxation is exactly the Weissenberg number from Chapter 2: $ Wi =
\lambda\dot\gamma $, the ratio of relaxation time to flow time. At high $ Wi $ the flow
stretches faster than the polymer can relax, and $ \mathbf{C} $ grows large — which is
precisely where the trouble starts.

A useful sanity check, which gale validates: in **steady simple shear** $ \mathbf{u} =
(\dot\gamma y, 0) $, the analytic steady state is

$$
C_{xx} = 1 + 2\,Wi^{2}, \qquad C_{xy} = Wi, \qquad C_{yy} = 1.
$$

Note the $ Wi^2 $ growth of $ C_{xx} $: at $ Wi = 10 $ the streamwise stretch is
$ C_{xx} = 201 $. The conformation tensor's largest eigenvalue blows up *quadratically*
in the Weissenberg number. Remember that number.

### How it couples back to momentum

The polymer feeds back into the flow exactly as Chapter 6 set up to receive it: as a body
force in the momentum predictor. The incompressible momentum balance carries the
**solvent** viscosity $ \eta_s $ in its viscous term, and the polymer enters *only*
through the divergence of its stress:

$$
\rho\big(\partial_t\mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}\big) = -\nabla p + \eta_s\nabla^2\mathbf{u} + \nabla\cdot\boldsymbol{\tau}_p + \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0.
$$

gale computes $ \nabla\cdot\boldsymbol{\tau}_p $ by nodal collocation and hands it to
the dual-splitting velocity solve as a nodal body force. There is a clean validation that
the coupling is wired correctly: drive a planar channel with a body force $ G $. At
steady state the velocity must be the parabola set by the **total** zero-shear viscosity
$ \eta_0 = \eta_s + \eta_p $,

$$
U(y) = \frac{G}{2\eta_0}\,y(1-y),
$$

even though the momentum solve only ever sees $ \eta_s $ explicitly. The missing
viscosity $ \eta_p $ re-enters entirely through $ \nabla\cdot\boldsymbol{\tau}_p $.
Recovering $ \eta_0 $ and *not* $ \eta_s $ is the proof the polymer stress is feeding
back correctly. (gale recovers $ \eta_0 $ to $ \sim 10^{-3} $, away from a known
stress singularity at the inlet-wall corners.)

## The High-Weissenberg-Number Problem

Now the centerpiece. Everything above is classical and, on paper, benign. Yet for two
decades viscoelastic flow solvers could not push past a modest Weissenberg number — a few
units — before crashing. Not converging slowly: *crashing*, producing `NaN`. This is the
**High-Weissenberg-Number Problem (HWNP)**, and understanding it is the key to
understanding why gale is built the way it is.

The mechanism is a collision between three facts:

- **$ \mathbf{C} $ grows exponentially.** Recall $ C_{xx} = 1 + 2\,Wi^2 $ in steady
  shear; in extensional or mixed flows the largest eigenvalue can grow *exponentially* in
  time before relaxation catches it. Near geometric features — a corner, the stagnation
  point in front of an immersed particle — the field develops thin, steep **stress
  boundary layers** where $ \mathbf{C} $ varies by orders of magnitude across a fraction
  of an element.

- **High-order interpolation overshoots steep gradients.** This is the dark side of the
  spectral accuracy we celebrated in Chapter 3. A high-degree polynomial fit to a sharp,
  nearly-exponential profile does not stay between its sample values — it **overshoots and
  undershoots** (the Gibbs phenomenon). The higher the polynomial order, the more violent
  the oscillation near a steep front. Where the true $ \mathbf{C} $ climbs steeply but
  stays positive, the polynomial *interpolant* dips below — and produces locally **smaller
  eigenvalues than the true field, including negative ones**. Be precise about why higher
  order hurts, since a skeptical reader will rightly object that higher order *converges
  faster*: both are true. The overshoot's *amplitude* near an under-resolved jump does not
  vanish with order — it is roughly the fixed Gibbs constant — but it gets squeezed into a
  narrower layer that rings at higher frequency, so at fixed (insufficient) resolution,
  raising $ p $ buys a sharper, more oscillatory interpolant that punches *below zero*
  more readily, not less. The honest framing: high order is a liability **only when the
  stress layer is under-resolved**; resolve it and the overshoot disappears. The HWNP bites
  because at high $ Wi $ the layer thins faster than you can afford to resolve it — so you
  are perpetually in the under-resolved regime.

- **$ \mathbf{C} $ must stay SPD or the model is undefined.** The instant an
  eigenvalue of the *discrete* $ \mathbf{C} $ crosses zero, $ \mathbf{C} $ leaves the
  SPD cone. The polymer stress $ \boldsymbol{\tau}_p = \frac{\eta_p}{\lambda}(\mathbf{C} -
  \mathbf{I}) $ becomes nonsensical, the upper-convected stretching term $ \mathbf{L}
  \mathbf{C} + \mathbf{C}\mathbf{L}^{T} $ — which acts as a *positive feedback* on large
  eigenvalues — pumps the now-wrong state even harder, and within a few steps the solution
  diverges.

Put together: at high $ Wi $, a high-order discretization of a steeply-varying,
near-singular $ \mathbf{C} $ overshoots, drives $ \mathbf{C} $ out of the SPD cone,
and the model's own feedback turns that small numerical excursion into a blow-up.

The single most important thing to understand about the HWNP is this: **it is a numerical
failure, not a physical one.** And the reason is precise: in the continuous equations $
\mathbf{C} $ is governed by a transport equation whose evolution operator maps the SPD
cone into itself — the stretching term $ \mathbf{L}\mathbf{C} + \mathbf{C}\mathbf{L}^{T}
$ is a congruence-like action that preserves positive-definiteness, and relaxation only
pulls toward $ \mathbf{I} $, which is interior to the cone. So an SPD initial condition
stays SPD *exactly*, for all time — that is a theorem about the continuous equations. The
discretization breaks this for a mundane reason: a degree-$ p $ polynomial interpolant
is a *projection*, and projection onto a polynomial space is **not** cone-preserving —
nothing in the $ L^2 $ projection knows the target must have positive eigenvalues. That
is the whole disease in one sentence: *the continuous flow respects the cone; the
projection step does not.* The discretization is free to produce non-SPD garbage, and at
high $ Wi $ it does. And note the bitter irony for a spectral-element code: in the
under-resolved regime, **higher order makes it worse**, because higher-order interpolants
overshoot harder. The very property that makes DG-SEM attractive is what sharpens the
knife.

So the question is not "how do we make the polynomial overshoot less?" (filtering and
limiting help, but do not fix it). The question is: **how do we discretize an
SPD-constrained field so that no matter how badly the scheme overshoots, the recovered $
\mathbf{C} $ is SPD by construction?**

## The log-conformation cure (Fattal–Kupferman)

The answer, due to Fattal and Kupferman (2004), is a beautiful piece of
**structure-preserving discretization**. Instead of evolving $ \mathbf{C} $ directly,
evolve its **matrix logarithm**:

$$
\boldsymbol{\Psi} = \log \mathbf{C}, \qquad \mathbf{C} = \exp(\boldsymbol{\Psi}).
$$

The matrix exponential of *any* symmetric matrix is SPD — its eigenvalues are $
e^{\mu_i} > 0 $ for any real $ \mu_i $, whatever the eigenvalues $ \mu_i $ of $
\boldsymbol{\Psi} $. So if we carry $ \boldsymbol{\Psi} $ as the discrete unknown and
recover $ \mathbf{C} = \exp(\boldsymbol{\Psi}) $ only when we need the stress, then **$
\mathbf{C} $ is SPD by construction** — period. The polynomial interpolant of $
\boldsymbol{\Psi} $ may overshoot all it likes; the worst it can do is make $
\boldsymbol{\Psi} $ somewhat wrong, which makes $ \mathbf{C} $ somewhat wrong — but
never *non-SPD*. The map $ \boldsymbol{\Psi}\mapsto\exp(\boldsymbol{\Psi}) $ cannot
leave the SPD cone.

This is the philosophical heart of the chapter, and it echoes the book's throughline:
**rather than hope the numerical scheme respects a physical invariant, change variables so
the invariant is enforced exactly, automatically, by the representation itself.** It is
the same move as choosing a numerical flux that is provably dissipative (Chapter 4) or a
penalty form that is provably coercive (Chapter 5) — build the stability in, do not pray
for it. A second, quieter benefit: $ \boldsymbol{\Psi} = \log\mathbf{C} $ grows only
*logarithmically* where $ \mathbf{C} $ grows exponentially, so the field the polynomial
must resolve is far gentler — the overshoot is smaller to begin with.

### The transformed evolution

The price is that the evolution equation, transformed into $ \boldsymbol{\Psi} $, is no
longer a simple matrix product. The upper-convected stretching term, written for $
\boldsymbol{\Psi} $, splits the velocity gradient into parts that act differently on the
log. The Fattal–Kupferman result is

$$
\partial_t\boldsymbol{\Psi} + (\mathbf{u}\cdot\nabla)\boldsymbol{\Psi} \;=\; \boldsymbol{\Omega}\boldsymbol{\Psi} - \boldsymbol{\Psi}\boldsymbol{\Omega} \;+\; 2\mathbf{B} \;+\; \frac{1}{\lambda}\big(e^{-\boldsymbol{\Psi}} - \mathbf{I}\big).
$$

The relaxation source here looks like it has flipped sign relative to the direct form $
-\tfrac{1}{\lambda}(\mathbf{C}-\mathbf{I}) $, but it has not: whenever $
\boldsymbol{\Psi} \succ 0 $ (stretched), $ e^{-\boldsymbol{\Psi}} \prec \mathbf{I} $ so
the source is negative — the log analogue of pulling $ \mathbf{C} $ back toward $
\mathbf{I} $, exactly as relaxation should. The convention is deliberate.

To define $ \boldsymbol{\Omega} $ and $ \mathbf{B} $ we diagonalize $
\boldsymbol{\Psi} $ (equivalently $ \mathbf{C} $, since they share eigenvectors):
$ \boldsymbol{\Psi} = \mathbf{R}\,\mathrm{diag}(\mu)\,\mathbf{R}^{T} $, and transform the
velocity gradient into that eigenframe, $ \mathbf{M} = \mathbf{R}^{T}\mathbf{L}\mathbf{R}
$. Note that $ \mathbf{M} $ is *not* symmetric — that is the whole point: we split it
into its diagonal-in-eigenframe part ($ \mathbf{B} $, pure stretch) and the rest (which
drives $ \omega $). If you assume $ \mathbf{M} $ symmetric, then $ M_{12} = M_{21}
$ and the rotation formula below looks as if it could be simplified — it cannot, and the
asymmetry is physical, the local vorticity entering. Then:

- $ \mathbf{B} = \mathbf{R}\,\mathrm{diag}(M_{11}, M_{22}, \dots)\,\mathbf{R}^{T} $ is the
  **pure-extensional** part — the diagonal of $ \mathbf{M} $ in the eigenframe. It
  represents stretching *along* the eigendirections of the coils, which in log-space adds
  linearly: stretching the coil by a factor multiplies an eigenvalue of $ \mathbf{C} $,
  i.e. *adds* to an eigenvalue of $ \boldsymbol{\Psi} $. That is why it appears as a
  clean additive source $ 2\mathbf{B} $.

- $ \boldsymbol{\Omega} $ is the **rotation** that spins the eigenframe. The off-diagonal
  velocity-gradient components in the eigenframe try to rotate the coils' principal axes;
  this cannot change eigenvalues (it does not stretch), only reorient, so it enters as the
  commutator $ \boldsymbol{\Omega}\boldsymbol{\Psi} - \boldsymbol{\Psi}\boldsymbol{\Omega}
  $ — an infinitesimal rotation of $ \boldsymbol{\Psi} $. In 2D the rotation is a single
  scalar rate $ \omega = (M_{12}\lambda_2 + M_{21}\lambda_1)/(\lambda_2 - \lambda_1) $,
  with $ \lambda_i = e^{\mu_i} $ the eigenvalues of $ \mathbf{C} $; in 3D it is a full
  antisymmetric tensor with three such rates. The weighting is by the $ \mathbf{C}
  $-eigenvalues $ \lambda_i = e^{\mu_i} $, *not* the $ \boldsymbol{\Psi} $-eigenvalues
  $ \mu_i $ — a subtle point that is easy to typo in a kernel, and the same $ \lambda_2 -
  \lambda_1 $ denominator is where the isotropic-point singularity below comes from.

- The relaxation becomes $ \frac{1}{\lambda}(e^{-\boldsymbol{\Psi}} - \mathbf{I}) $ — the
  matrix exponential of $ -\boldsymbol{\Psi} $, again applied in the eigenframe.

The decomposition into a symmetric-extensional $ \mathbf{B} $ and an antisymmetric
rotation $ \boldsymbol{\Omega} $ is exactly the trick that keeps the log formulation
well-defined: stretching adds to log-eigenvalues, rotation reorients, and the two never
get confused.

Notice what *every* term now requires: a matrix function — $ \log $, $ \exp $, $
e^{-\boldsymbol{\Psi}} $ — and the eigenframe transform $ \mathbf{M} =
\mathbf{R}^{T}\mathbf{L}\mathbf{R} $. You cannot apply $ \log $ or $ \exp $ to a
matrix without diagonalizing it. **The eigendecomposition is unavoidable.** It is the
price of guaranteed positivity — and as we will see, gale pays it on the GPU, per node,
every timestep.

### The isotropic-point subtlety

One trap deserves mention because gale handles it explicitly. When $ \mathbf{C} $ is
near isotropic — which includes the most common starting condition, $ \mathbf{C} =
\mathbf{I} $ at rest — its eigenvalues coincide and the **eigenframe $ \mathbf{R} $ is
indeterminate**. Worse, the rotation rate $ \omega = (M_{12}\lambda_2 +
M_{21}\lambda_1)/(\lambda_2 - \lambda_1) $ has $ \lambda_2 - \lambda_1 \to 0 $ in the
denominator: it is singular. A naive guard that just sets $ \omega = 0 $ there drops the
extensional driving along with it, and the flow gets **stuck at equilibrium** — $
\dot{\boldsymbol{\Psi}} = 0 $, the coils never start deforming, the simulation does
nothing.

gale's fix (in both 2D and 3D): near-degenerate eigenvalues, **align the eigenframe with
the rate-of-strain tensor** $ \mathbf{D} = \tfrac{1}{2}(\mathbf{L} + \mathbf{L}^{T}) $
instead of with the indeterminate eigenframe of $ \boldsymbol{\Psi} $. In that frame the
extensional term reduces cleanly to $ 2\mathbf{B} \to \mathbf{L} + \mathbf{L}^{T} $,
recovering the correct small-deformation limit $ \dot{\boldsymbol{\Psi}} \approx
2\mathbf{D} - \tfrac{1}{\lambda}\boldsymbol{\Psi} $, and $ \omega \to 0 $ harmlessly.
With this, startup from rest works and the eigenvalues separate correctly under shear.

## The eigendecomposition machinery

A per-node eigendecomposition of a symmetric matrix, at every node, every timestep, is
the computational heart of the log-conformation method — and the differences between 2D
and 3D are instructive.

In 2D the eigendecomposition of a symmetric $ 2\times2 $ matrix is closed-form, a
direct algebraic expression. In 3D there is no usable closed form, so gale runs an
**iterative eigensolver per-node, on the GPU** — a genuinely hard kernel, arguably the
hardest in the whole crate. This iterative solve, every node and every step, is the price
of guaranteed positivity.

The 3D algorithm is the **cyclic Jacobi** method: repeatedly apply Givens rotations that
zero out the largest off-diagonal entry, accumulating the rotations into the eigenvector
matrix $ \mathbf{V} $, until the matrix is diagonal to working precision. gale uses a
**fixed 12-sweep** schedule (each sweep zeroing the three off-diagonal pairs $ (0,1),
(0,2), (1,2) $ in turn). The fixed sweep count is deliberate on two counts: it is
**branch-light** (no data-dependent loop length to make threads in a warp diverge), and
Jacobi is **robust at degenerate eigenvalues** — exactly the isotropic-point case that
would wreck a more delicate algorithm. (A handful of cuda-oxide intrinsic gaps force
equivalent-but-different device formulas for the eigensolve and the matrix exponential;
Chapter 12 tells that story.)

Frame it this way: the price of guaranteed positivity is a per-node symmetric eigensolve,
and gale pays that price on-device, in 2D and 3D, validated bit-for-bit against the CPU
oracle ($ \sim 10^{-9} $, limited by the libdevice `exp`).

## Coupling order: one tightly-coupled step

How do the velocity solve and the conformation update interleave within a single
timestep? gale uses the **decoupled (segregated) split** that the dual-splitting scheme
naturally affords (Chapter 6) — the same partitioned strategy recommended by gale's
solver-strategy research pass for getting started at high $ Wi $:

1. **Momentum first.** Solve for the new velocity using $ \nabla\cdot\boldsymbol{\tau}_p
   $ computed from the **old** conformation, fed in as a body force to the dual-splitting
   predictor → pressure-Poisson → projection (Chapter 6, Chapter 7 for the solves).
2. **Conformation second.** Advance $ \boldsymbol{\Psi} $ (or $ \mathbf{C} $ for the
   direct form) by one step using the **new** velocity, via **SSP-RK3** — a
   strong-stability-preserving Runge–Kutta scheme (Chapter 4) for the explicit
   constitutive transport. At low Reynolds number the advective CFL is relaxed, so an
   explicit conformation update is appropriate; the stiff parts (incompressibility, viscous
   diffusion) are what the implicit dual-splitting already handles.

One step, tightly coupled: old stress drives velocity, new velocity drives conformation.
In gale this is owned by a single integrator, `ViscoelasticDualSplitting`, which reads the
velocity and conformation fields, runs the validated `ViscoelasticFlow::step` against the
current mesh, and writes both fields back. It is the third integrator family in the
framework (Chapter 12), and it owns the coupling explicitly precisely because the split
order does not fit the generic `updaters → integrator → writers` schedule — a faithful
mapping of the validated operator rather than a forced fit.

This first-order segregated splitting is the right place to start, and it is what gale has
built and validated. A more strongly-coupled or higher-order scheme (Picard/Newton
iteration on the flow↔stress coupling) is the documented escalation path *if* the coupling
fails to converge at very high $ Wi $ — a known frontier, not a solved problem.

## A natural next model: FENE-P

Oldroyd-B has one glaring unphysical feature: its dumbbells are **infinitely
extensible** (Hookean springs with no limit). Real polymers have finite contour length —
they cannot stretch forever. This is why Oldroyd-B's $ C_{xx} = 1 + 2\,Wi^2 $ grows
without bound and why purely-extensional flows can drive its stress to infinity: in a
*steady* extensional flow Oldroyd-B does not merely grow large, it has **no steady state at
all** above a critical extension rate ($ \lambda\dot\varepsilon = 1/2 $) — the
coil-stretch resonance where stretching outruns relaxation outright. That is the sharpest
statement of "infinitely extensible is unphysical," and the precise failure a finite
extensibility exists to cure.

The natural next constitutive model is **FENE-P** (Finitely-Extensible Nonlinear
Elastic, Peterlin closure). It replaces the Hookean spring with one that stiffens as the
coil approaches a maximum extensibility $ L^2 $, bounding the stretch. The relaxation
term picks up a nonlinear factor (the Peterlin function) that diverges as $
\mathrm{tr}\,\mathbf{C} \to L^2 $, pulling the coils back ever harder near full
extension. FENE-P is more physical, captures shear-thinning that Oldroyd-B misses, and is
the workhorse for many real polymer solutions.

Two points for gale. First: FENE-P is **planned, not built** — Oldroyd-B (direct and
log-conformation, 2D and 3D, CPU and GPU) is what exists and is validated today. Second,
and reassuringly: the **same log-conformation idea applies directly**. FENE-P's $
\mathbf{C} $ is also SPD-constrained and suffers the same HWNP, and the same $
\boldsymbol{\Psi} = \log\mathbf{C} $ reformulation cures it — only the algebraic
relaxation term changes. The eigendecomposition machinery built for Oldroyd-B is exactly
what FENE-P would reuse. The hard infrastructure is already in place.

## How gale does it

Mapping the chapter onto the source:

- **`gale::dg::OldroydB`** (2D) and **`OldroydB3d`** — the direct Oldroyd-B form. State is
  the conformation itself: `[Vec<f64>; 3]` = $ [C_{xx}, C_{xy}, C_{yy}] $ in 2D, `[Vec<f64>;
  6]` in 3D. `conformation_rhs` assembles advection + upper-convected stretching +
  relaxation by nodal collocation; `step_ssp_rk3` time-steps it. Validated against steady
  simple shear ($ C_{xx}=1+2Wi^2 $) and exponential stress relaxation. Adequate at
  low–moderate $ Wi $.

- **`gale::dg::LogConfOldroydB`** (2D) and **`LogConfOldroydB3d`** — the log-conformation
  form. State is $ \boldsymbol{\Psi} = \log\mathbf{C} $; equilibrium is $
  \boldsymbol{\Psi} = 0 $. `psi_rhs` implements the full Fattal–Kupferman decomposition
  (eigenframe, $ \mathbf{B} $, $ \boldsymbol{\Omega} $, $ e^{-\boldsymbol{\Psi}} $,
  isotropic-point fix); `conformation`/`from_conformation` are the $ \exp $/$ \log $
  round-trip; `recover_c` hands an SPD $ \mathbf{C} $ back for the stress. The 3D version
  uses `sym_eig3`, the 12-sweep Jacobi solver. Validated against the round-trip $
  \exp(\log\mathbf{C}) = \mathbf{C} $, and — the decisive test — **$ Wi = 10 $ steady
  shear** reaching $ C_{xx}=201 $ with $ \mathbf{C} $ verified SPD ($ \det > 0 $) at
  *every* step, the regime where the direct form is fragile.

- **GPU kernels `logconf_psi_rhs`** (2D) and **`logconf3d_psi_rhs`** (3D) in `gale-gpu` —
  one block per element, one thread per node, the velocity gradient and $
  \nabla\boldsymbol{\Psi} $ by sum factorization, then the per-node eigenframe algebra
  (the `atan2`-avoiding direct eigenvector in 2D, the flat-scalar Jacobi `sym_eig3` in 3D).
  Both validated **bit-for-bit against the CPU oracle**.

- **`gale::sim::ViscoelasticDualSplitting`** — the coupled integrator described above,
  generic over the constitutive model.

And the headline: the full **immersed-particle-in-viscoelastic-flow capstone runs
end-to-end on the GPU, in both 2D and 3D** — exercising the elliptic solvers (Chapter 7),
the dual-splitting flow (Chapter 6), the log-conformation polymer model of this chapter,
and the immersed-boundary penalization (Chapter 9) all at once.

What remains honestly open is the coupled-system robustness at *very* high $ Wi $: the
log-conformation form guarantees $ \mathbf{C} $ stays SPD, but it does **not** by itself
resolve localized non-convergence near geometric stress singularities, and the
preconditioner behavior of the pressure solve as elastic stress grows is genuinely
uncharted territory (Chapter 13 keeps that score). What gale *does* guarantee — positivity
of the conformation, by construction, on the GPU, in 2D and 3D — is the foundation on
which everything past a Weissenberg number of a few units must be built. That is the
HWNP's lesson, and the reason this chapter is the centerpiece of the stability
throughline: **you do not stabilize a constraint by being careful; you stabilize it by
making it impossible to violate.**
