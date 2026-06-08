# Elliptic Operators: The Interior-Penalty Method

[Chapter 4](04-hyperbolic.md) handled transport — the part of fluid motion where
information races along characteristics and stability is won by *adding dissipation* in
the right places. This chapter is about the opposite kind of operator and the opposite
kind of danger. **Elliptic** operators — diffusion-like, built around the Laplacian
$ -\nabla^2 $ — have no characteristics and no preferred direction; a disturbance
anywhere is felt *everywhere* instantly. They are the slow, global, expensive part of a
fluid solver, and getting them stable isn't about *draining* the right energy (as in
Chapter 4) — it's about making sure the discrete operator is positive-definite *to begin
with*. Get that wrong and there is no solution to find, stable or otherwise. This chapter
is the story of one parameter that decides whether that is true.

## Why elliptic operators are the hot path

It is tempting to think of diffusion as the easy, friendly term — it smooths things, it
cannot blow up the way advection can. That intuition is right about the physics and wrong
about the cost. In gale's incompressible solver ([Chapter 6](06-incompressible.md)) the
dual-splitting (projection) time scheme turns *every single time step* into **two
elliptic solves**:

- a **pressure-Poisson** solve, $ -\nabla^2 p = \text{(something)} $, which enforces
  incompressibility — this is the dominant cost of the whole simulation; and
- a **viscous Helmholtz** solve, $ \big(\tfrac{\gamma_0}{\nu\Delta t}\,I - \nabla^2\big)\mathbf{u} = \text{(something)} $,
  which advances the velocity implicitly so the viscous term does not impose a punishing
  explicit time-step limit.

Both are the *same operator* — a SIPG Laplacian, optionally with a reaction term — applied
inside a Krylov iteration that calls it dozens of times per solve. So the SIPG Laplacian
is gale's **most-reused and most performance-critical kernel**, which is exactly why the
GPU build order ([implicit-solver-strategy.md](../../docs/implicit-solver-strategy.md) §7)
puts "scalar elliptic GPU solve" *first*, before any flow at all. Get this operator right
and fast, and the flow solver is mostly plumbing on top of it.

## The difficulty: a second derivative of a discontinuous function

Here is the snag. DG solutions are **discontinuous** across element faces — that is the
"D" in DG, and in Chapter 4 it was a feature, because the numerical flux gracefully
handled the double-valued solution. But a second-order operator needs *more* continuity
than a first-order one. The weak form of $ -\nabla^2 u $ involves the **gradient** of
the solution, and now we have two problems at every face at once: the solution $ u $ is
double-valued (a jump $ [u] $), *and* its normal gradient $ \nabla u\cdot\mathbf{n} $
is double-valued. If you naively integrate by parts element-by-element and do nothing
special at the faces, the resulting discrete operator is **inconsistent and not even
invertible** — it is genuinely ill-posed. You cannot just average and hope; a second-order
operator on a discontinuous space has to be *built* to make sense of the missing
continuity, and there is a right way and several wrong ways.

## The Symmetric Interior Penalty method

The **Symmetric Interior Penalty Galerkin (SIPG)** method is the right way gale uses. It
adds three face contributions to the volume term, and the trio is best understood by what
each one *does*. Writing $ \{\cdot\} $ for the average across a face and $ [\cdot] $
for the jump, the bilinear form gains, on each interior face:

$$
-\oint \{\nabla u\cdot\mathbf{n}\}\,[v]
\;-\;\oint \{\nabla v\cdot\mathbf{n}\}\,[u]
\;+\;\oint \tau\,[u]\,[v].
$$

**The consistency term** $ -\oint\{\nabla u\cdot\mathbf{n}\}[v] $. This is the honest
one: it is the face flux of the gradient, and it falls straight out of integrating
$ -\nabla^2 u $ by parts. Using the *average* gradient $ \{\nabla u\cdot\mathbf{n}\} $
is the natural single-valued choice for the double-valued normal derivative. By itself it
makes the method *consistent* — the true smooth solution satisfies the discrete equations
— but it does **not** make it stable or symmetric.

**The symmetry / adjoint term** $ -\oint\{\nabla v\cdot\mathbf{n}\}[u] $. This is the
clever one and the reason "Symmetric" is in the name. It is the consistency term with the
roles of the trial function $ u $ and the test function $ v $ swapped. Adding it makes
the bilinear form **symmetric** in $ u $ and $ v $. Why insist on that? Because a
symmetric bilinear form yields a **symmetric matrix**, and symmetry is not cosmetic — it
is the property that unlocks the conjugate gradient solver and gives the method clean
*adjoint consistency* (the discrete adjoint problem is also consistent, which is what
delivers the optimal high-order convergence rate). That accuracy payoff is concrete: the
symmetric form converges in the $ L^2 $ norm at the full rate $ h^{p+1} $, whereas
dropping the adjoint term (the non-symmetric NIPG/IIPG variants) costs a full power —
$ L^2 $ error degrades to $ h^{p} $, and on even-degree elements can be worse. The
reason is a duality (Aubin–Nitsche) argument: the $ L^2 $ estimate borrows accuracy
from the adjoint problem, and only the symmetric form makes the discrete adjoint
consistent. So the symmetry term is not just "free symmetry for CG" — it buys a full order
of accuracy in the norm that matters, and that is the historically correct reason SIPG
won. It is consistent to add because for the
true solution $ [u]=0 $, so this term vanishes on the exact solution and does not spoil
consistency. We pay a price for it, though: with the symmetry term added, the consistency
term alone is no longer enough to keep the operator positive. That is what the third term
fixes.

**The penalty term** $ +\oint \tau\,[u][v] $. This is the stability term. It penalizes
the jump in the solution directly: the larger the discontinuity across a face, the larger
the energy cost. Physically, it is a spring pulling the two sides of every face toward
agreement — it does not *enforce* continuity (this is still DG, the solution stays
discontinuous), it *weakly* discourages discontinuity with a stiffness $ \tau $. With
$ \tau $ large enough, the penalty term dominates the indefinite contribution the
symmetry term introduced, and the whole operator becomes **coercive** (positive-definite).
With $ \tau $ too small, it does not, and the operator is indefinite and the solver
fails.

How big is that indefinite contribution, exactly? The one-line coercivity sketch answers
it. Bound the two cross terms with Young's inequality,
$ 2\{\nabla u\cdot\mathbf{n}\}[u] \le \epsilon\,h\,\{\nabla u\cdot\mathbf{n}\}^2 + \tfrac{1}{\epsilon h}[u]^2 $:
the first piece is reabsorbed into the volume gradient energy *using the inverse-trace
inequality* (which bounds the face-gradient by the volume energy with a constant
$ \sim (p+1)^2/h $), and the leftover $ \tfrac{1}{\epsilon h}[u]^2 $ is precisely
what the penalty $ \tau[u]^2 $ must beat. So the indefinite part the penalty fights
*is* the inverse-trace constant — which is why the very same $ (p+1)^2/h $ reappears in
the threshold below. Which brings us to the one number that matters.

## The penalty parameter: the stability knob

gale uses

$$
\boxed{\;\tau = \alpha\,\frac{(p+1)^2}{h}\;}
$$

with $ \alpha $ a user scale (typically around 5), $ p $ the polynomial degree, and
$ h $ a representative element size. Every piece of that formula is forced on us by
stability, and it is worth understanding *why* it looks exactly like this — because if you
get it wrong in either direction, the method breaks, just in opposite ways.

**Too small ⇒ loss of coercivity.** If $ \tau $ is below the threshold, the penalty no
longer beats the indefinite part contributed by the symmetry term. The discrete operator
stops being positive-definite — it becomes **indefinite**. Now $ u^\top A u $ can be
negative for some $ u $, the energy estimate that proves stability collapses, and a
conjugate gradient solver (which *assumes* positive-definiteness) breaks down: it divides
by a non-positive $ p^\top A p $ and produces garbage or diverges. This is not a slow
accuracy degradation; it is a hard failure.

**Too large ⇒ ill-conditioning.** If $ \tau $ is far above what is needed, the operator
is still SPD, but the penalty term now dominates everything and the **condition number**
of $ A $ grows linearly in $ \tau $. A Krylov solver's iteration count grows with the
square root of the condition number, so the solve gets slow — and slow matters when this
is the per-step bottleneck. It is worth being honest that the SIPG operator is already
ill-conditioned at the *minimal* $ \tau $ — its condition number is $ O(p^4/h^2) $
from the discretization itself — so $ \tau $-inflation worsens a bad situation rather
than creating one from nothing; this is one reason the raw operator needs preconditioning
at all ([Chapter 7](07-solvers.md)'s p-multigrid). Worse still, an enormous penalty
over-stiffens the weak continuity and can actually *degrade accuracy*. So $ \tau $ must
be large enough to be stable and no larger.

**Why $ (p+1)^2/h $, specifically.** The threshold is set by a sharp inequality from
finite-element analysis — the **inverse trace inequality** — which bounds the size of a
polynomial's normal gradient *on the face* by its size *in the volume*. For a degree-$ p $
polynomial on an element of size $ h $, that boundary-gradient term scales like
$ (p+1)^2/h $: it grows quadratically with the polynomial order and inversely with
element size. The $ p^2 $ is the inverse-trace constant for the Markov-brothers-type
bound on the reference element; as *intuition* (not the derivation), think of higher-order
polynomials as having steeper gradients packed near the element edges. The $ 1/h $ is
the geometric scaling of the face measure relative to the volume under the
reference-to-physical map — a mapping/scaling factor, not a claim that the solution's
gradient is physically sharper on small elements (it need not be). The penalty has to
*beat* this term to guarantee coercivity, so it must scale the same way — hence
$ \tau \propto (p+1)^2/h $.
This is the reason the parameter is not a free fudge factor but a precisely shaped
quantity: it is the minimum stiffness that dominates the worst the geometry and polynomial
order can throw at it. In gale's `Poisson::penalty`, $ h $ on an interior face is taken
as the *minimum* of the two neighbours' sizes (the tighter constraint wins), and on a 2:1
non-conforming face it likewise uses the smaller element — the conservative choice that
keeps the operator coercive across hanging nodes.

## Symmetry → SPD → conjugate gradient

The payoff of all that care is a single, decisive property: the SIPG operator (for the
Poisson or Helmholtz problem with the BCs below) is **symmetric positive-definite (SPD)**.
gale checks both halves explicitly — `operator_is_symmetric` confirms
$ u^\top A v = v^\top A u $ to $ 10^{-9} $, and `spd_positive_on_nonzero` confirms
$ u^\top A u > 0 $ — and crucially `nonconforming_poisson_is_symmetric` confirms symmetry
*survives* the mortar coupling on a refined mesh, which is the decisive guard, because if
the mortar broke symmetry the solver would silently fail.

SPD is exactly the precondition for **conjugate gradient (CG)**, the Krylov method that is
both the cheapest and the fastest for symmetric positive-definite systems. This is why the
method's *symmetry* directly dictates the *solver choice*: the symmetry term we worked to
include earns us the right to use CG ([Chapter 7](07-solvers.md) builds the GPU CG, its
p-multigrid preconditioner, and the deflated variant). Had we chosen the non-symmetric
interior-penalty variant (NIPG), the matrix would be non-symmetric and we would be forced
into GMRES — more memory, more cost per iteration. The chain *symmetry → SPD → CG* is one
of the tightest method-to-solver couplings in the whole code, and it traces straight back
to that adjoint term.

## The diagonal mass matrix, and why it is a gift

Recall from Chapter 3 that GLL **collocation** — putting the basis on the same GLL nodes
used for quadrature — zeros every off-diagonal of the mass matrix $ M $, so
$ M = \mathrm{diag}(Jw) $, the Jacobian-weighted quadrature weights ("mass lumping").
(This is an *aliasing* of the mass integral, in the Chapter 4 sense, but here it is
entirely benign and in fact the point.) The elliptic operator exploits this twice.

Two gifts follow.

**$ M^{-1} $ is trivial.** Inverting a diagonal matrix is pointwise division. In the
hyperbolic solver of Chapter 4 the step $ \partial_t u = M^{-1}(\cdots) $ is just a
divide by $ Jw $; in the elliptic Krylov solves there is no global mass-matrix inversion
to fight at all. On a GPU, where a general sparse triangular solve is poison, this is the
difference between a method that maps cleanly to hardware and one that does not.

**The Helmholtz reaction term is pointwise.** The viscous solve needs the operator
$ \lambda M + A $, where $ \lambda = \gamma_0/(\nu\Delta t) $ is the reaction
coefficient. Because $ M $ is diagonal, the reaction term $ \lambda M u $ is simply
$ +\lambda\,(Jw)\cdot u $ **pointwise** — no matrix at all, just a per-node scale-and-add.
gale exploits exactly this: in `Poisson::apply`, after the SIPG stiffness is accumulated,
the reaction term is added as `self.reaction * jw[k] * u[...]`, and on the GPU
(`gale-gpu/src/operators/poisson.rs`) it is the single trailing term
`lambda * jw[b] * u[b]` in the `operator` kernel. The same kernel serves pure Poisson
($ \lambda=0 $, bit-identical to before) and the viscous Helmholtz solve
($ \lambda>0 $) — one operator, two jobs, because the mass matrix is diagonal.
`helmholtz_patch_test_exact_on_quadratic` and `helmholtz_converges_and_is_spd` verify the
reaction operator is exact on low-degree solutions and stays SPD at the
$ \lambda\sim 1/(\nu\Delta t) $ scale the viscous solve actually uses.

## Boundary conditions

SIPG handles boundaries by treating a Dirichlet boundary face as an interior face whose
"other side" is the prescribed data — the jump becomes $ [u] = u - g $ and the average
becomes one-sided. The same three terms apply, now lifting the boundary data $ g $ into
the residual; the consistency-, symmetry-, and penalty-style contributions all reappear on
the boundary face, which is why Dirichlet BCs in SIPG are sometimes called **weak** or
**Nitsche** conditions — the boundary value is imposed by penalty, not by pinning nodes.
gale's `Poisson::rhs` builds exactly this lifting, and the patch tests
(`patch_test_exact_on_quadratic`, `patch_test_exact_on_harmonic_cubic`) confirm the DG
solution reproduces low-degree exact solutions to round-off.

**Neumann** conditions are different and pleasingly easy: a prescribed normal flux is a
**natural** boundary condition. It enters the right-hand side as a surface integral
$ +\oint q\,v $ and contributes *nothing* to the operator — the boundary face is simply
skipped in the matrix action. In gale this is the `neumann_tags` set on the `Poisson`
operator: a tagged boundary face adds no consistency/penalty/symmetry contribution at all
(a "do-nothing" face); the homogeneous case is literally a `continue` in both the CPU
`apply` and the GPU `operator` kernel (the `NEU` sentinel).

This matters because the **pressure-Poisson solve uses all-Neumann boundaries**, and that
makes the operator **singular**: with no Dirichlet data anywhere, pressure is only defined
up to an additive constant — the operator has a one-dimensional **nullspace of constants**,
$ A\cdot\mathbf{1}=0 $. A plain CG cannot solve a singular system. The cure is
**deflation**: project the constant out of the residual every iteration so the iteration
lives on the complement of the nullspace. gale's `cg_deflated` (and its GPU twin
`pressure_cg_solve`) does exactly this — subtract the mean each step — and
`neumann_poisson_converges_up_to_a_constant` verifies it recovers $ \cos\pi x\cos\pi y $
up to the expected free constant.

Deflation is necessary but not sufficient: a singular system $ A p = b $ has a solution
only if $ b \perp \ker A $, i.e. the right-hand side must have zero mean (the discrete
compatibility, or Fredholm, condition — the discrete analogue of
$ \oint \mathbf{u}\cdot\mathbf{n} = 0 $). Subtracting the mean each step handles the
*nullspace* of $ A $; it does not by itself put the RHS in the range, so if the velocity
field fed in is not discretely divergence-compatible the residual never reaches zero.

The full deflation story belongs to
[Chapter 7](07-solvers.md); the point here is that the *boundary condition choice* of the
flow solver is what makes the operator singular in the first place.

## How gale does it

The elliptic machinery lives in `src/dg/operators/poisson.rs`, built around the `Poisson`
struct:

- **`Poisson::new(mesh, alpha)`** is the pure SIPG Laplacian; **`with_reaction(mesh, alpha, λ)`**
  gives the Helmholtz operator $ \lambda M + A $ for the viscous solve; **`with_bc(…, neumann_tags)`**
  marks boundary tags as Neumann (the pressure-Poisson uses all-Neumann).
- **`apply`** is the matrix-free operator action $ A u $: per-element gradients, the
  fused volume stiffness, the three SIPG face terms (interior, Dirichlet boundary, and 2:1
  non-conforming via the conservative mortar), and the optional pointwise reaction term.
  **`apply_volume`** isolates the element-local volume stiffness in the exact fused `pr/ps`
  form the GPU kernel uses, so `apply − apply_volume` is precisely the face contribution —
  a deliberate seam for validation.
- **`cg`**, **`cg_deflated`**, and the RHS builders (`rhs`, `rhs_mixed`) complete the CPU
  reference solver.

On the GPU (`gale-gpu/src/operators/poisson.rs`), the operator action is a deliberate
**2-kernel pipeline**: a `gradient` kernel computes the per-node physical gradient (one
block per element, staging the differentiation matrix and nodal values in shared memory),
then an `operator` kernel consumes those gradients to assemble the fused volume stiffness,
the symmetry-lift, the consistency/penalty face terms, and the $ \lambda M $ reaction —
the whole SIPG $ A u $ in one pass. Three host wrappers build on it: `poisson_apply`
(one application, validated against `Poisson::apply`), the device-resident `poisson_cg_solve`
/ `helmholtz_cg_solve`, the deflated `pressure_cg_solve`, and the p-multigrid-preconditioned
`poisson_pcg_solve`. Setup (meshes, transfer matrices, diagonals) is reused from the
validated CPU code; only the iterations run on-device. Every one of these is checked
**bit-for-bit** against its CPU oracle ([Chapter 1](01-introduction.md)'s "the CPU is the
oracle" principle), and because the device math is pure arithmetic with no transcendental
calls, the kernels are libdevice-free and run on the Titan V's sm_70.

With transport (Chapter 4) and diffusion (this chapter) both discretized and stable,
[Chapter 6](06-incompressible.md) assembles them into an incompressible Navier–Stokes
solver — where the pressure-Poisson and viscous-Helmholtz solves built here become the
engine that runs every step, and [Chapter 7](07-solvers.md) makes that engine fast on the
GPU.
