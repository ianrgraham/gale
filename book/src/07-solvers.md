# Linear Solvers on the GPU

Chapter 6 reduced incompressible flow to a sequence of **SPD elliptic solves** — a
pressure-Poisson and a viscous-Helmholtz problem every timestep. Those solves are the
*bottleneck* of high-order incompressible flow: profiling and the literature both put the
overwhelming majority of the runtime there. So if gale is going to be fast, this is the
part that has to be fast, and it has to run on the GPU. This chapter is about how.

The throughline of the chapter is again numerical: the methods here are chosen because
they (a) only need the operations the GPU does well, (b) stay robust as we crank up the
resolution and polynomial order, and (c) keep the data on the device so the inner loop
never waits on the host. We build up in four layers — matrix-free CG, then
preconditioning, then deflation for the singular pressure system, then the GPU residency
pattern that ties it together.

## Why matrix-free, iterative solvers

The first instinct from a linear-algebra course is to *assemble* the matrix \\( A \\) and
factorize it (a direct solve) or hand it to an algebraic multigrid library. For high-order
DG this is the wrong instinct, for a concrete and quantitative reason.

The SIPG operator (Chapter 5) couples every node in an element to every other node in that
element, plus the nodes across each face. On an element of polynomial degree \\( p \\) in
\\( d \\) dimensions there are \\( (p+1)^d \\) nodes, and the element block is dense, so the
assembled matrix has roughly \\( (p+1)^{2d} \\) nonzeros *per element*. In 3D at order 8
that is \\( 9^6 \approx 5.3\times10^5 \\) entries per element — the matrix balloons in both
memory and apply-cost as the order rises. The literature is blunt about this: assembling
the operator at high \\( p \\) is "prohibitively expensive." On a Titan V's 12 GB of memory,
storing such a matrix for a real mesh is simply a non-starter.

The way out is **matrix-free**: never form \\( A \\), only ever compute its *action* on a
vector, \\( v \mapsto A v \\). That action is exactly the "apply" kernel from Chapter 5 —
the volume stiffness, the symmetry lift, and the face penalty/consistency terms, evaluated
on the fly. Done with the tensor-product (sum-factorization) structure of the spectral
element, the apply costs only \\( O(p^{d+1}) \\) work per element instead of \\( O(p^{2d}) \\),
and it touches only \\( O(p^d) \\) memory. The catch is that, with only the action of
\\( A \\) available, you cannot do Gaussian elimination — you need a solver that asks for
*nothing but* matrix–vector products. That is precisely what **Krylov subspace methods**
provide, and it is why gale's elliptic solvers are all matrix-free iterative methods.

## Conjugate gradient: the right method for SPD systems

The workhorse is **conjugate gradient (CG)**, the canonical Krylov method for systems that
are **symmetric positive-definite** — which, by the design of Chapter 5, the SIPG Poisson
and Helmholtz operators are. (SIPG is built to be SPD on purpose; that is what makes CG
applicable, and it is why "symmetric" is in the name.)

The intuition is an energy minimization. Solving \\( A x = b \\) with \\( A \\) SPD is
equivalent to minimizing the quadratic energy

\\[
  \phi(x) = \tfrac{1}{2}\,x^\top A x - b^\top x,
\\]

whose unique minimum is exactly \\( A x = b \\). A naïve descent would slide downhill along
the residual \\( r = b - Ax \\), but successive steps interfere and zig-zag. CG's trick is to
choose search directions that are **conjugate** — mutually orthogonal in the inner product
defined by \\( A \\), i.e. \\( p_i^\top A p_j = 0 \\) for \\( i \neq j \\). Stepping along
conjugate directions means each step's progress is never undone by a later one, so in
exact arithmetic CG reaches the solution in at most \\( n \\) steps, and in practice
converges far sooner.

The reason CG is perfect for the GPU is what each iteration needs:

- one **matrix–vector product** \\( Ap \\) (our matrix-free apply),
- a couple of **dot products** (to form the step length \\( \alpha \\) and the conjugacy
  coefficient \\( \beta \\)),
- a couple of **axpy** updates \\( y \leftarrow y + \alpha x \\) (advance the solution and
  the residual).

No factorization, no matrix storage — just matvec, dot, and axpy, all of which are
embarrassingly parallel. The one thing to watch is **convergence speed**, which is
governed by the **condition number** \\( \kappa(A) \\): the number of iterations to a fixed
tolerance scales like \\( \sqrt{\kappa} \\) — though that is only the worst-case bound. CG's
real convergence depends on the *whole spectrum*, not just its extremes: clustered
eigenvalues converge superlinearly (CG "deflates" them as it goes) and a few stray large
ones cost only a handful of extra iterations. This is the key to what follows —
preconditioning helps precisely because it does not merely shrink \\( \kappa \\) but
*clusters* the spectrum near 1. For the SIPG operators \\( \kappa \\) grows with
resolution (smaller \\( h \\)) and with polynomial order \\( p \\). Unpreconditioned CG is
fine for modest problems — and gale validates it directly — but on a real mesh the
iteration count grows uncomfortably, which motivates the next layer.

## Preconditioning, and gale's p-multigrid

A **preconditioner** is an operator \\( M^{-1} \approx A^{-1} \\) that is cheap to apply and
makes the *preconditioned* system \\( M^{-1}A x = M^{-1} b \\) much better conditioned, so CG
converges in far fewer iterations. The art is finding an \\( M^{-1} \\) that is a good
approximate inverse without being as expensive as solving the original system.

gale uses **p-multigrid**, the nekRS-style preconditioner that is the recommended choice
in the solver-strategy document. To understand it, start with the observation that drives
all multigrid: a cheap **smoother** — here a **damped-Jacobi** sweep, \\( x \leftarrow x +
\omega\,D^{-1}(b - Ax) \\) with \\( D = \mathrm{diag}(A) \\) — is very good at killing the
*high-frequency* (oscillatory) components of the error, but very bad at the *low-frequency*
(smooth) components. A few Jacobi sweeps leave a smooth error that just sits there. The
damping factor \\( \omega \\) (the classic 1D optimum is \\( \omega \approx 2/3 \\)) is what
earns the word "damped": undamped Jacobi barely touches the highest-frequency mode — it is
an eigenvector with eigenvalue near \\( -1 \\), so it gets flipped in sign and hardly
shrunk — and the damping is exactly what turns Jacobi into the high-frequency smoother
multigrid needs.

The multigrid insight is that a smooth error on a fine discretization looks *oscillatory*
when viewed on a coarse one — so transfer the problem to a coarser level and let the
smoother kill it there cheaply, then transfer the correction back. "Coarse" in
**p**-multigrid means **lower polynomial degree**: gale coarsens \\( p \to p-1 \to \cdots \\)
down a chain of spectral-element levels on the *same* mesh, rather than coarsening the mesh
itself. This is natural for DG-SEM, where the nodal basis at each order is readily
interpolated to the next.

One full **V-cycle** — the shape that gives multigrid its name — does, recursively:

1. **pre-smooth** on the fine level (a few damped-Jacobi sweeps to remove high-frequency
   error);
2. compute the residual \\( r = b - Ax \\) and **restrict** it to the coarser (lower-order)
   level;
3. recurse — smooth-restrict down to the coarsest level, where the system is small enough
   to solve cheaply (gale does a short inner CG there);
4. **prolong** (interpolate) the coarse correction back up and add it to the fine
   solution;
5. **post-smooth** on the fine level.

Going *down* the V handles progressively lower-frequency error on progressively coarser
levels; coming back *up* carries the corrections home. The V-cycle is built symmetric
(matched pre- and post-smoothing) so it remains a valid SPD preconditioner for CG, which
relies on \\( M^{-1} \\) being SPD to preserve its conjugacy. One V-cycle is the
preconditioner: inside CG, each iteration applies one V-cycle as \\( M^{-1} \\). The result
is an iteration count that grows far more slowly with \\( p \\) and \\( h \\) than
unpreconditioned CG.

A caveat the strategy document is careful about: the *best* preconditioner is
mesh-dependent — cheap p-multigrid wins on regular meshes, but on highly distorted meshes
or very large problems an AMG-based coarse correction can overtake it, which is why
production solvers like nekRS ship a *poly-algorithmic* autotuner rather than one fixed
choice. gale's p-multigrid is the well-validated default, not a claim that it is optimal in
every regime.

## Deflation for the singular pressure system

The pressure-Poisson solve of Chapter 6 has a special difficulty: with velocity Dirichlet
on every wall, the pressure carries an **all-Neumann** boundary condition, and is therefore
defined only **up to an additive constant**. In operator terms, \\( A\,\mathbf{1} = 0 \\) —
the constant vector \\( \mathbf{1} \\) is in the **nullspace** of the pressure operator,
which is thus **singular**.

A singular SPD system \\( A x = b \\) with \\( A\mathbf{1} = 0 \\) has a solution *if and
only if* the right-hand side satisfies the **consistency (compatibility) condition**
\\( b \perp \mathbf{1} \\) — i.e. \\( b \\) has zero mean. This is the Fredholm alternative,
and physically it is the discrete statement that the net mass source must balance for an
all-Neumann pressure problem. In practice \\( b \\) carries a small spurious mean component
(from discretization and rounding), so the system handed to CG is slightly *inconsistent*:
there is no finite solution for that component, the iteration drifts along the nullspace,
and the residual stops dropping cleanly — sometimes failing the tolerance test entirely.
The fix is **deflation**, which does two jobs at once: it removes the \\( \mathbf{1} \\)-
component of the residual (enforcing the consistency condition so a solution exists) and it
pins the iterate to the unique zero-mean representative.

Since the nullspace is the constants, projecting it out just means **removing the mean**,

\\[
  v \;\leftarrow\; v - \frac{\mathbf{1}^\top v}{n}\,\mathbf{1},
\\]

applied to the residual at every iteration. With the constant component continuously
removed, CG sees an effectively SPD system on the orthogonal complement and converges to
the unique zero-mean pressure — which is all the projection in Chapter 6 needs, because
only \\( \nabla p \\) matters and the gradient is blind to the constant. gale's pressure
solver applies exactly this mean-removal each iteration. (Strictly, projecting out the
*known* constant nullspace is the nullspace-projection special case of deflation; the term
"deflated CG" in the Krylov literature usually means the heavier machinery of projecting
out approximate eigenvectors, which coincides here because the deflation subspace *is* the
nullspace.)

## The GPU residency / hybrid pattern

CG is iterative, often hundreds of iterations, and each iteration is several kernel
launches. The performance trap is **host↔device traffic**: if you copied a vector back to
the CPU on every iteration, the PCIe transfer would dwarf the actual arithmetic and the GPU
would sit idle waiting. The cardinal rule is therefore: **keep the iteration vectors
resident on the device for the entire solve**, and transfer only the few **scalar**
dot-products the host needs to form \\( \alpha \\) and \\( \beta \\).

In gale's CG kernels the solution, residual, search direction, and the matvec scratch all
live in `DeviceBuffer`s allocated once before the loop. Each iteration launches the
matrix-free apply (the `gradient` → `operator` two-kernel pipeline from Chapter 5), the
`dot_partial` reduction, and the `axpy`/`xpby` vector updates — all on-device. The only
data crossing the bus per iteration is a single `f64` partial-sum per dot product. This is
the same **hybrid** philosophy as the flow solver in Chapter 6 and the architecture chapter
(Chapter 11): the expensive, iterated work lives on the GPU; the cheap, one-time,
element-local assembly stays on the validated host code. Crucially, because the device
solver reuses the host-assembled right-hand side and the same metric/face data, it can be
checked **bit-for-bit** against the CPU reference (Chapter 12) — the GPU is fast *and*
trusted, not one at the expense of the other.

## How gale does it

The device kernels and host launch wrappers live in `gale-gpu/src/operators/poisson.rs`.
The kernels are deliberately small and composable:

- `gradient` and `operator` — the two-kernel matrix-free SIPG apply (\\( v \mapsto Av \\)),
  order-agnostic so one kernel set serves every p-multigrid level. The `operator` kernel
  carries an optional reaction term \\( \lambda \\), so the *same* code is the Poisson
  operator (\\( \lambda = 0 \\)) and the viscous Helmholtz operator (\\( \lambda > 0 \\)).
- `dot_partial` — a block-reduction dot product (the only thing that produces a
  host-visible scalar);
- `axpy` (\\( y \leftarrow y + a x \\)), `xpby` (\\( y \leftarrow x + b y \\)), `scal`, `sub` —
  the CG/V-cycle vector updates;
- `jacobi` — the damped-Jacobi smoother sweep \\( y \leftarrow y + \omega D^{-1}(b - Ap) \\);
- `prolong` / `restrict` — the inter-level transfer operators for the p-multigrid V-cycle,
  applied tensor-product-wise per element.

The host wrappers assemble these into complete solvers, each validated against its CPU
oracle:

- **`poisson_cg_solve`** — device-resident unpreconditioned CG for the SIPG Poisson system.
- **`helmholtz_cg_solve`** — the same CG with the reaction term \\( \lambda M \\) active; this
  is the viscous-velocity solve of the dual-splitting scheme (Chapter 6).
- **`pressure_cg_solve`** — **deflated** CG for the singular pure-Neumann pressure-Poisson:
  every boundary face is the natural Neumann condition, and the constant nullspace is
  removed from the residual each iteration via the mean-removal above.
- **`poisson_pcg_solve`** — the full **p-multigrid-preconditioned** CG, with the entire
  V-cycle (down-sweep Jacobi smoothing → coarsest-level inner CG → up-sweep prolong +
  smoothing) running on the GPU. Setup — the per-level meshes, interpolation matrices,
  inverse diagonals, and Jacobi weights — is taken from the validated CPU `PMultigrid`; only
  the iterations run on-device.

For non-conforming (AMR) meshes there are mortar-capable siblings (`poisson_nc_cg_solve`,
`pressure_nc_cg_solve`) that the flow solver selects automatically; the mortar machinery is
Chapter 10.

**Validation.** Every one of these solvers is held to the CPU reference `Poisson::cg` (and
`PMultigrid::pcg` for the preconditioned path) **bit-for-bit to solver tolerance**: same
convergence criterion \\( \|r\|/\|b\| < \mathrm{tol} \\), same reductions, same answer to
about \\( 10^{-14} \\) relative. That is the discipline of Chapter 1's "the CPU is the
oracle" — the GPU solvers are not trusted because the linear algebra looks right, but
because they reproduce a validated host computation exactly.

These solvers are the engine under everything in Part III: the flow solver of Chapter 6
calls them twice per step, and the viscoelastic coupling of Chapter 8 calls the same flow
solver inside its split. Next, Chapter 8 adds the polymers — and with them the
**high-Weissenberg-number problem**, the stability villain that the log-conformation
representation is built to defeat.
