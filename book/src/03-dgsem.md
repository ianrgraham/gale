# Why Discontinuous Galerkin?

We have the equations (Chapter 2). Now we need a way to turn continuous fields and
derivatives into finite arrays of numbers a computer — specifically a GPU — can march
forward. This is *spatial discretization*, and the choice gale makes is the
**discontinuous Galerkin spectral element method** (DG-SEM). This chapter explains what
that means and, in keeping with the book's habit, *why* it is the right choice for
low-Re, high-Wi, particle-laden flow on a GPU.

The short version: DG buys us high-order accuracy on smooth flow, tolerates unstructured
and adaptively-refined meshes for free, and — because elements barely talk to each other
— maps onto massively parallel hardware almost perfectly. No other method gives all three
at once.

## The discretization landscape, briefly

There are four broad families for discretizing a PDE, and it helps to know what each
trades.

- **Finite difference (FD).** Replace derivatives by difference quotients on a grid.
  Simple and fast, and high order is easy — *on a structured grid*. The fatal weakness
  for us is geometry: FD is awkward on unstructured meshes and around complex or moving
  boundaries.

- **Finite volume (FV).** Track cell averages and exchange *fluxes* across cell faces.
  This bakes in local conservation (what leaves one cell enters its neighbor exactly) and
  handles unstructured meshes and shocks gracefully. But pushing FV to *high order*
  requires wide reconstruction stencils that reach across many cells — which spoils
  locality and gets unwieldy on unstructured grids.

- **Continuous-Galerkin finite elements (CG-FEM).** Expand the solution in basis
  functions and enforce the equation in a weighted-average ("weak") sense. Naturally
  high-order and unstructured-friendly. But the basis is *continuous*: neighboring
  elements **share** degrees of freedom at their common nodes. That shared, globally
  coupled structure is the problem — it produces a large coupled system and, on a GPU,
  forces threads working on adjacent elements to coordinate over shared data.

- **Discontinuous Galerkin (DG).** Take the weak form of FEM, but let each element carry
  its **own, independent** polynomial — free to be discontinuous across element
  boundaries. Elements then couple *only* through a numerical flux on their shared faces,
  exactly as in finite volume.

DG is deliberately a hybrid: it inherits **high-order accuracy and unstructured meshing**
from finite elements and **flux-based local conservation and upwind stabilization** from
finite volume. The combination — high order *and* unstructured-friendly *and*
element-local — is precisely the trio that points to GPUs. We will see at the end of the
chapter that "element-local" is the load-bearing word.

## The weak form, from scratch

DG, like all Galerkin methods, starts from the *weak form*. Let us build it concretely
for a scalar conservation law in one dimension — the prototype for everything gale does
with advection:

\\[
\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0,
\\]

where \\( f(u) \\) is a flux (for linear advection at speed \\( a \\), \\( f = a\,u \\)). The
*strong* form above demands the equation hold pointwise. The weak form asks for less, and
that turns out to be both more flexible and more honest about polynomial approximations.

**Steps 1–2 — test, integrate, integrate by parts.** The standard Galerkin opening: pick
a test function \\( \phi \\) from our chosen space, integrate the equation over a *single*
element \\( \Omega_k \\) (the first DG-specific move — each element gets its own local
statement), and integrate the flux term by parts. Out pops a volume term plus a surface
flux term:

\\[
\int_{\Omega_k} \frac{\partial u}{\partial t}\,\phi\; dx
- \int_{\Omega_k} f\,\frac{\partial \phi}{\partial x}\; dx
+ \Big[\, f\,\phi \,\Big]_{\partial \Omega_k}
= 0,
\\]

required to hold for every test function \\( \phi \\). Two things just happened, both
important. First, the derivative was lifted off the (possibly rough) solution \\( u \\) and
placed onto the (smooth, chosen) test function \\( \phi \\) — so we only ever differentiate
things we control. Second, **a boundary term appeared**: the flux \\( f \\) evaluated on the
*surface* of the element, \\( \partial \Omega_k \\). In 1D that surface is just the two
endpoints; in 2D/3D it is the element's faces.

That first move is the real prize, and it is what makes the whole discontinuous game
legal: the strong form's pointwise derivative would be meaningless across a jump, but the
weak form only ever asks for an integral, which a piecewise — even discontinuous —
polynomial happily supplies. "Weak" is not a compromise; it is the thing that lets a
discontinuous polynomial be a legal solution at all.

**Step 3 — confront the discontinuity.** Here is the crux, and the place where DG actually
begins. Because each element has its own independent polynomial, at a shared face the
solution has *two* values — one from each side — and they generally disagree. So what is
the flux \\( f \\) in that surface term? There is no single answer; we must *choose* one. We
replace the ambiguous boundary flux with a single-valued **numerical flux** \\( f^* \\),
computed from both neighboring traces:

\\[
\Big[\, f\,\phi \,\Big]_{\partial \Omega_k}
\;\longrightarrow\;
\Big[\, f^*(u^-, u^+)\,\phi \,\Big]_{\partial \Omega_k}.
\\]

**That surface term is the *only* place neighbors enter.** It is the entire coupling
between elements, and choosing \\( f^* \\) well is where stability is won or lost — upwind
fluxes add just enough dissipation to keep advection stable, and that is the subject of
Chapter 4. The volume integrals, by contrast, involve only one element's own data.

Collecting the volume integrals into operators, the per-element semi-discrete form has the
shape every DG operator in gale takes:

\\[
\mathbf{M}\,\frac{d\mathbf{u}}{dt}
= \underbrace{\mathbf{S}\,f(\mathbf{u})}_{\text{volume (element-local)}}
- \underbrace{\mathbf{L}\,\big(f^* - f\big)\big|_{\text{faces}}}_{\text{surface (neighbor coupling)}},
\\]

with \\( \mathbf{M} \\) the mass matrix, \\( \mathbf{S} \\) the stiffness/differentiation
operator, and \\( \mathbf{L} \\) the *lift* operator that maps face data back into the
element interior. The diffusive (viscous, elliptic) terms get the same treatment but need
a more careful flux — the interior-penalty method of Chapter 5 — because second
derivatives are involved.

## "Discontinuous" means local, and local means GPU

Step back and look at what we have built. The solution is a **separate polynomial on each
element**, allowed to jump at interfaces. Compare:

- **Continuous FEM**: elements *share* nodal values at their common boundaries. The
  unknowns are stitched together into one globally coupled object.
- **Discontinuous Galerkin**: each element owns its nodes privately. The only thing
  crossing an interface is the numerical flux \\( f^* \\), a small amount of face data.

This locality is not a stylistic preference — it is *the* reason gale is built on DG. On
a GPU, the ideal workload is a large number of independent tasks that each touch only a
small, private chunk of memory. A DG element is almost exactly that: nearly all the work
(the volume integrals, the differentiation, the constitutive update) is element-private,
and the only communication is exchanging face traces with face-neighbors — a comparatively
tiny amount of data. You can assign **one GPU thread-block per element**, keep that
element's data in fast on-chip memory, and run thousands of elements in parallel with only
a thin layer of face exchange between them. Chapter 11 makes this concrete (including
multi-GPU, where the *only* inter-GPU traffic is face traces on the partition boundary).

The same locality is why DG is unusually friendly to **adaptive mesh refinement** (Chapter
10): when a fine element meets a coarse one at a hanging node, the mismatch is resolved by
the same numerical-flux machinery — projected through a small *mortar* — rather than by
re-stitching a global continuous basis. Continuous FEM has to work much harder here.

So "discontinuous" is not a defect we tolerate; it is the feature we exploit.

## Spectral elements: high order done cheaply

DG tells us *each element holds a polynomial* and *elements couple through fluxes*. It does
not yet tell us how to represent the polynomial or compute the integrals. gale's answer is
the **spectral element** construction (DG-SEM), and it is chosen to make the per-element
work both accurate and fast.

**Nodal Lagrange basis on GLL points.** Inside an element we represent a function by its
*values at a set of nodes*, with Lagrange interpolating polynomials as the basis (the
value at node \\( i \\) is the coefficient of the \\( i \\)-th basis function). The nodes are
the **Gauss–Lobatto–Legendre (GLL)** points — the roots of \\( (1-x^2)P_p'(x) \\) on the
reference interval \\( [-1,1] \\), which notably *include the endpoints* \\( \pm 1 \\).
Including the endpoints is what lets neighboring elements read each other's face values
directly, which is convenient for the flux. In gale these live in
`src/dg/reference/reference.rs` as `Reference1d`: the nodes, the quadrature weights, and a
dense **differentiation matrix** \\( \mathbf{D} \\) such that \\( (\mathbf{D}\,\mathbf{f})_i =
f'(r_i) \\) — exact for any polynomial of degree \\( \le p \\).

**Collocation: nodes = quadrature points.** The defining trick of DG-SEM is to use the
*same* GLL points both as interpolation nodes and as the quadrature points for the
integrals in the weak form. This collocation has a striking consequence: the **mass
matrix becomes diagonal** — it is literally just the vector of quadrature weights. (In
`Reference1d`, `mass_diagonal()` returns the weights directly.) This is technically an
under-integration: GLL quadrature is exact only to degree \\( 2p-1 \\), but \\( \phi_i\phi_j
\\) is degree \\( 2p \\), so the integral is *lumped* rather than exact — the true
consistent GLL mass matrix is full. It is the same benign aliasing we revisit in Chapter 5,
and the payoff is large: a diagonal mass matrix means inverting \\( \mathbf{M} \\) in the
semi-discrete form is a pointwise division, not a linear solve — a major saving repeated
every time step, and a recurring stability ally because it keeps operators clean and
symmetric. This is the first reason GLL specifically is chosen: it makes the mass matrix
diagonal cheaply.

**Tensor-product structure on quads and hexes.** In 2D a quadrilateral element's nodes are
just the Cartesian product of two 1D GLL point sets; in 3D a hexahedron is the product of
three. gale builds these in `src/dg/reference/quad.rs` (`Reference2dQuad`) and
`src/dg/reference/hex.rs` (`Reference3dHex`) directly as tensor products of `Reference1d`.
The mass diagonal in 2D is simply \\( w_i\,w_j \\); in 3D, \\( w_i\,w_j\,w_k \\).

**Sum factorization.** The tensor-product structure is the second, decisive reason for
quads/hexes — it is what makes high order *affordable*. A 2D derivative looks like it
should be a dense \\( (p+1)^2 \times (p+1)^2 \\) matrix acting on the nodal vector, costing
\\( O(p^{2d}) \\) per element. But because the operator factors along directions, you can
instead apply the *1D* differentiation matrix \\( \mathbf{D} \\) swept along one axis at a
time, costing only \\( O(p^{d+1}) \\). This is **sum factorization**, and you can see it
literally in `Reference2dQuad::diff_r` / `diff_s`: a triple loop that contracts the small
1D \\( \mathbf{D} \\) against one index of the nodal field. At \\( p \sim 4\!-\!8 \\) the
savings are enormous, and the work recasts into small dense matrix–matrix products — the
operation GPUs are fastest at. (Simplex elements — triangles, tetrahedra — lose this
tensor structure and force dense per-element operators, which is exactly why gale's fast
path is quads and hexes.)

**Spectral convergence, and why we want few, large, high-degree elements.** For an
*analytic* solution, increasing the polynomial degree \\( p \\) makes the error fall
**exponentially** (spectral convergence), not just algebraically as in low-order methods.
The caveat matters here: a merely-smooth-but-finitely-differentiable solution converges
only algebraically, at a rate set by how many derivatives it actually has — and a
viscoelastic stress layer or a boundary layer near an immersed particle is exactly where
analyticity fails and the exponential rate quietly degrades. Where the solution *is*
analytic, though, the convergence changes the economics entirely: instead of many tiny
low-order cells, we can use comparatively *few, large, high-degree* elements — gale
typically runs \\( p \approx 4\!-\!8 \\) — and still resolve smooth flow to high accuracy.
That matters enormously for the application: when you eventually want *many* particles in a
domain, you cannot afford a fine low-order grid everywhere, and high-order elements let you
keep the bulk flow cheap while spending resolution only where the physics is sharp (Chapter
10). The flip side — the honest stability caveat — is that high-order polynomials *ring*
(Gibbs oscillations) when the solution is *not* smooth, e.g. across a sharp stress layer.
That is why the book keeps returning to stabilization: numerical fluxes (Chapter 4),
filtering, and refinement (Chapter 10) exist to keep high order from turning into
high-amplitude noise.

## From reference element to physical element: the mapping

Everything above lives on a tidy reference element — \\( [-1,1] \\) in 1D, \\( [-1,1]^2 \\) on
the quad, \\( [-1,1]^3 \\) on the hex. Real meshes are made of physical elements of varying
shape and size. We connect the two with a **geometric map** from reference coordinates
\\( (r,s) \\) to physical coordinates \\( (x,y) \\).

For a straight-sided quad, that map is the bilinear interpolation of the four corner
vertices; gale builds it in `src/dg/geometry/geometry.rs` (`QuadGeometry::from_corners`).
The map matters because derivatives transform: a physical derivative \\( \partial/\partial
x \\) is obtained from reference derivatives \\( \partial/\partial r, \partial/\partial s \\)
via the chain rule, which brings in **metric terms** (\\( \partial r/\partial x \\), etc.)
and the **Jacobian determinant** \\( \det J \\) of the map. The Jacobian also rescales the
quadrature: the physical mass diagonal is \\( \det J \cdot w_{\text{ref}} \\) at each node.

A clean engineering trick used throughout gale: rather than carry separate analytic
formulas for the geometry, it computes the metric terms by **differentiating the nodal
coordinate fields with the very same reference operators** (`diff_r`/`diff_s` applied to
the \\( x \\)- and \\( y \\)-coordinate values). For a straight-sided quad the coordinate
fields are low-degree polynomials, so this is exact — and it generalizes to *curved*
(higher-order) elements later, which gale will want for body-fitted viscoelastic
benchmarks. One caveat rides along with that promise: on curved elements the *naive*
coordinate-differentiation above does **not** preserve free-stream — the metric terms fail
the discrete metric identities unless computed in conservative *curl* form (Kopriva's
conservative curl metrics). gale gets away with the naive approach today only because
straight-sided quads make the coordinate fields affine; curved elements will need the
conservative formulation. Two stability-relevant facts ride along here: \\( \det J > 0 \\)
everywhere is the condition that an element is not tangled or inverted (the failure mode of
moving and curved meshes), and the metric terms are precisely the per-node data the GPU
kernels carry alongside the field values.

## How gale does it

The pieces above are real, modular code, organized exactly along the
reference-element → physical-element → mesh hierarchy:

- **`src/dg/reference/`** — the reference elements that *are* DG-SEM.
  `Reference1d` (in `reference.rs`) holds the GLL nodes, weights, and the closed-form 1D
  differentiation matrix \\( \mathbf{D} \\), with the diagonal mass equal to the weights.
  `Reference2dQuad` (`quad.rs`) and `Reference3dHex` (`hex.rs`) build the 2D/3D elements as
  tensor products, exposing `diff_r`/`diff_s`(`/diff_t`) — the sum-factorization
  derivatives — and their transposes for the adjoint (stiffness / lift) terms. Their tests
  verify the load-bearing properties: GLL quadrature exact to degree \\( 2p-1 \\), the
  derivative exact on polynomials, and the mass diagonal summing to the reference measure.

- **`src/dg/geometry/`** — the physical-element layer. `QuadGeometry` (`geometry.rs`, and
  its 3D sibling `geometry3d.rs`) stores the per-node metric terms (\\( r_x, r_y, s_x, s_y
  \\)), the Jacobian \\( \det J \\), and the physical mass diagonal \\( \det J\cdot w \\),
  computed by differentiating the nodal coordinates with the shared reference operators.
  Geometry is deliberately *decoupled* from the operators: one shared reference element
  serves many physical elements, each carrying only its own small metric arrays — the
  one-reference / many-elements layout the GPU port wants.

- **`src/dg/mesh/`** — the connectivity backbone. `Mesh2d` (`mesh.rs`) and `Mesh3d`
  (`mesh3d.rs`) are **face-based and unstructured-capable from the start**: each element
  stores its geometry, its face traces, and a `Neighbor` per face. The `Neighbor` enum
  encodes exactly the coupling DG cares about — an `Interior` face (the neighbor element,
  its edge, and a node permutation), a `Boundary`, and the `CoarseToFine` /
  `FineToCoarse` variants for the 2:1 non-conforming interfaces that AMR (Chapter 10)
  introduces. Interior-face trace nodes are matched by physical-coordinate proximity, so
  the connectivity is orientation-agnostic. A small `DgMesh` trait (`dgmesh.rs`) exposes
  just element count, nodes-per-element, and total measure, letting the rest of the
  framework be generic over 2D vs 3D.

The shape of all of this — one shared reference, per-element geometry, faces as the only
coupling — is exactly what the GPU kernels exploit: **one thread-block per element**,
operators resident in fast memory, and a thin face-exchange layer between blocks. The next
chapter takes the surface term we exposed in the weak form and asks the question that
decides stability: *what numerical flux do we put there?*
