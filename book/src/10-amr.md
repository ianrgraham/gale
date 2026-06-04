# Adaptive Mesh Refinement

> 🎓 **Reviewer — chapter verdict:** The best section here is "Three properties the mortar must have" — each property is argued as a genuine *requirement* with a named failure mode (leak / phantom free-stream kink / CG stalls), which is exactly the "what breaks if you skip it" framing the mandate wanted, and it's correct. The symmetry-preservation argument in particular is the kind of insight most textbooks assert and never justify; gale's book earns it. Two things to fix: the Persson–Peraire description is a defensible simplification but drops the one feature that makes it *Persson–Peraire* (the log-scale / single-mode comparison against a \\( p \\)-dependent threshold), and the conservation Jacobian arithmetic (the "½ × ½ = ¼" sentence) is currently a magic-numbers claim a reader can't follow. The honesty about the p-/hp gap is appropriately handled and not over-claimed — leave that alone.

The features that matter in gale's target flows are *thin and local*. The viscoelastic
stress boundary layers of Chapter 8 collapse into birefringent strands a fraction of a
millimetre wide. The immersed interface of Chapter 9 is smeared over a cell that you would
dearly like to make smaller. The wake behind a particle, the stagnation point where two
particles nearly touch — all of these need fine resolution in a small region and none
elsewhere.

You could just make the *whole* mesh fine. But a uniformly fine mesh pays the cost of the
worst feature everywhere, and for a suspension with many particles scattered through a
large domain that cost is ruinous — you would resolve empty fluid to the same tolerance as
the contact zones. The answer is **adaptive mesh refinement (AMR)**: let the mesh add
resolution where the solution needs it and remove it where it does not, dynamically, as the
flow evolves.

DG is unusually well suited to this. Recall from Chapter 4 that DG elements communicate
*only* through numerical fluxes on shared faces — there is no global continuity constraint
tying nodes together. That locality means you can refine one element without disturbing the
representation inside its neighbors; the only thing you must handle is the *flux on the
shared face*, which DG already treats as a first-class object. AMR in a continuous
finite-element method requires fussy constraint equations to keep the solution continuous
across a refinement boundary; in DG it reduces to "what is the flux across a mismatched
face?" That is the entire technical content of this chapter.

> 🎓 **Reviewer:** This is the right hook and it's correct — DG buys you AMR almost for free precisely because the coupling is already a face object. Keep it. The one nuance worth a clause: "the only thing you must handle is the flux" is true for the *hyperbolic* operator, but the elliptic SIPG operator also couples *gradients* across the face, which is why section "Three properties" has to worry about symmetry at all. You half-acknowledge this later ("the gradient and its transpose are also an adjoint pair"); a forward nod here ("for elliptic operators there's a second trace — the gradient — but the same mortar handles it") would stop a sharp reader thinking the flux is the whole story.

## The flavors of adaptivity

There are three ways to add resolution to a high-order element method:

- **h-adaptivity** — *subdivide* an element into smaller children (in 2D, one quad becomes
  four). More, smaller elements. Best for sharp or under-resolved features where the field
  is locally non-smooth.
- **p-adaptivity** — *raise the polynomial degree* \\( p \\) inside an element, leaving its
  size alone. Best for features that are smooth but under-resolved, where spectral accuracy
  rewards higher order exponentially.
- **hp-adaptivity** — do both, choosing per element which knob to turn. This is the
  powerful combination: \\( h \\) where the field is rough, \\( p \\) where it is smooth.

State plainly where gale is: **2D h-adaptivity is built and validated.** p-adaptivity and
hp-adaptivity are **documented gaps** — `Mesh2d` carries a single uniform order, and there
is no per-element \\( p \\) axis to vary (Chapter 13, and the forward link at the end of this
chapter). 3D non-conforming meshes are likewise a future step. Everything below is the
\\( h \\)-adaptive 2D story, which is real.

## The non-conforming (hanging-node) problem

The moment you refine one element and not its neighbor, you have a problem. Consider a
coarse element whose east face is shared with two refined elements stacked along that edge:

```text
  ┌─────────┬────────┐
  │         │   R1   │     coarse C  ↔  two fine R0, R1
  │    C    ├────────┤     C.East (full edge)
  │         │   R0   │       ↔ R0.West (lower half) + R1.West (upper half)
  └─────────┴────────┘
```

The shared face now has **mismatched nodes on each side**: the coarse element places its
\\( p+1 \\) edge nodes across the full edge, while the two fine elements together place
\\( 2(p+1) \\) nodes across the same physical length. The node that sits at the midpoint of
the coarse edge — where the two fine elements meet — is a *hanging node*: it exists on the
fine side but has no partner on the coarse side. You cannot simply match node to node and
compute a flux; the two sides do not even agree on how many nodes there are.

The fix is the **mortar method**. Instead of pretending the nodes line up, you introduce a
shared *mortar* on the interface at the **fine resolution**, and you project both sides
onto it. Concretely, for the coarse–fine interface:

1. Take the coarse element's edge trace and **project it up** to each fine half-edge —
   evaluate the coarse polynomial at the fine nodes. Call this operator \\( P \\). It is
   exact for polynomials of degree \\( \le p \\).
2. Compute the numerical flux on each fine half-edge, where now *both* the coarse
   (projected) and the fine traces live at the same fine nodes.
3. **Project the result back down** to the coarse edge with the transpose \\( P^{T} \\), so
   the coarse element receives a single, consistent flux contribution on its \\( p+1 \\)
   edge nodes.

In gale these are `RefineQuad::mortar_to_fine` (the prolongation \\( P \\)) and
`RefineQuad::mortar_to_coarse` / `mortar_gather` (the restriction \\( P^{T} \\)). The fine
side does its flux loop normally; the coarse side gathers the back-projected mortar flux.

## Three properties the mortar must have

Computing the flux on the fine resolution is only half the battle. The projection
operators \\( P \\) and \\( P^{T} \\) must satisfy three properties, and *each one is a
stability or correctness requirement* — this is the chapter's "why."

**1. Conservation.** The flux integral over the coarse edge must equal the *sum* of the
flux integrals over the two fine half-edges:

\\[
  \int_{\text{coarse edge}} F^\ast \, ds
  \;=\;
  \sum_{h=0}^{1}\ \int_{\text{fine half } h} F^\ast \, ds .
\\]

Whatever mass or momentum leaves the coarse element across that face must arrive,
*exactly*, at the two fine elements — not a little more, not a little less. If the mortar
gets this wrong, the discrete conservation law has a *leak at every non-conforming
interface*, and in a suspension with hundreds of such interfaces those leaks accumulate
into spurious mass/momentum sources that corrupt the whole solution. gale's mortar is
conservative by construction: the conservative restriction carries a \\( \tfrac12 \\)
Jacobian, and the geometry conspires so that (coarse-edge Jacobian \\( \tfrac12 \\)) \\(\times\\)
(mortar restriction \\( \tfrac12 \\)) equals the fine-edge Jacobian \\( \tfrac14 \\). The tests

> 🎓 **Reviewer (flag):** As written this is magic-number arithmetic — \\( \tfrac12 \times \tfrac12 = \tfrac14 \\) is arithmetically obvious but the reader has no idea *what* these three Jacobians are or why two of them should multiply to the third. The real statement is geometric and worth one clean sentence: each fine half-edge is *half the length* of the coarse edge, so the fine-edge surface Jacobian is half the coarse-edge one; the mortar restriction \\( P^{T} \\) carries the matching \\( \tfrac12 \\) so that summing the two half-edge contributions exactly reconstitutes the coarse-edge integral. (The "\\( \tfrac14 \\)" only appears if you're measuring against a 2D *element* Jacobian rather than an *edge* Jacobian — which is it? Right now the reader can't tell, and the dimensional bookkeeping looks off: a 1D edge integral picks up a 1D length factor, not an area factor.) Please pin down whether these are edge or element Jacobians and state the length-halving explicitly; the conclusion is correct, the derivation as shown is unfollowable.

The tests
confirm the coarse-edge flux integral equals the sum of the fine half-edge integrals to
\\( 10^{-12} \\), and that a refined mesh's total rate of change matches the analytic
outer-boundary flux to \\( 10^{-9} \\).

**2. Consistency.** A *uniform* field must produce *no spurious interface jump*. If the
solution is the constant \\( u \equiv C \\) everywhere, the numerical flux across the
hanging-node interface must be exactly the physical flux of that constant — the mortar must
not invent a jump where the field is smooth. Violate this and you corrupt the **free
stream**: a uniform flow develops phantom kinks at every refinement boundary, which is both
wrong and a seed for instability. This is the single most decisive test for a mortar
implementation, and gale checks it directly — `free_stream_preserved_across_hanging_node`
holds a constant field and demands the residual stay below \\( 10^{-10} \\); the general
refined-mesh test does the same across many hanging nodes at once. Because \\( P \\)
reproduces a constant exactly, consistency follows.

**3. Symmetry preservation.** For the elliptic operators — the SIPG Poisson and Helmholtz
solves that are the bottleneck of every incompressible step (Chapter 5) — the discrete
operator must stay **symmetric**. Recall from Chapter 7 that the Krylov solver of choice,
conjugate gradient (CG), *requires* a symmetric positive-definite matrix; lose symmetry and
CG loses its convergence guarantee and can stall or diverge. The mortar threatens this:
if you projected the coarse trace up with \\( P \\) but scattered the coarse test-function
contribution back with something *other than* the exact transpose \\( P^{T} \\), the operator
\\( A \\) would no longer satisfy \\( \langle A u, v\rangle = \langle A v, u\rangle \\). gale
deliberately uses \\( P \\) and its *exact* transpose \\( P^{T} \\) (`mortar_to_fine` and
`mortar_gather` are an adjoint pair), so that — combined with the fact that the gradient and
its transpose are also an adjoint pair — the SIPG operator stays symmetric across the
hanging node. The test verifies \\( \langle A u, v\rangle = \langle A v, u\rangle \\) to
\\( 10^{-9} \\) on a refined mesh, and a manufactured solve converges. This is why the same
mortar serves both the hyperbolic flux and the elliptic operator: conservation and
consistency keep the flux right; symmetry preservation keeps CG (Chapter 7) working.

> 🎓 **Reviewer:** This is the best paragraph in the chapter. "Use \\( P \\) and its *exact* transpose, or CG loses its convergence guarantee" is a real requirement with a real, named failure mode, and most treatments simply assert "the mortar must be adjoint-consistent" and move on. The throughline payoff — same mortar, three properties, three different things that break — is genuinely satisfying. Keep verbatim. (If you wanted to twist the knife one notch: a *non*-symmetric mortar doesn't always blow up loudly; it often just degrades CG into a slow stall that looks like a preconditioner problem, so people waste days before suspecting the mortar. That "it fails quietly" detail is what a practitioner would add.)

## Where to refine: the smoothness indicator

AMR needs a criterion: *which elements need more resolution?* gale uses the **Persson–Peraire
smoothness indicator**, a spectral-decay measure. The idea is that a well-resolved field,
expanded in the element's Legendre modes, has coefficients that *decay rapidly* toward the
highest mode — the top modes carry almost no energy. An under-resolved field, or one with a
sharp feature the polynomial cannot represent, has energy *piled up in its highest modes*.
So the indicator is simply the fraction of the element's L2 energy living in its highest
modes:

\\[
  s_e \;=\; \frac{\text{energy in the top modes}}{\text{total energy}} \in [0,1].
\\]

A smooth, well-resolved field gives \\( s_e \approx 0 \\); a pure top-mode field gives
\\( s_e = 1 \\); an under-resolved oscillation lands in between. Elements whose indicator
exceeds a threshold are flagged for refinement

> 🎓 **Reviewer (flag):** The *intuition* (spectral decay = smoothness) is exactly right and well told, but as a description of the *Persson–Peraire* indicator specifically it's missing the two features that make it that indicator rather than a generic high-mode-energy ratio. PP (Persson & Peraire 2006) (1) typically compares the energy in the *single highest mode* (or the top mode against the field minus its top mode), not a band of "top modes," and (2) crucially works on a **logarithmic** scale, \\( s_e = \log_{10}(\text{top-mode energy}/\text{total}) \\), then compares against a threshold that *scales like* \\( \sim 1/p^4 \\) — because for a smooth field the modal energy decays algebraically and the natural resolution scale is \\( p \\)-dependent. The linear \\( s_e \in [0,1] \\) "fraction of energy" you describe is a reasonable cousin and may well be what gale actually computes — but then it's a simplification of PP, not PP itself, and the chapter should either (a) say "a Persson–Peraire-*style* decay measure" or (b) describe the log/\\( p^{-4} \\) threshold. As stated, a reader who later reads the PP paper will be confused that the formulas don't match. Cheap fix: one sentence acknowledging the log-scale, \\( p \\)-dependent threshold, or soften to "PP-style."

Elements whose indicator
exceeds a threshold are flagged for refinement — for example the high-stress strands at a
particle's near-contact, or the cells the immersed mask cuts through. gale's
`SmoothnessIndicator` computes exactly this via a nodal-to-modal transform and Parseval's
theorem, and the tests confirm it flags top-mode and oscillatory fields while ignoring
smooth ones.

This same indicator is, in principle, what tells an *hp* scheme which knob to turn: a cell
that is under-resolved but *smooth* (its high-mode energy decaying, just not fast enough)
wants \\( p \\); a cell that is genuinely rough or discontinuous wants \\( h \\). gale only acts
on it with \\( h \\) today.

## Dynamic AMR: refining while the simulation runs

Flagging elements is static; the point of AMR is to do it *during* a run, re-adapting the
mesh as features move. That requires a **conservative solution remap**: when a cell is
newly refined, its field must be transferred to the four children; when four children are
coarsened back, their fields must be combined into the parent — and the cell integral
(mass/momentum) must be preserved through both directions.

gale provides exactly this pair:

- **prolong** (parent → children): evaluate the parent polynomial at each child's nodes.
  Exact for degree \\( \le p \\), so refinement *loses nothing*.
- **restrict** (children → parent): the conservative, mass-weighted L2 adjoint of prolong.
  It preserves the cell integral, and the restriction of a constant is exactly that
  constant.

`adapt_scalar` runs the closed loop — indicate (`SmoothnessIndicator`) → refine
(`Mesh2d::cartesian_refined`) → transfer (prolong) — and `remap_scalar` handles a general
transition between two refinement states (prolong the newly-refined, conservatively
restrict the newly-coarsened, copy the unchanged). The capstone test steps a uniform field
through the non-conforming advection operator *while re-adapting the mesh* (refine, then
coarsen) and keeps the free stream to \\( 1.3\times10^{-14} \\): the mesh changes underneath
the solution without corrupting it or leaking mass.

## The hp-adaptivity gap versus the literature

It is worth being precise about what gale trails on. The reference point is Nayak &
Mavriplis, *"Immersed boundaries in the discontinuous Galerkin spectral element method
through hp-adaptivity"* — a paper that uses **the same DG-SEM family**, **the same Brinkman
volume-penalty IBM** as Chapter 9, and adds **hp-adaptivity localized to the immersed
boundary**. Its core result is that piling \\( h \\)- and \\( p \\)-refinement into a band
around the smeared interface *suppresses the volume-penalty accuracy loss* — it is the
recipe for making the IBM accurate, exactly the remedy flagged in Chapter 9.

gale overlaps on the core (DG-SEM + volume-penalty IBM + \\( h \\)-AMR) and goes well beyond
the paper on physics, dimensionality, and hardware (the paper solves the 2D linear acoustic
wave equation on CPU; gale has GPU, multi-GPU, 3D, incompressible Navier–Stokes, and
viscoelasticity). But it trails on the one axis the paper specializes in: **p- and
hp-adaptivity**. gale's AMR is \\( h \\)-only, and the refinement criterion is the smoothness
indicator, not the IB mask. Closing this would mean generalizing the mortar from an
\\( h \\)-projection to a \\( (p_\text{left}\!\to\!p_\text{right}) \\) projection — the same
machinery, with a rectangular 1D operator between mismatched-order traces — giving each
element its own order, and adding an IB-targeted refinement band. That is the documented
path forward (Chapter 13); it is honestly *not built yet*, and localized hp-refinement is
how one would eventually make the immersed boundaries of Chapter 9 high-order accurate.

> 🎓 **Reviewer:** The honesty here is well-judged — neither defensive nor over-promising — and "generalize the mortar from an \\( h \\)-projection to a \\( (p_L\!\to\!p_R) \\) projection" is *exactly* the right characterization of the gap; that's the same Galerkin-projection machinery with a rectangular operator, and it's good that the book names it precisely rather than waving at "future work." One caution so the reader doesn't underestimate the lift: \\( p \\)-adaptivity isn't *only* a mortar change. A per-element \\( p \\) axis ripples through the whole data layout — `Mesh2d`'s "single uniform order" assumption (you flag this earlier) is baked into how nodal arrays are sized and how the GPU kernels are launched with a compile-time order. So "the same machinery, with a rectangular 1D operator" is true at the *math* layer but undersells the *engineering* (variable-length per-element state, divergent kernel launches). Worth half a sentence, because a newcomer reading this might think it's a weekend's work on one operator.

## How gale does it

- **`RefineQuad`** (`src/dg/amr/`): the transfer toolkit. `prolong` / `restrict` move a
  solution between refinement levels (exact prolongation, conservative restriction); the
  mortar projections `mortar_to_fine` (\\( P \\)) and `mortar_to_coarse` / `mortar_gather`
  (\\( P^{T} \\)) couple a non-conforming face — verified conservative and consistent, and an
  exact adjoint pair so the SIPG operator stays symmetric.
- **`SmoothnessIndicator`** (`src/dg/amr/`): the Persson–Peraire spectral-decay criterion
  for *where* to refine.
- **`NcMesh`** (`src/dg/mesh/nonconforming.rs`): builds 2:1-balanced non-conforming meshes
  (any set of single-level-refined Cartesian cells), with the four neighbor relations
  (boundary, conforming, coarse-to-fine, fine-to-coarse) and the mortar flux wired into the
  face loop. Validated: free-stream preserved and linear advection exact across every
  hanging node, conservation to the analytic boundary flux.
- **Dynamic AMR** (`src/dg/amr/`): `adapt_scalar` (indicate → refine → transfer) and
  `remap_scalar` (general state transfer), enabling refine/coarsen mid-run with a
  conservative remap.
- **On the GPU** (`gale-gpu/src/operators/poisson_nc.rs`): the `poisson_nc` operator
  evaluates the matrix-free SIPG Laplacian on a 2:1 non-conforming mesh — reconstructing the
  mortar matrices \\( P \\) and \\( P^{T} \\) host-side and applying them on-device — validated
  bit-for-bit against the CPU oracle, with `GpuStokes` running the full dual-splitting flow
  on refined meshes. 3D non-conforming refinement is a documented future step.

With immersed boundaries (Chapter 9) and adaptive meshes (this chapter) in hand, Part IV is
complete: gale can place objects in the flow without meshing them, and resolve the thin
features they create without paying for fine resolution everywhere. Part V turns to how all
of this is implemented on the GPU and how we convince ourselves it is correct.
