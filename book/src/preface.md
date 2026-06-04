# Preface

> 🎓 **Reviewer — chapter verdict:** A clean, confident preface that knows exactly who it's for and sets the "stability is the throughline" hook early — that hook is the best thing here, keep it front and center. Biggest improvement: the "What you will get out of it" bullet list reads like a syllabus learning-outcomes box, which is the one un-bloggy moment; loosen it.

This book explains the *theory* behind **gale** — a from-scratch, GPU-native fluid
solver written in Rust — and, just as importantly, the *reasons* behind the methods it
uses. It is written for someone who is comfortable with the idea of simulating fluids
but has never worked with discontinuous Galerkin (DG) methods, immersed boundaries,
viscoelastic constitutive models, or two-phase / particle-laden flows. If that is you,
you are in exactly the right place.

## What you will get out of it

By the end you should be able to:

- read gale's source and know *why* each operator is built the way it is;
- explain, to someone else, what "discontinuous Galerkin spectral element method" means
  and why it is a good fit for GPUs;
- understand the chain that turns the incompressible Navier–Stokes equations into code:
  weak forms, numerical fluxes, the interior-penalty Laplacian, projection methods,
  Krylov solvers;
- understand what makes viscoelastic flow *hard* — the High-Weissenberg-Number Problem —
  and why the log-conformation representation is the standard cure;
- understand how a solid object can live inside a fluid mesh without being meshed
  (immersed boundaries), and how the grid can refine itself (adaptive mesh refinement);
- and, throughout, recognize the recurring villain of the whole subject: **numerical
  instability**, and the handful of ideas we use to defeat it.

> 🎓 **Reviewer (rewrite):** This six-bullet "By the end you should be able to..." block is the most textbook-flavored moment in an otherwise blog-voiced preface — it reads like an accreditation learning-outcomes table. The content is good; the framing is stiff. Consider collapsing the lead-in to something with momentum, e.g.: "Here's the deal: by the last page you should be able to open gale's source and know not just *what* each operator does but *why* it's shaped that way — from the weak form, to the High-Weissenberg cure, to how a solid hides inside a mesh it never touches." Then keep maybe two or three bullets for the genuinely distinct payoffs rather than six near-parallel ones. The last bullet (instability as the villain) is the strongest — it earns its place.

## The throughline: stability

If there is one theme in this book, it is that **most of the design decisions in a
high-order fluid solver exist to keep the simulation stable.** A scheme that is merely
*accurate* on paper will happily produce `NaN` on a real problem. So at almost every
step we will ask the same question — *what could blow up here, and what are we doing
about it?* — and the answer usually explains the design:

- numerical fluxes and upwinding (Chapter 4),
- split / entropy-stable forms and filtering (Chapter 4),
- the interior-penalty parameter (Chapter 5),
- operator-splitting and implicit viscous solves (Chapter 6),
- the log-conformation representation (Chapter 8),
- implicit volume penalization (Chapter 9),
- conservative, symmetry-preserving mortars (Chapter 10).

Each of those is a *stability* story first and an accuracy story second.

> 🎓 **Reviewer:** This is the best section in the preface — naming "what could blow up here, and what are we doing about it?" as the recurring question is exactly the kind of organizing spine a blog series wants, and it pays off because every Part II–IV chapter genuinely does answer it. Keep this verbatim. One small caveat to consider adding in a half-sentence: it's not *only* stability — pressure-robustness and conservation also drive several of these choices, and a reader who internalizes "everything is stability" might be surprised in Ch. 6/10. A parenthetical "(stability is the loudest motive, though accuracy and conservation share the bill)" would keep you honest without diluting the hook.

## How the book is organized

- **Part I** sets up the problem: what gale is for, and the physics of incompressible,
  viscoelastic, particle-laden flow.
- **Part II** builds the spatial discretization — the discontinuous Galerkin spectral
  element method — for hyperbolic (advection-like) and elliptic (diffusion-like) terms.
- **Part III** assembles those pieces into an incompressible Navier–Stokes solver, adds
  the GPU linear solvers it needs, and then adds polymers (viscoelasticity).
- **Part IV** handles complex and moving geometry: immersed boundaries and adaptive
  meshes.
- **Part V** covers the software architecture, the GPU implementation, and how we
  convince ourselves the whole thing is correct.
- **Closing** maps the road ahead and collects notation.

You can read it front to back, or jump to a Part — each chapter opens with a short "what
and why" and closes with "how gale does it," pointing at the actual modules.

## A note on prerequisites and notation

You need multivariable calculus (gradients, divergence, integration by parts) and a
little linear algebra (eigenvalues, symmetric matrices, conjugate gradients are
explained as we go). No prior PDE-discretization coursework is assumed; we build the
weak form from scratch in Chapter 3.

> 🎓 **Reviewer (flag):** "No prior PDE-discretization coursework is assumed" sits a little awkwardly against the audience line in the opening paragraph ("comfortable with the idea of simulating fluids"). A reader genuinely new to *all* discretization who is also comfortable simulating fluids is a narrow target. This is fine, but be aware you're promising a fairly steep on-ramp — DG-SEM from a standing start is a real climb. Worth one honest sentence somewhere: this assumes you've *seen* a simulation, even if you've never built a discretization yourself.

Mathematical conventions used throughout (collected in the [Appendix](appendix-notation.md)):

- \\( \mathbf{u} = (u, v, w) \\) is velocity, \\( p \\) pressure, \\( t \\) time.
- Bold lowercase = vectors, bold uppercase or sans = tensors/matrices, e.g. the
  conformation tensor \\( \mathbf{C} \\).
- \\( \nabla \cdot \\) is divergence, \\( \nabla \\) gradient, \\( \nabla^2 = \Delta \\)
  the Laplacian.
- \\( \rho \\) density, \\( \mu \\) (or \\( \eta \\)) viscosity, \\( \nu = \mu/\rho \\)
  kinematic viscosity.
- "Element" = one cell of the mesh; \\( p \\) (overloaded with pressure, but the meaning
  is always clear from context) = the polynomial degree inside an element; \\( h \\) = a
  representative element size.

> **A word on honesty.** gale is a research code under active construction. Where a
> capability is *built and validated* this book says so; where it is *planned* or
> *parked* (two-phase flow and deformable particles are mostly the latter) the book
> says that too, and Chapter 13 keeps the score. The theory is worth learning either
> way — it is the same whether the feature ships today or next month.

> 🎓 **Reviewer:** Excellent — this "word on honesty" is exactly the right tone for a research-code book and it builds trust immediately. Keep it. It also defuses the over-promise risk in the bullet list above, so the two should stay consistent: if a payoff in "What you will get" is really a parked feature (two-phase), make sure the verb tense there matches the honesty you promise here.
