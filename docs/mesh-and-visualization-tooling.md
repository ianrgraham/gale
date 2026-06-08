# Mesh generation & headless visualization tooling — research findings

*Verified deep-research pass, 2026-06-08. 5 angles, 22 sources fetched, 88 claims
extracted, 25 adversarially verified (22 confirmed / 3 refuted). Confidence tags are
the pass's own. This doc records the decision basis; the build work it implies is
tracked in the roadmap.*

## TL;DR / decisions

1. **Roll our own mesher** — the Rust ecosystem cannot serve boundary-fitted curved
   quad/hex meshing today, and won't soon.
2. **Model it on HOHQMesh** — geometry-as-code control file; curve primitives
   (parametric equation, cubic spline, line, circular/elliptic arc) + a polynomial-order
   directive; 3D hex by extrusion/sweep/rotation of a 2D quad model. That matches gale's
   extrudable microfluidic geometries exactly.
3. **Import path = HOHQMesh ISM-V2** — the one "bring your own curved mesh" format to
   support.
4. **Output = high-order `.vtu`** (VTK XML, arbitrary-order Lagrange cells), written by
   the pure-Rust **`vtkio`** crate — no VTK C++ install, no subsampling, no ParaView in
   the loop to produce files.

## 1. Rust meshing/CAD ecosystem — a genuine gap *(high confidence)*

There is **no Rust crate for boundary-fitted curved quad/hex meshing**.
- **Fornjot** (b-rep CAD kernel, *not* a mesher): mainline "has not been developed in
  over a year"; last release v0.49.0 on 2024-03-21; effort moved to `experiments/`. Even
  if revived, a CAD kernel is the wrong category — it emits triangle meshes for export,
  not SEM quad/hex. *(The stronger claim "contains no mesh functionality at all" was
  refuted 0-3.)*
- **spade**: 2D-only Delaunay/CDT.
- **parry**: collision-detection library; mesh support is TriMesh/HeightField/Polyline/
  Voxels only — no quad/hex element type.

Curvilinear mesh generation is an acknowledged hard problem "served by only a few
open-source tools" (NekMesh, CPC 2024) — **none Rust-native**. So the gap is real and
specific; an in-house Rust mesher fills it rather than reinventing a solved problem.

## 2. Reference design — HOHQMesh *(high confidence)*

HOHQMesh (Trixi.jl) "automatically creates quadrilateral/hexahedral meshes with
high-order boundary information," with "curved elements sized according to the geometry."
Workflow = a **MODEL** of boundary curves + a **CONTROL_INPUT** (background grid size,
smoothing, refinement regions). Curve primitives:
- `PARAMETRIC_EQUATION_CURVE` — arbitrary `x(t), y(t), z(t)` ("any legal equation");
- `SPLINE_CURVE` — cubic spline through knots;
- `END_POINTS_LINE` — straight;
- `CIRCULAR_ARC`, `ELLIPTIC_ARC`.

These map directly onto channels (lines), 4:1 contractions (lines/arcs), cross-slots,
and cylinders (arcs / parametric circles). `polynomial order = N` sets the boundary-curve
interpolant degree. **3D hex is produced only by extrusion/sweep/rotation of the 2D quad
mesh** — perfect for gale's extrudable targets, a hard limit for anything non-extrudable.

The in-house build is therefore a small **transfinite (Gordon–Hall) block-structured
mapper** over this primitive set, geometry expressed in Rust code.

## 3. Import format — ISM-V2 *(high confidence)*

ISM-V2 stores curved geometry **natively**: `(N+1)` nodal points per curved quad side and
`(N+1)²` per curved hex face, sampled at **reversed Chebyshev–Gauss–Lobatto** points
`t_j = −cos(jπ/N)`. Connectivity = 4 corner-node IDs (quad) / 8 (hex) + per-side/face
curved flags (0 straight, 1 curved) + boundary names (`---` interior); ISM-V2's
distinguishing feature over ISM is an explicit **edge-to-element** connectivity list.
HOHQMesh also emits ISM, ISM-MM, ABAQUS. ISM-V2 is the best single import target.

> **Node-convention caveat (open question):** gale's DG-SEM uses **Legendre**-Gauss-
> Lobatto nodes; ISM-V2 stores boundary nodes at **reversed Chebyshev**-Gauss-Lobatto
> points. Import needs an interpolation/remap from CGL boundary data onto gale's LGL
> geometry. Confirm the exact convention and remap before relying on imported curvature.

## 4. Field output — high-order `.vtu`, no VTK install *(high confidence)*

High-order fields can be written **natively, no subsampling**:
- VTK "can render … quadrilaterals … hexahedra … of any order up to 10" via Lagrange
  shape functions, representing curved geometry and nonlinear fields directly. *(The
  claims that high-order must be subsampled to linear were refuted 0-3.)*
- Point ordering is a defined recursive scheme: corner vertices (linear order) → mid-edge
  → face → interior, boundary-to-interior along axis-aligned parameter directions. This is
  the spec needed to emit valid high-order `.vtu` by hand.
- **No VTK C++ dependency.** VTKHDF (HDF5-based) is "much easier to write … without
  depending on VTK itself" — **but as of Nov 2025 VTKHDF does NOT yet support high-order
  elements** (listed as a planned addition). So the emit target is the **`.vtu` XML**
  format (which does support arbitrary-order Lagrange cells).
- **`vtkio`** is pure-Rust (XML via `quick-xml`+`serde`, legacy via `nom`), no VTK C++
  install — gale can emit `.vtu` in-process.

> **Open question:** does `vtkio` actually expose the high-order Lagrange cell types
> (`VTK_LAGRANGE_QUADRILATERAL` = 70, `VTK_LAGRANGE_HEXAHEDRON` = 72) and their point
> ordering, or only linear cells? If only linear, we hand-write the XML (the spec above)
> or contribute the cell types upstream. Cross-check ordering against a known-good
> reference file before production emit.

**Smooth-viz fallback** for any consumer that can't read native high-order cells: the
canonical Remacle et al. (IJNME 2005, the Gmsh high-order-viz method) approach —
recursively subdivide each element into same-type sub-cells until the *exactly known*
error between the high-order field and its piecewise-linear drawable falls below a
threshold; goal-oriented (refine only near a cut plane / iso-surface).

## 5. Offline / Rust-native renderers — not yet evidence-backed

Focus area 4 (rerun, wgpu, three-d, plotters; PyVista-headless, VisIt, vtk.js) produced
**no surviving verified claims** — treat any renderer recommendation as unverified. Moot
for now: the target boxes are headless and the workflow is file-export-first. Revisit if
gale moves to display-capable consumer hardware (see the consumer-GPU precision thread).

## Caveats & re-check triggers

- Fornjot's stall is current as of mid-2026; its `experiments` could replace mainline —
  re-check before fully dismissing.
- VTKHDF's lack of high-order support is the claim most likely to expire (Kitware lists it
  as planned) — re-verify before assuming VTKHDF can't carry high-order.
- The hand-written `.vtu` high-order spec comes from one (authoritative, Kitware) source;
  cross-check point ordering against the XML format spec + a reference file.
- HOHQMesh 3D = extrusion only.

## Sources (primary)

- Fornjot · spade · parry (GitHub) — Rust ecosystem status.
- HOHQMesh repo + control-input docs + ISM mesh-format docs (Trixi.jl).
- NekMesh, *Comput. Phys. Commun.* (2024) — curvilinear meshing is hard/served.
- Kitware: arbitrary-order Lagrange in VTK; VTKHDF 2025 status update; VTKHDF spec.
- `vtkio` (pure-Rust VTK I/O).
- Remacle et al., "Efficient Visualization of High Order Finite Elements," IJNME (2005).
