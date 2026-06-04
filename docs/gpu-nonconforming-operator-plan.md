# AMR-with-GPU-flow: GPU non-conforming SIPG operator — implementation plan

**Status:** in progress (scoping + design complete 2026-06-04). The GPU SIPG operators
currently `unreachable!()` on `Neighbor::{CoarseToFine, FineToCoarse}` — i.e. GPU flow
runs only on uniform conforming meshes. The CPU stack already solves on 2:1
non-conforming (mortar) meshes (`Poisson`/`Stokes`, validated; see
`docs/mesh-and-adaptivity-strategy.md` "As-built" sections). This plan ports that to
the GPU so adaptive flow runs on-device.

## Goal & sequencing
1. **GPU NC Poisson operator-apply** (this increment): `operator` kernel handles 2:1
   faces; validated vs CPU `Poisson::apply` on a refined `Mesh2d` (symmetry + MMS).
2. NC CG / Helmholtz / deflated-pressure (reuse the operator; the CG vector ops are
   mesh-agnostic) — validated vs CPU `cg`/`cg_deflated`.
3. `GpuStokes` / `GpuDualSplitting` on refined meshes (the host assembly — divergence,
   gradient correction — is already element-local; only the elliptic solves change).
4. 3D non-conforming (after the 2D path; needs `Mesh3d` octree NC + hex mortar).

## What the CPU does (the reference, `Poisson::apply`)
Per 2:1 interface, SIPG is integrated on the **fine mortar**. Mortar operators
(`RefineQuad`, fixed (p+1)×(p+1) per half):
- `P = axis(half)` (`mortar_to_fine`): coarse edge trace → fine half trace.
- `Pᵀ` (`mortar_gather`): fine-mortar weighted contributions → coarse test nodes
  (adjoint of `P`, no mass/Jacobian — keeps the operator symmetric).

**CoarseToFine** (coarse element `e`, 2 fine halves `h`): for each fine node, form
`jump = (P·u_c)[i] − u_f[i]`, `avg = ½((P·∂ₙu_c)[i] + ∂ₙu_f[i])`; the fine-test
consistency/penalty go **direct** to `r[fine]`, the coarse-test ones are `Pᵀ`-gathered
to `r[coarse]`; symmetry-lift sources `g = ½ sw·jump` go to both sides' `H` then through
`gradᵀ` (the lift). **FineToCoarse** is the same interface seen from the fine side.

## GPU encoding (gather form — each block computes its own element's `r`)
The conforming kernel already does: per face node → read neighbor `u`/grad → `avg,jump`
→ accumulate `RF` (consistency+penalty) and `H` (lift sources) → subtract lift into
`PR/PS` → volume sum-fac. NC extends this:

- **Upload** the two mortar matrices `P0 = axis(0)`, `P1 = axis(1)` ((p+1)² each) as
  device constants (one per refinement; same for all faces of an order).
- **`flatten_mesh`**: tag each face with a type {Conforming(BND/NEU/interior), CoarseToFine,
  FineToCoarse} + the neighbor element id(s) + the mortar `half`. For NC faces, store the
  face-node→global-node maps for self and neighbor(s) and the `sorted` ordering the CPU
  mortar assumes (carefully reproduce `sorted(e,edge)`).
- **Kernel branches** in the thread-0 face loop (`n1` face nodes per edge):
  - *FineToCoarse* (fine self, coarse neighbor): for each face node `i`, compute
    `(P_half · u_coarse_trace)[i]` and `(P_half · ∂ₙu_coarse_trace)[i]` by a length-`n1`
    dot over the coarse trace (read coarse `u`,`gx`,`gy`), then accumulate `RF[vl]`,
    `H[vl]` exactly as the conforming branch but with the projected coarse values.
  - *CoarseToFine* (coarse self, 2 fine neighbors): for each half, for each fine node,
    compute the per-fine quantity (using `P_half·u_c` for the coarse trace), then
    `Pᵀ`-gather (`Σ_i P_half[i,j]·q_i`) into `RF[coarse node j]` and `H[coarse node j]`.
    This is the only branch that needs a gather matmul into the coarse nodes.
  - The lift (`PR/PS -= metric·H`) and volume sum-fac are unchanged — NC just feeds
    different `RF`/`H`.
- Symmetry is preserved iff the same `P`/`Pᵀ` pair is used for project/gather (mirror
  the CPU exactly).

## Validation plan
- Build a refined `Mesh2d` (`cartesian_refined` with a couple of cells refined).
- `gpu-poisson-nc-check`: (a) operator symmetry `⟨Au,v⟩=⟨Av,u⟩` < 1e-9; (b) `A·u` vs CPU
  `Poisson::apply` on a random `u` to ~1e-13; (c) conforming meshes stay bit-identical
  (NC branches never taken). Then a manufactured solve via the NC CG vs CPU.

## Risks / notes
- The `sorted()` face-node ordering and the per-half `axis()` matrix must match the CPU
  exactly or symmetry/accuracy breaks — copy the CPU indexing verbatim.
- Keep kernel signature narrow (pack metrics/face data as in `poisson3d.rs`); the mortar
  matrices add 2 small array params.
- The conforming fast-path must remain bit-identical (gate NC work behind the face-type
  tag; `flatten_mesh` of a conforming mesh emits no NC faces).
- This is the 2D path; 3D NC additionally needs `Mesh3d` octree non-conforming
  connectivity (currently conforming-only) before a GPU 3D NC operator.
