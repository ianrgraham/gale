#!/usr/bin/env python3
"""View a gale HDF5 trajectory (see gale-traj). Renders to image files (headless-friendly):
a montage of snapshots and/or an animated GIF. Each high-order DG element is tessellated
*independently* into sub-triangles, so inter-element jumps render faithfully.

Usage:
  python view_traj.py traj.h5 [--field u] [--comp 0|mag] [--out PREFIX]
                              [--snapshots N] [--gif] [--fps F] [--cmap NAME]
"""
import argparse
import glob
import os
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: render to files, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.tri import Triangulation


def open_set(path):
    """Open a trajectory as an ordered list of HDF5 handles. Accepts a single `.h5`, or any member
    of a split set `<stem>.NNNN.h5` (or the bare `<stem>`); returns (handles, stem)."""
    m = re.match(r"(.*)\.\d{4}\.h5$", path)
    if m:
        stem = m.group(1)
        files = sorted(glob.glob(f"{stem}.[0-9][0-9][0-9][0-9].h5"))
    elif path.endswith(".h5") and os.path.exists(path):
        return [h5py.File(path, "r")], path[:-3]
    else:
        stem = path[:-3] if path.endswith(".h5") else path
        files = sorted(glob.glob(f"{stem}.[0-9][0-9][0-9][0-9].h5"))
    if not files:
        raise SystemExit(f"no trajectory file(s) found for {path!r}")
    return [h5py.File(p, "r") for p in files], stem


def read_radii(f):
    """Static rigid-body radii, or None if the trajectory has no bodies."""
    if "bodies" in f and "radius" in f["bodies"]:
        return f["bodies/radius"][:]
    return None


def frame_poses(f, fr):
    """This frame's body poses [nbody, 3] = (cx, cy, phi), or None."""
    g = f[f"frames/{fr}"]
    return g["body_pose"][:] if "body_pose" in g else None


def draw_bodies(ax, poses, radii):
    """Draw each rigid body as a white circle outline + a radial orientation tick; return artists."""
    artists = []
    if poses is None or radii is None:
        return artists
    for i, (cx, cy, phi) in enumerate(poses):
        rad = float(radii[i] if i < len(radii) else radii[0])
        circ = mpatches.Circle((cx, cy), rad, fill=False, edgecolor="white", linewidth=1.6)
        ax.add_patch(circ)
        (tick,) = ax.plot([cx, cx + rad * np.cos(phi)], [cy, cy + rad * np.sin(phi)],
                          color="white", linewidth=1.0)
        artists += [circ, tick]
    return artists


def build_triangulation(f, tid):
    """Per-element sub-triangulation for topology `tid`. DG fields are discontinuous, so each
    element is tessellated on its own nodes (no inter-element vertex sharing)."""
    nodes = f[f"topology/{tid}/elem_nodes"][:]          # [ne, nn, 2]
    ne, nn, _ = nodes.shape
    n1 = int(round(nn ** 0.5))                          # p+1 (tensor-product quad)
    x = nodes[:, :, 0].reshape(-1)
    y = nodes[:, :, 1].reshape(-1)
    tris = []
    for e in range(ne):
        base = e * nn
        for b in range(n1 - 1):
            for a in range(n1 - 1):
                k00 = base + b * n1 + a
                k10 = base + b * n1 + a + 1
                k01 = base + (b + 1) * n1 + a
                k11 = base + (b + 1) * n1 + a + 1
                tris.append([k00, k10, k11])
                tris.append([k00, k11, k01])
    return Triangulation(x, y, np.asarray(tris, dtype=np.int64)), ne, nn


def frame_values(f, fr, field, comp):
    d = f[f"frames/{fr}/{field}"][:]                    # [ne, nn, ncomp]
    if comp == "mag":
        return np.sqrt((d.astype(np.float64) ** 2).sum(axis=2)).reshape(-1)
    return d[:, :, int(comp)].reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--field", default="u")
    ap.add_argument("--comp", default="0", help="component index, or 'mag'")
    ap.add_argument("--out", default=None, help="output prefix (default: input path stem)")
    ap.add_argument("--snapshots", type=int, default=6)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()

    handles, stem = open_set(args.path)
    prefix = args.out or stem
    order = int(handles[0]["ref"].attrs["order"])
    radii = read_radii(handles[0])
    # Global ordered frame list spanning the whole file set: (file_index, frame_group_key).
    frames = [(hi, k) for hi, h in enumerate(handles) for k in sorted(h["frames"].keys())]
    nfr = len(frames)
    print(f"{nfr} frames across {len(handles)} file(s), p={order}, field='{args.field}' comp={args.comp}")

    # Cache triangulations by (file index, topology id) — topologies are re-emitted per file.
    tri_cache = {}
    def tri_for(hi, fr):
        tid = int(handles[hi][f"frames/{fr}"].attrs["topology"])
        key = (hi, tid)
        if key not in tri_cache:
            tri_cache[key] = build_triangulation(handles[hi], tid)
        return tri_cache[key]

    def values(i):
        hi, fr = frames[i]
        return frame_values(handles[hi], fr, args.field, args.comp)

    def time_of(i):
        hi, fr = frames[i]
        return float(handles[hi][f"frames/{fr}"].attrs["time"])

    def poses(i):
        hi, fr = frames[i]
        return frame_poses(handles[hi], fr)

    # Global color range across all frames (consistent scale); NaN = masked (e.g. inside a body).
    vmin, vmax = np.inf, -np.inf
    for i in range(nfr):
        v = values(i)
        if np.isfinite(v).any():
            vmin = min(vmin, np.nanmin(v))
            vmax = max(vmax, np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-12

    # ---- Montage of N evenly-spaced snapshots ----
    n = min(args.snapshots, nfr)
    idx = np.linspace(0, nfr - 1, n).round().astype(int)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    tpc = None
    for k, i in enumerate(idx):
        hi, fr = frames[i]
        tri, _, _ = tri_for(hi, fr)
        ax = axes.flat[k]
        ax.axis("on")
        tpc = ax.tripcolor(tri, values(i), shading="gouraud", cmap=args.cmap, vmin=vmin, vmax=vmax)
        draw_bodies(ax, poses(i), radii)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={time_of(i):.3f}", fontsize=10)
    fig.colorbar(tpc, ax=axes.ravel().tolist(), shrink=0.8, label=f"{args.field}[{args.comp}]")
    montage = f"{prefix}_montage.png"
    fig.savefig(montage, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {montage}")

    # ---- Optional animated GIF ----
    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        figa, axa = plt.subplots(figsize=(5, 5))
        hi0, fr0 = frames[0]
        tri0, _, _ = tri_for(hi0, fr0)
        coll = axa.tripcolor(tri0, values(0), shading="gouraud", cmap=args.cmap, vmin=vmin, vmax=vmax)
        axa.set_aspect("equal"); axa.set_xticks([]); axa.set_yticks([])
        figa.colorbar(coll, ax=axa, shrink=0.8, label=f"{args.field}[{args.comp}]")
        ttl = axa.set_title("")
        body_artists = draw_bodies(axa, poses(0), radii)
        cur_tri = [(hi0, int(handles[hi0][f"frames/{fr0}"].attrs["topology"]))]

        step = max(1, nfr // 200)  # subsample to keep the GIF light
        anim_frames = list(range(0, nfr, step))

        def update(i):
            nonlocal body_artists
            hi, fr = frames[i]
            tid = int(handles[hi][f"frames/{fr}"].attrs["topology"])
            if (hi, tid) != cur_tri[0]:
                # topology changed (new file / AMR remesh) — rebind the mesh
                tri, _, _ = tri_for(hi, fr)
                coll.set_triangulation(tri) if hasattr(coll, "set_triangulation") else None
                cur_tri[0] = (hi, tid)
            coll.set_array(values(i))  # gouraud TriMesh: per-vertex colors
            for a in body_artists:
                a.remove()
            body_artists = draw_bodies(axa, poses(i), radii)
            ttl.set_text(f"t={time_of(i):.3f}")
            return [coll, ttl, *body_artists]

        anim = FuncAnimation(figa, update, frames=anim_frames, blit=False)
        gif = f"{prefix}.gif"
        anim.save(gif, writer=PillowWriter(fps=args.fps))
        plt.close(figa)
        print(f"wrote {gif} ({len(anim_frames)} frames)")


if __name__ == "__main__":
    main()
