#!/usr/bin/env python3
"""Spectral validation diagnostic for viscoelastic-turbulence trajectories.

Computes, from a `.h5` trajectory (the DG/AMR fields written by `TrajectoryWriter`):
  - the kinetic-energy spectrum  E(k)  from the velocity field `u`, and
  - the conformation-trace spectrum  Sigma(k)  from `trC`,
then fits the log-log power-law slope of each in a chosen k-band.

The #1 elastic-turbulence validation benchmark (deep-research verdict): a STEEP velocity
spectrum  E(k) ~ k^-alpha  with  alpha > 3  (Berti & Boffetta 2008/2010, 2D Kolmogorov
Oldroyd-B: k^-3.8; literature 3.5-3.8) is the discriminating signature of ET — a spatially
smooth, large-scale-dominated (Batchelor-regime) flow, vs a shallow inertial/EIT spectrum.
Secondary: the conformation-trace spectrum Sigma(k) ~ k^-2 (Garg et al. 2018).

Each DG field is per-element discontinuous on an AMR mesh; we interpolate the scattered nodal
values onto a uniform grid (Delaunay + linear, matplotlib.tri — no scipy needed), Hann-window
to suppress wall/non-periodic leakage, FFT, and radially bin. Spectra are averaged over the
selected (developed-phase) frames for statistics, since ET is statistically steady.

Usage:
  python3 spectrum.py /tmp/ko_fine.h5 [--grid 256] [--last 150] [--kfit 4 32] [--out /tmp/ko_spec.png]
"""
import argparse
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation


def load_handles(path):
    if os.path.exists(path):
        return [h5py.File(path, "r")]
    stem = path[:-3] if path.endswith(".h5") else path
    files = sorted(glob.glob(stem + ".[0-9]*.h5"))
    if not files:
        raise FileNotFoundError(path)
    return [h5py.File(p, "r") for p in files]


def frames(handles):
    return [(hi, k) for hi, h in enumerate(handles) for k in sorted(h["frames"].keys())]


def interp_to_grid(coords, vals, n, box):
    """Scattered DG nodes (coords [P,2], vals [P]) -> uniform n x n grid over `box`=(x0,x1,y0,y1).
    Dedup coincident shared-corner nodes, Delaunay-triangulate, linear-interpolate; outside-hull -> 0."""
    x, y = coords[:, 0], coords[:, 1]
    _, idx = np.unique(np.round(np.c_[x, y], 9), axis=0, return_index=True)
    tri = Triangulation(x[idx], y[idx])
    interp = LinearTriInterpolator(tri, vals[idx])
    gx = box[0] + (np.arange(n) + 0.5) * (box[1] - box[0]) / n
    gy = box[2] + (np.arange(n) + 0.5) * (box[3] - box[2]) / n
    gx2, gy2 = np.meshgrid(gx, gy)
    return np.ma.filled(interp(gx2, gy2), 0.0)


def radial_power(grids):
    """Sum the radial power spectrum over a list of [n,n] real fields (fluctuations: mean removed,
    Hann-windowed). Returns (k_centers, shell-summed power P(k)) with integer wavenumbers."""
    n = grids[0].shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    wcorr = (win ** 2).mean()
    power = np.zeros((n, n))
    for f in grids:
        ff = np.fft.fft2((f - f.mean()) * win)
        power += (np.abs(ff) ** 2)
    power /= (n * n) ** 2 * wcorr
    kf = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(kf, kf)
    kmag = np.sqrt(kx ** 2 + ky ** 2)
    kmax = n // 2
    edges = np.arange(0.5, kmax + 1.0, 1.0)
    kcen = 0.5 * (edges[1:] + edges[:-1])
    pk = np.array([power[(kmag >= edges[i]) & (kmag < edges[i + 1])].sum() for i in range(len(kcen))])
    return kcen, pk


def fit_slope(k, e, klo, khi):
    m = (k >= klo) & (k <= khi) & (e > 0)
    if m.sum() < 3:
        return float("nan"), m
    c = np.polyfit(np.log(k[m]), np.log(e[m]), 1)
    return c[0], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--grid", type=int, default=256, help="uniform FFT grid (per side)")
    ap.add_argument("--last", type=int, default=0, help="average over the last N frames (0 => 2nd half)")
    ap.add_argument("--kfit", type=float, nargs=2, default=[4.0, 32.0], help="k-band for the slope fit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    handles = load_handles(args.file)
    fr = frames(handles)
    n = args.last if args.last > 0 else max(1, len(fr) // 2)
    sel = fr[-n:]
    order = int(handles[0]["ref"].attrs["order"])
    box = None
    t0 = float(handles[sel[0][0]][f"frames/{sel[0][1]}"].attrs["time"])
    t1 = float(handles[sel[-1][0]][f"frames/{sel[-1][1]}"].attrs["time"])
    print(f"{len(fr)} frames, p={order}; averaging spectra over {len(sel)} developed-phase frames "
          f"(t={t0:.2f}..{t1:.2f})  grid={args.grid}^2")

    u_grids, c_grids = [], []
    for hi, k in sel:
        h = handles[hi]
        tid = int(h[f"frames/{k}"].attrs["topology"])
        coords = h[f"topology/{tid}/elem_nodes"][:].reshape(-1, 2)
        if box is None:
            box = (coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max())
        u = h[f"frames/{k}/u"][:].reshape(-1, 2)
        u_grids.append((interp_to_grid(coords, u[:, 0], args.grid, box),
                        interp_to_grid(coords, u[:, 1], args.grid, box)))
        if "trC" in h[f"frames/{k}"]:
            c = h[f"frames/{k}/trC"][:].reshape(-1)
            c_grids.append(interp_to_grid(coords, c, args.grid, box))

    # E(k): kinetic-energy spectrum = 1/2 * shell-summed |u_hat|^2 (both components), frame-averaged.
    ke = []
    for gx, gy in u_grids:
        kc, p = radial_power([gx, gy])
        ke.append(0.5 * p)
    kcen = kc
    Ek = np.mean(ke, axis=0)

    a_u, mu = fit_slope(kcen, Ek, *args.kfit)
    print(f"  velocity E(k): slope alpha = {-a_u:.2f}  (fit k in [{args.kfit[0]:.0f},{args.kfit[1]:.0f}])"
          f"   [ET target: alpha > 3, ~3.5-3.8]")

    have_c = len(c_grids) > 0
    if have_c:
        sig = []
        for g in c_grids:
            _, p = radial_power([g])
            sig.append(p)
        Sig = np.mean(sig, axis=0)
        a_c, mc = fit_slope(kcen, Sig, *args.kfit)
        print(f"  trC Sigma(k): slope delta = {-a_c:.2f}  [target ~2]")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(kcen, Ek / Ek[mu][0] if mu.any() else Ek, "o-", ms=3, label=f"E(k) velocity  (α≈{-a_u:.2f})")
    if have_c:
        ax.loglog(kcen, Sig / Sig[mc][0] if mc.any() else Sig, "s-", ms=3, alpha=0.7,
                  label=f"Σ(k) trC  (δ≈{-a_c:.2f})")
    kref = kcen[(kcen >= args.kfit[0]) & (kcen <= args.kfit[1])]
    if len(kref):
        ax.loglog(kref, (kref / kref[0]) ** -3.8, "k--", lw=1, label="k$^{-3.8}$ (Berti&Boffetta)")
        ax.loglog(kref, (kref / kref[0]) ** -3.0, "k:", lw=1, label="k$^{-3}$ (ET threshold)")
    ax.axvspan(args.kfit[0], args.kfit[1], color="gray", alpha=0.08)
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("normalized spectrum")
    ax.set_title(f"Viscoelastic Kolmogorov spectra ({os.path.basename(args.file)})")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.2)
    out = args.out or (os.path.splitext(args.file)[0] + "_spectrum.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
