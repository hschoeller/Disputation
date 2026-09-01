"""Generate the panels for the dimension-heuristic figure.

Writes five bare PDFs (no titles, no captions) that are assembled
by fig_dimension.tex:
    cloud_d1.pdf, cloud_d2.pdf, cloud_d3.pdf
    kernel_sum.pdf
    slope.pdf
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.spatial.distance import pdist
from matplotlib import ticker as mticker

RNG = np.random.default_rng(3)
N_POINTS = 4000
EDGE_LENGTH = 1.0
EPSILONS = np.logspace(-7, 2, 140)
# Bandwidth for the CLOUD-PANEL SHADING ONLY -- chosen so the
# kernel is visible across the whole cloud extent. Deliberately
# NOT the same as the eps* marked on the slope panel: that one
# resolves spacing between NEIGHBOURING points (tiny, for 4000
# points in a unit cube); this one shows the kernel's shape
# relative to the cloud's CENTRE point, over the full domain.
KERNEL_BANDWIDTH_VIS = 0.05
ALPHA_FLOOR = 0.02
VIEW_ANGLES = dict(elev=18, azim=-60)
OUT_DIR = Path("fig")

# sequential blues: deliberately NOT the P1/P2/P3 deck colours
COLOURS = {1: "#74A9CF", 2: "#2B8CBE", 3: "#023858"}

base_font_size = 10  # pt

plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.texsystem'] = "pdflatex"
plt.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath,amsfonts,amssymb,cmbright,standalone}')
plt.rcParams['font.size'] = base_font_size


def sample_cloud(intrinsic_dim, n_points=N_POINTS, edge=EDGE_LENGTH):
    """Cloud of intrinsic dimension d, centred in the unit box."""
    coords = RNG.uniform(0, edge, size=(n_points, intrinsic_dim))
    embedded = np.full((n_points, 3), 0.5 * edge)
    embedded[:, :intrinsic_dim] = coords
    return embedded


def kernel_sum(cloud, epsilons):
    squared = pdist(cloud) ** 2
    return np.array(
        [2 * np.exp(-squared / eps).sum() + len(cloud)
         for eps in epsilons]
    )


def local_slope(epsilons, sums):
    return 2 * np.gradient(np.log(sums), np.log(epsilons))


def kernel_alpha_from_centre(cloud, bandwidth):
    """Diffusion-kernel value of every point relative to the
    cloud's centre point -- exactly D(bandwidth)_{centre, j}."""
    centre = np.full(3, 0.5 * EDGE_LENGTH)
    squared_distance = np.sum((cloud - centre) ** 2, axis=1)
    return np.exp(-squared_distance / bandwidth)


def write_cloud(intrinsic_dim, cloud):
    alpha = kernel_alpha_from_centre(cloud, KERNEL_BANDWIDTH_VIS)
    alpha = np.clip(alpha, ALPHA_FLOOR, 1.0)
    print(f"  d={intrinsic_dim} shading: alpha "
          f"mean={alpha.mean():.3f}, "
          f"frac>0.3={(alpha > 0.3).mean():.2f}")

    rgba = np.tile(to_rgba(COLOURS[intrinsic_dim]), (len(cloud), 1))
    rgba[:, 3] = alpha
    order = np.argsort(alpha)  # faint points drawn first

    fig = plt.figure(figsize=(1.05, 1.05))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cloud[order, 0], cloud[order, 1], cloud[order, 2],
               s=2.4, color=rgba[order], depthshade=False,
               linewidths=0)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(**VIEW_ANGLES)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(0, EDGE_LENGTH)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(OUT_DIR / f"cloud_d{intrinsic_dim}.pdf",
                transparent=True)
    plt.close(fig)


def write_kernel_sum(sums, slopes):
    fig, ax = plt.subplots(figsize=(2, 1.5))
    for d in (1, 2, 3):
        ax.loglog(EPSILONS, sums[d] / N_POINTS ** 2,
                  color=COLOURS[d], lw=1.5, label=f"$d={d}$")
        peak = int(np.argmax(slopes[d]))
        eps_star = EPSILONS[peak]
        ref = sums[d][peak] / N_POINTS ** 2
        span = np.array([eps_star / 15, eps_star * 15])
        ax.loglog(span, ref * (span / eps_star) ** (d / 2),
                  color="0.35", lw=0.8, ls="--")
    ax.axhline(1 / N_POINTS, color="0.8", lw=0.7, ls=":")
    ax.axhline(1.0, color="0.8", lw=0.7, ls=":")
    ax.set_xlabel(r"$\epsilon$", labelpad=1)
    ax.set_ylabel(r"$S(\epsilon)\, m^{-2}$", labelpad=1)
    ax.set_ylim(0.35 / N_POINTS, 4)
    # ax.legend(loc="lower right", frameon=False, handlelength=1.2,
    #           borderpad=0.2, labelspacing=0.25)
    # ax.legend_.remove()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=100.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=100.0))
    fig.savefig(OUT_DIR / "kernel_sum.pdf", bbox_inches="tight",
                pad_inches=0.02, transparent=True)
    plt.close(fig)


def write_slope(slopes):
    fig, ax = plt.subplots(figsize=(2, 1.5))
    for d in (1, 2, 3):
        ax.semilogx(EPSILONS, slopes[d], color=COLOURS[d], lw=1.5)
        ax.axhline(d, color=COLOURS[d], lw=0.7, ls=":")
        peak = int(np.argmax(slopes[d]))
        ax.plot(EPSILONS[peak], slopes[d][peak], "o", ms=3.5,
                color=COLOURS[d])
    ax.set_xlabel(r"$\epsilon$", labelpad=1)
    ax.set_ylabel(r"$2\,\partial \log S / \partial \log \epsilon$",
                  labelpad=1)
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=100.0))
    fig.savefig(OUT_DIR / "slope.pdf", bbox_inches="tight",
                pad_inches=0.02, transparent=True)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    clouds = {d: sample_cloud(d) for d in (1, 2, 3)}
    sums = {d: kernel_sum(c, EPSILONS) for d, c in clouds.items()}
    slopes = {d: local_slope(EPSILONS, s) for d, s in sums.items()}

    for d, cloud in clouds.items():
        write_cloud(d, cloud)
        peak = int(np.argmax(slopes[d]))
        print(f"d={d}: estimate {slopes[d][peak]:.2f} "
              f"at eps*={EPSILONS[peak]:.3g}")
    write_kernel_sum(sums, slopes)
    write_slope(slopes)


if __name__ == "__main__":
    main()
