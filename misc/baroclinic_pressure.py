"""Pressure contour plots at a horizontal z-slice, matching Ullrich et al. Fig 5."""
import glob, re, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pyvista as pv

# --- Settings ---
rundir = sys.argv[1] if len(sys.argv) > 1 else \
    "baroclinic-results/run_20260219_092850_job16021878_Kx12_Ky2_Kz1_L3"
z_target = 500.0  # meters
target_days = [8, 10, 12, 15]

# --- Find saves sorted by step number ---
step_of = lambda f: int(re.search(r"step(\d+)", f).group(1))
saves = sorted(glob.glob(f"{rundir}/vtk/*.pvtu"), key=step_of)
steps_per_halfday = step_of(saves[1]) - step_of(saves[0])
day_of = lambda f: step_of(f) / steps_per_halfday * 0.5


def read_slice(pvtu):
    """Slice a .pvtu file at z = z_target. Returns x (km), y (km), p (hPa)."""
    mesh = pv.read(pvtu)
    s = mesh.slice(normal="z", origin=(0, 0, z_target))
    return s.points[:, 0] / 1e3, s.points[:, 1] / 1e3, s["pressure"] / 100


# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, td, label in zip(axes, target_days, labels):
    pvtu = min(saves, key=lambda f: abs(day_of(f) - td))
    x, y, p = read_slice(pvtu)

    # Fixed contour levels matching Ullrich et al.: 926–954 hPa by 2, 942 enhanced
    line_levels = np.arange(926, 956, 2)
    fill_levels = np.arange(926, 955, 1)
    norm = mcolors.TwoSlopeNorm(vcenter=942, vmin=926, vmax=954)

    cf = ax.tricontourf(x, y, p, levels=fill_levels, cmap="RdBu_r", norm=norm, extend="both")
    ax.tricontour(x, y, p, levels=line_levels, colors="k", linewidths=0.4, alpha=0.5)
    ax.tricontour(x, y, p, levels=[942], colors="k", linewidths=1.5)
    ax.set_aspect("equal")
    ax.text(0.005, 0.93, f"{label} Day {day_of(pvtu):.1f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    cb = fig.colorbar(cf, ax=ax, pad=0.01, aspect=15)
    cb.locator = mticker.MaxNLocator(nbins=6)
    cb.update_ticks()
    cb.set_label("hPa", fontsize=9)

# Only label bottom x-axis
for ax in axes[:-1]:  ax.set_xticklabels([])
axes[-1].set_xlabel("x (km)")
for ax in axes:  ax.set_ylabel("y (km)")

outpath = f"{rundir}/pressure_z{z_target:.0f}m.png"
plt.savefig(outpath, dpi=200)
print(f"Saved to {outpath}")
