"""Pressure and temperature at day 12, z = 500 m. Compare with Ullrich et al. Fig 5.

Same as baroclinic_day12.py but samples onto a fine regular grid via pv.ImageData
to avoid DG element boundary artifacts in contour lines.
"""
import glob, re, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pyvista as pv

# --- Settings ---
rundir = sys.argv[1] if len(sys.argv) > 1 else \
    "baroclinic-results/run_20260220_233508_job16075485_Kx12_Ky2_Kz1_L3"
z_target = 500.0
target_day = 12

# --- Find saves sorted by step number ---
step_of = lambda f: int(re.search(r"step(\d+)", f).group(1))
saves = sorted(glob.glob(f"{rundir}/vtk/*.pvtu"), key=step_of)
steps_per_halfday = step_of(saves[1]) - step_of(saves[0])
day_of = lambda f: step_of(f) / steps_per_halfday * 0.5

pvtu = min(saves, key=lambda f: abs(day_of(f) - target_day))
actual_day = day_of(pvtu)

# --- Read, slice, sample onto fine regular grid ---
mesh = pv.read(pvtu)
s = mesh.slice(normal="z", origin=(0, 0, z_target))

xmin, xmax, ymin, ymax = s.bounds[0:4]
nx, ny = 4000, 600
grid = pv.ImageData(
    dimensions=(nx, ny, 1),
    spacing=((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), 1),
    origin=(xmin, ymin, s.bounds[4]),
)
grid = grid.sample(s)

x = grid.points[:, 0].reshape(ny, nx) / 1e3
y = grid.points[:, 1].reshape(ny, nx) / 1e3
p = grid["pressure"].reshape(ny, nx) / 100   # Pa -> hPa
T = grid["temperature"].reshape(ny, nx)       # K

# --- Contour specs from Ullrich et al. Fig 5 ---
fields = [
    ("(a)", p, "hPa", np.arange(926, 955, 1), np.arange(926, 956, 2), [942], 926, 954, 942),
    ("(b)", T, "K",   np.arange(268, 305, 1), np.arange(268, 306, 2), [290], 268, 304, 290),
]

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

for ax, (label, field, unit, fill_lev, line_lev, bold_lev, vmin, vmax, vc) in zip(axes, fields):
    norm = mcolors.TwoSlopeNorm(vcenter=vc, vmin=vmin, vmax=vmax)
    cf = ax.contourf(x, y, field, levels=fill_lev, cmap="RdBu_r", norm=norm, extend="both")
    ax.contour(x, y, field, levels=line_lev, colors="k", linewidths=0.4, alpha=0.5)
    ax.contour(x, y, field, levels=bold_lev, colors="k", linewidths=1.5)
    ax.set_aspect(2.5)
    ax.set_ylabel("y (km)")
    ax.text(0.005, 0.93, f"{label} Day {actual_day:.1f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))
    cb = fig.colorbar(cf, ax=ax, pad=0.01, aspect=15)
    cb.locator = mticker.MaxNLocator(nbins=6)
    cb.update_ticks()
    cb.set_label(unit, fontsize=9)

axes[0].set_xticklabels([])
axes[-1].set_xlabel("x (km)")

outpath = f"{rundir}/day{target_day}_pT_z{z_target:.0f}m_sampled.png"
plt.savefig(outpath, dpi=200)
print(f"Saved to {outpath}")
