"""Rising bubble delta-theta at y-midplane: Float64 vs Float32 side by side.

Usage: python rb_slice.py <run-directory> [--time-end 1000]
  e.g. python rb_slice.py rb-visual-results/run_20260228_002901_job16252011_N4_L6
"""
import glob, re, sys, os
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv

# Physics constants (match risingbubble.jl)
GAMMA, CV, G, P0, TH0 = 1.4, 719.0, 9.81, 1e5, 300.0
CP = GAMMA * CV
R = CP - CV

TIME_END = 1000.0
rundir = "."
if len(sys.argv) > 1:
    rundir = sys.argv[1]
for i, a in enumerate(sys.argv):
    if a == "--time-end" and i + 1 < len(sys.argv):
        TIME_END = float(sys.argv[i + 1])


def compute_dtheta(s):
    z = s.points[:, 2]
    pi = 1 - G / (CP * TH0) * z
    rho0 = P0 / (R * TH0) * pi ** (CV / R)
    re0 = rho0 * (CV * TH0 * pi + G * z)

    rho = rho0 + np.array(s["\u03b4\u03c1"])
    ru = np.array(s["\u03b4\u03c1u"])
    rv = np.array(s["\u03b4\u03c1v"])
    rw = np.array(s["\u03b4\u03c1w"])
    re = re0 + np.array(s["\u03b4\u03c1e"])

    KE = (ru**2 + rv**2 + rw**2) / (2 * rho)
    T = (re - rho * G * z - KE) / (rho * CV)
    p = rho * R * T
    th = T * (P0 / p) ** (R / CP)
    return th - TH0


def c0_project(x, z, f):
    coords = np.round(np.column_stack([x, z]), decimals=4)
    _, _, inv = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    c = np.bincount(inv)
    return np.bincount(inv, x) / c, np.bincount(inv, z) / c, np.bincount(inv, f) / c


# Find matching saves
step_of = lambda f: int(re.search(r"step(\d+)", f).group(1))
saves64 = sorted(glob.glob(f"{rundir}/vtk-Float64/*.pvtu"), key=step_of)
saves32 = sorted(glob.glob(f"{rundir}/vtk-Float32/*.pvtu"), key=step_of)
assert len(saves64) == len(saves32), "Mismatched save counts"

# Pick every other non-zero save (t ~ 200, 400, 600, 800, 1000)
nonzero = [(a, b) for a, b in zip(saves64, saves32) if step_of(a) > 0]
picks = nonzero[1::2] if len(nonzero) > 5 else nonzero
last_step = step_of(saves64[-1])

nrows = len(picks)
fig, axes = plt.subplots(nrows, 2, figsize=(8, 3.2 * nrows),
                         squeeze=False, constrained_layout=True)

# Scan range for consistent colorbar
absmax = 0
for f64, f32 in picks:
    for pvtu in (f64, f32):
        s = pv.read(pvtu).slice(normal="y", origin=(0, 0, 1000))
        dth = compute_dtheta(s)
        absmax = max(absmax, abs(dth.min()), abs(dth.max()))
levels = np.linspace(-absmax, absmax, 41)
norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax)

# Plot
for row, (f64, f32) in enumerate(picks):
    step = step_of(f64)
    t = step / last_step * TIME_END

    for col, pvtu in enumerate((f64, f32)):
        ax = axes[row, col]
        s = pv.read(pvtu).slice(normal="y", origin=(0, 0, 1000))
        dth = compute_dtheta(s)
        x, z, dth = c0_project(s.points[:, 0], s.points[:, 2], dth)

        cf = ax.tricontourf(x, z, dth, levels=levels, cmap="RdBu_r",
                            norm=norm, extend="both")
        ax.tricontour(x, z, dth, levels=[0], colors="k", linewidths=0.4)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(0, 2000)
        ax.set_aspect("equal")

        if row == 0:
            ax.set_title("Float64" if col == 0 else "Float32", fontsize=12)
        if col == 0:
            ax.set_ylabel(f"t = {t:.0f} s\nz (m)")
        else:
            ax.set_yticklabels([])
        if row == nrows - 1:
            ax.set_xlabel("x (m)")
        else:
            ax.set_xticklabels([])

    print(f"  t = {t:.0f} s (step {step})")

cbar = fig.colorbar(cf, ax=axes, shrink=0.8, aspect=40, pad=0.02)
cbar.set_label(r"$\delta\theta$ (K)", fontsize=11)
fig.suptitle(r"Rising Bubble: $\delta\theta$ (y = 0 midplane)",
             fontsize=13, fontweight="bold")

outpath = os.path.join(rundir, "rb_dtheta_slice.png")
plt.savefig(outpath, dpi=200, bbox_inches="tight")
print(f"Saved to {outpath}")
