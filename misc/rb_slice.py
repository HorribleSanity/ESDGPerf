"""Rising bubble δθ at y=0 midplane: Float64 vs Float32.

Usage: python rb_slice.py <run-directory> [--time-end 1000]
"""
import glob, re, sys, os
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import pyvista as pv

GAMMA, CV, G, P0, TH0 = 1.4, 719.0, 9.81, 1e5, 300.0
CP = GAMMA * CV
R = CP - CV

TIME_END = 1000.0
rundir = "."
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    rundir = sys.argv[1]
for i, a in enumerate(sys.argv):
    if a == "--time-end" and i + 1 < len(sys.argv):
        TIME_END = float(sys.argv[i + 1])


def compute_dtheta(mesh):
    """δθ = θ - θ₀ from conserved-variable perturbations."""
    z = mesh.points[:, 2]
    pi_ref = 1 - G / (CP * TH0) * z
    rho0 = P0 / (R * TH0) * pi_ref ** (CV / R)
    re0 = rho0 * (CV * TH0 * pi_ref + G * z)

    rho = rho0 + np.array(mesh["δρ"])
    ru = np.array(mesh["δρu"])
    rv = np.array(mesh["δρv"])
    rw = np.array(mesh["δρw"])
    re = re0 + np.array(mesh["δρe"])

    KE = (ru**2 + rv**2 + rw**2) / (2 * rho)
    T = (re - rho * G * z - KE) / (rho * CV)
    p = rho * R * T
    theta = T * (P0 / p) ** (R / CP)
    return theta - TH0


def slice_and_grid(pvtu_path, nx=800, nz=800):
    """Extract y=0 midplane and interpolate δθ onto regular grid."""
    mesh = pv.read(pvtu_path)
    mid = mesh.extract_points(np.abs(mesh.points[:, 1]) < 0.1)
    dth = compute_dtheta(mid)
    x, z = mid.points[:, 0], mid.points[:, 2]

    xi = np.linspace(-1000, 1000, nx)
    zi = np.linspace(0, 2000, nz)
    Xi, Zi = np.meshgrid(xi, zi)
    Di = griddata((x, z), dth, (Xi, Zi), method="linear")
    return Xi, Zi, Di


# --- Discover VTK directories ---
step_of = lambda f: int(re.search(r"step(\d+)", f).group(1))

for prefix in ["vtk-Float", "vtk-visual-Float"]:
    d64 = os.path.join(rundir, f"{prefix}64")
    d32 = os.path.join(rundir, f"{prefix}32")
    if os.path.isdir(d64) and os.path.isdir(d32):
        break
else:
    sys.exit(f"Could not find Float64/Float32 VTK dirs in {rundir}")

# Pair saves by sorted index (step numbers differ between precisions)
saves64 = sorted(glob.glob(f"{d64}/*.pvtu"), key=step_of)
saves32 = sorted(glob.glob(f"{d32}/*.pvtu"), key=step_of)
n_saves = min(len(saves64), len(saves32))
assert n_saves >= 2, f"Need at least 2 saves, got {n_saves}"

pairs = list(zip(saves64[:n_saves], saves32[:n_saves]))
last64, last32 = step_of(saves64[-1]), step_of(saves32[-1])

# Pick 3: IC, middle, end
if n_saves >= 3:
    pick_idx = [0, n_saves // 2, n_saves - 1]
else:
    pick_idx = list(range(n_saves))
picks = [pairs[i] for i in pick_idx]
nrows = len(picks)


def time_label(f64, f32):
    s64, s32 = step_of(f64), step_of(f32)
    if s64 == 0 and s32 == 0:
        return "0"
    t64 = s64 / last64 * TIME_END
    t32 = s32 / last32 * TIME_END
    return f"{(t64 + t32) / 2:.0f}"


# --- Global color range ---
vmin_global, vmax_global = 0.0, 0.0
for f64, f32 in picks:
    for pvtu in (f64, f32):
        _, _, Di = slice_and_grid(pvtu)
        vmin_global = min(vmin_global, np.nanmin(Di))
        vmax_global = max(vmax_global, np.nanmax(Di))

# --- Colormap: brown (neg) → gray (zero) → purple (pos) ---
cmap = mcolors.LinearSegmentedColormap.from_list("dtheta", [
    (0.00, "#C06828"),
    (0.45, "#E8D8C8"),
    (0.50, "#EAEAEA"),
    (0.55, "#DDD8E8"),
    (0.75, "#8868A8"),
    (1.00, "#180830"),
])
norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin_global, vmax=vmax_global)

pos_lines = np.arange(0.15, vmax_global + 0.01, 0.15)
neg_lines = np.arange(-0.15, vmin_global - 0.01, -0.15)[::-1]
contour_levels = np.concatenate([neg_lines, pos_lines])

# --- Plot ---
fig, axes = plt.subplots(nrows, 2, figsize=(9, 3.8 * nrows + 0.5),
                         squeeze=False, constrained_layout=True)

for row, (f64, f32) in enumerate(picks):
    t_str = time_label(f64, f32)

    for col, (pvtu, label) in enumerate([(f64, "Float64"), (f32, "Float32")]):
        ax = axes[row, col]
        Xi, Zi, Di = slice_and_grid(pvtu)

        ax.pcolormesh(Xi, Zi, Di, cmap=cmap, norm=norm, shading="auto", rasterized=True)
        ax.contour(Xi, Zi, Di, levels=contour_levels, colors="k", linewidths=0.3)

        ax.set_xlim(-1000, 1000)
        ax.set_ylim(0, 2000)
        ax.set_aspect("equal")

        if row == 0:
            ax.set_title(label, fontsize=13)
        if col == 0:
            ax.set_ylabel(f"t = {t_str} s\nz (m)")
        else:
            ax.set_yticklabels([])
        if row == nrows - 1:
            ax.set_xlabel("x (m)")
        else:
            ax.set_xticklabels([])

    print(f"  row {row}: t ≈ {t_str} s")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=35, pad=0.02)
cbar.set_label(r"$\delta\theta$ [K]")
fig.suptitle(r"Rising Bubble: $\delta\theta$ (y = 0 midplane)",
             fontsize=14, fontweight="bold")

outpath = os.path.join(rundir, "rb_dtheta_slice.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved to {outpath}")
