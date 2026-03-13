"""Baroclinic instability Float64 vs Float32: density perturbation at z=500m.

Reads δρ directly from VTK (prognostic variable, not derived).
Rows: day 0, 6, 12. Cols: Float64, Float32.

Usage: python baro_compare_density.py <run-directory>
"""
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv

rundir = sys.argv[1] if len(sys.argv) > 1 else "."
z_target = 500.0
target_days = [0, 6, 12]
SECS_PER_DAY = 86400


def read_pvd_times(vtk_dir):
    pvd_files = glob.glob(os.path.join(vtk_dir, "*.pvd"))
    assert pvd_files, f"No PVD file in {vtk_dir}"
    reader = pv.PVDReader(pvd_files[0])
    times_days = [t / SECS_PER_DAY for t in reader.time_values]
    pvtu_files = sorted(glob.glob(os.path.join(vtk_dir, "*.pvtu")))
    assert len(times_days) == len(pvtu_files), \
        f"Mismatch: {len(times_days)} times vs {len(pvtu_files)} pvtu files"
    return list(zip(pvtu_files, times_days))


def closest_to_day(saves, day):
    return min(saves, key=lambda s: abs(s[1] - day))


def read_slice(pvtu):
    """Read PVTU, slice at z_target -> (x_km, y_km, delta_rho)."""
    mesh = pv.read(pvtu)
    s = mesh.slice(normal="z", origin=(0, 0, z_target))
    x = s.points[:, 0] / 1e3
    y = s.points[:, 1] / 1e3
    drho = np.array(s["δρ"])
    return x, y, drho


# --- Discover VTK directories ---
for prefix in ["vtk-Float", "vtk-visual-Float"]:
    d64 = os.path.join(rundir, f"{prefix}64")
    d32 = os.path.join(rundir, f"{prefix}32")
    if os.path.isdir(d64) and os.path.isdir(d32):
        break
else:
    sys.exit(f"Could not find Float64/Float32 VTK dirs in {rundir}")

saves64 = read_pvd_times(d64)
saves32 = read_pvd_times(d32)

print(f"Float64: {len(saves64)} saves")
print(f"Float32: {len(saves32)} saves")

# --- Read all slices ---
data = {}
for day in target_days:
    f64, t64 = closest_to_day(saves64, day)
    f32, t32 = closest_to_day(saves32, day)
    print(f"Day {day}: F64 -> day {t64:.1f}, F32 -> day {t32:.1f}")
    x64, y64, drho64 = read_slice(f64)
    x32, y32, drho32 = read_slice(f32)
    data[("Float64", day)] = (t64, x64, y64, drho64)
    data[("Float32", day)] = (t32, x32, y32, drho32)

    dd = drho64 - drho32
    print(f"  δρ diff: L2={np.linalg.norm(dd):.4e}, "
          f"Linf={np.max(np.abs(dd)):.4e}, mean={np.mean(np.abs(dd)):.4e} kg/m³")

# --- Auto-compute contour levels from Float64 data at last day ---
all_drho = np.concatenate([data[("Float64", d)][3] for d in target_days])
vmin, vmax = np.percentile(all_drho, [1, 99])
pad = 0.05 * (vmax - vmin)
vmin -= pad
vmax += pad
vc = 0.5 * (vmin + vmax)
nfill = 30
fill_levels = np.linspace(vmin, vmax, nfill)
nlines = 15
line_levels = np.linspace(vmin, vmax, nlines)
bold_level = [vc]

print(f"Contour range: [{vmin:.5f}, {vmax:.5f}] kg/m³, center={vc:.5f}")

# --- Plot ---
fig, axes = plt.subplots(len(target_days), 2, figsize=(14, 9.5),
                         squeeze=False, constrained_layout=True)
norm = mcolors.TwoSlopeNorm(vcenter=vc, vmin=vmin, vmax=vmax)

for row, day in enumerate(target_days):
    for col, prec in enumerate(["Float64", "Float32"]):
        t, x, y, drho = data[(prec, day)]
        ax = axes[row, col]

        ax.tricontourf(x, y, drho, levels=fill_levels, cmap="RdBu_r",
                       norm=norm, extend="both")
        ax.tricontour(x, y, drho, levels=line_levels, colors="k",
                      linewidths=0.4, alpha=0.5)
        ax.tricontour(x, y, drho, levels=bold_level, colors="k",
                      linewidths=1.5)
        ax.set_aspect(2.5)

        if row == 0:
            ax.set_title(prec, fontsize=13)
        if col == 0:
            ax.set_ylabel(f"Day {t:.0f}\ny (km)")
        else:
            ax.set_yticklabels([])
        if row < len(target_days) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x (km)")

sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
fig.colorbar(sm, ax=axes, shrink=0.8, aspect=35, pad=0.02).set_label("kg/m³")
fig.suptitle("Baroclinic Instability: δρ at z=500m — Float64 vs Float32",
             fontsize=13, fontweight="bold")

out = f"{rundir}/baro_compare_density.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
plt.close()
