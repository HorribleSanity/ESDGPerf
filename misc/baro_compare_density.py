"""Baroclinic instability Float64 vs Float32: conserved variables at z=500m.

Reads ρ and ρe directly from VTK (full conserved variables).
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

fields = {
    "ρ":  {"unit": "kg/m³", "file": "rho"},
    "ρe": {"unit": "J/m³",  "file": "rhoe"},
}


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
    """Read PVTU, slice at z_target -> (x_km, y_km, {field: values})."""
    mesh = pv.read(pvtu)
    s = mesh.slice(normal="z", origin=(0, 0, z_target))
    x = s.points[:, 0] / 1e3
    y = s.points[:, 1] / 1e3
    vals = {name: np.array(s[name]) for name in fields}
    return x, y, vals


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
    x64, y64, v64 = read_slice(f64)
    x32, y32, v32 = read_slice(f32)
    data[("Float64", day)] = (t64, x64, y64, v64)
    data[("Float32", day)] = (t32, x32, y32, v32)

    for name, cfg in fields.items():
        dd = v64[name] - v32[name]
        print(f"  {name} diff: L2={np.linalg.norm(dd):.4e}, "
              f"Linf={np.max(np.abs(dd)):.4e}, mean={np.mean(np.abs(dd)):.4e} {cfg['unit']}")

# --- One figure per field ---
for name, cfg in fields.items():
    # Auto-compute contour levels from Float64 data
    all_vals = np.concatenate([data[("Float64", d)][3][name] for d in target_days])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    pad = 0.05 * (vmax - vmin)
    vmin -= pad
    vmax += pad
    vc = 0.5 * (vmin + vmax)
    fill_levels = np.linspace(vmin, vmax, 30)
    line_levels = np.linspace(vmin, vmax, 15)
    bold_level = [vc]

    print(f"{name} contour range: [{vmin:.4e}, {vmax:.4e}] {cfg['unit']}, center={vc:.4e}")

    fig, axes = plt.subplots(len(target_days), 2, figsize=(14, 9.5),
                             squeeze=False, constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vcenter=vc, vmin=vmin, vmax=vmax)

    for row, day in enumerate(target_days):
        for col, prec in enumerate(["Float64", "Float32"]):
            t, x, y, vals = data[(prec, day)]
            ax = axes[row, col]

            ax.tricontourf(x, y, vals[name], levels=fill_levels, cmap="RdBu_r",
                           norm=norm, extend="both")
            ax.tricontour(x, y, vals[name], levels=line_levels, colors="k",
                          linewidths=0.4, alpha=0.5)
            ax.tricontour(x, y, vals[name], levels=bold_level, colors="k",
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
    fig.colorbar(sm, ax=axes, shrink=0.8, aspect=35, pad=0.02).set_label(cfg["unit"])
    fig.suptitle(f"Baroclinic Instability: {name} at z=500m — Float64 vs Float32",
                 fontsize=13, fontweight="bold")

    out = f"{rundir}/baro_compare_{cfg['file']}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()
