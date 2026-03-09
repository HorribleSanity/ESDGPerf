"""Baroclinic instability Float64 vs Float32 comparison.

Pressure and temperature at z=500m. Rows: day 0, 6, 12. Cols: Float64, Float32.

Usage: python baro_compare.py <run-directory>
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
    """Read PVD via pyvista for physical times, pair with sorted pvtu files."""
    pvd_files = glob.glob(os.path.join(vtk_dir, "*.pvd"))
    assert pvd_files, f"No PVD file in {vtk_dir}"
    reader = pv.PVDReader(pvd_files[0])
    times_days = [t / SECS_PER_DAY for t in reader.time_values]
    pvtu_files = sorted(glob.glob(os.path.join(vtk_dir, "*.pvtu")))
    assert len(times_days) == len(pvtu_files), \
        f"Mismatch: {len(times_days)} times vs {len(pvtu_files)} pvtu files"
    return list(zip(pvtu_files, times_days))


def closest_to_day(saves, day):
    """Find (pvtu_path, actual_day) closest to target day."""
    return min(saves, key=lambda s: abs(s[1] - day))


def read_slice(pvtu):
    """Read PVTU, slice at z_target → (x_km, y_km, p_hPa, T_K)."""
    mesh = pv.read(pvtu)
    s = mesh.slice(normal="z", origin=(0, 0, z_target))
    x = s.points[:, 0] / 1e3
    y = s.points[:, 1] / 1e3
    p = np.array(s["pressure"]) / 100   # Pa → hPa
    T = np.array(s["temperature"])       # K
    return x, y, p, T


# --- Discover VTK directories and read times from PVD ---
for prefix in ["vtk-Float", "vtk-visual-Float"]:
    d64 = os.path.join(rundir, f"{prefix}64")
    d32 = os.path.join(rundir, f"{prefix}32")
    if os.path.isdir(d64) and os.path.isdir(d32):
        break
else:
    sys.exit(f"Could not find Float64/Float32 VTK dirs in {rundir}")

saves64 = read_pvd_times(d64)
saves32 = read_pvd_times(d32)

print(f"Float64: {len(saves64)} saves, days: {[f'{t:.1f}' for _, t in saves64]}")
print(f"Float32: {len(saves32)} saves, days: {[f'{t:.1f}' for _, t in saves32]}")

# --- Pre-read all slices (each PVTU read once) ---
data = {}
for day in target_days:
    f64, t64 = closest_to_day(saves64, day)
    f32, t32 = closest_to_day(saves32, day)
    print(f"Day {day}: F64 → day {t64:.1f} ({os.path.basename(f64)}), "
          f"F32 → day {t32:.1f} ({os.path.basename(f32)})")
    x64, y64, p64, T64 = read_slice(f64)
    x32, y32, p32, T32 = read_slice(f32)
    data[("Float64", day)] = (t64, x64, y64, p64, T64)
    data[("Float32", day)] = (t32, x32, y32, p32, T32)

    # Norms of Float64-Float32 difference (interpolate onto same grid isn't
    # needed since both meshes have identical topology — same x,y points)
    dp = p64 - p32
    dT = T64 - T32
    print(f"  Pressure diff:    L2={np.linalg.norm(dp):.4e} hPa, "
          f"Linf={np.max(np.abs(dp)):.4e} hPa, mean={np.mean(np.abs(dp)):.4e} hPa")
    print(f"  Temperature diff: L2={np.linalg.norm(dT):.4e} K,   "
          f"Linf={np.max(np.abs(dT)):.4e} K,   mean={np.mean(np.abs(dT)):.4e} K")

# --- Plot configs (Ullrich et al. Fig 5 levels) ---
field_cfgs = {
    "pressure": {
        "fill": np.arange(926, 955, 1), "lines": np.arange(926, 956, 2),
        "bold": [942], "vmin": 926, "vmax": 954, "vc": 942, "unit": "hPa",
    },
    "temperature": {
        "fill": np.arange(268, 305, 1), "lines": np.arange(268, 306, 2),
        "bold": [290], "vmin": 268, "vmax": 304, "vc": 290, "unit": "K",
    },
}

# --- Generate one figure per field ---
for name, cfg in field_cfgs.items():
    fig, axes = plt.subplots(len(target_days), 2, figsize=(14, 9.5),
                             squeeze=False, constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vcenter=cfg["vc"], vmin=cfg["vmin"], vmax=cfg["vmax"])

    for row, day in enumerate(target_days):
        for col, prec in enumerate(["Float64", "Float32"]):
            t, x, y, p, T = data[(prec, day)]
            val = p if name == "pressure" else T
            ax = axes[row, col]

            ax.tricontourf(x, y, val, levels=cfg["fill"], cmap="RdBu_r",
                           norm=norm, extend="both")
            ax.tricontour(x, y, val, levels=cfg["lines"], colors="k",
                          linewidths=0.4, alpha=0.5)
            ax.tricontour(x, y, val, levels=cfg["bold"], colors="k",
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
    fig.suptitle(f"Baroclinic Instability: {name.title()} at z=500m — Float64 vs Float32",
                 fontsize=13, fontweight="bold")

    out = f"{rundir}/baro_compare_{name}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()
