"""Baroclinic instability: energy and mass conservation time series.

Integrates total mass (∫ρ dV) and total energy (∫ρe dV) over the full 3D
domain at each output timestep.  Plots relative change from t=0.

Supports a single VTK directory or Float64/Float32 side-by-side.

Usage:
    python baro_conservation.py <run-directory>
    python baro_conservation.py <vtk-directory>   # single precision
"""
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

rundir = sys.argv[1] if len(sys.argv) > 1 else "."
SECS_PER_DAY = 86400


def read_pvd_times(vtk_dir):
    """Return [(pvtu_path, time_days), ...] sorted by time."""
    pvd_files = glob.glob(os.path.join(vtk_dir, "*.pvd"))
    assert pvd_files, f"No PVD file in {vtk_dir}"
    reader = pv.PVDReader(pvd_files[0])
    times_days = [t / SECS_PER_DAY for t in reader.time_values]
    pvtu_files = sorted(glob.glob(os.path.join(vtk_dir, "*.pvtu")))
    assert len(times_days) == len(pvtu_files), \
        f"Mismatch: {len(times_days)} times vs {len(pvtu_files)} pvtu files"
    return list(zip(pvtu_files, times_days))


def integrate_fields(pvtu_path):
    """Integrate ρ and ρe over full 3D domain.  Returns (mass, energy)."""
    mesh = pv.read(pvtu_path)
    result = mesh.integrate_data()
    mass = result["ρ"][0]
    energy = result["ρe"][0]
    return mass, energy


def compute_series(saves):
    """Return arrays: days, mass, energy for a list of (pvtu, day) saves."""
    days = np.empty(len(saves))
    mass = np.empty(len(saves))
    energy = np.empty(len(saves))
    for i, (pvtu, day) in enumerate(saves):
        print(f"  [{i+1}/{len(saves)}] day {day:.2f} ... ", end="", flush=True)
        m, e = integrate_fields(pvtu)
        days[i] = day
        mass[i] = m
        energy[i] = e
        print(f"mass={m:.10e}  energy={e:.10e}")
    return days, mass, energy


# --- Discover VTK directories ---
series = {}  # label -> (days, mass, energy)

# Try Float64/Float32 pair first
for prefix in ["vtk-Float", "vtk-visual-Float"]:
    d64 = os.path.join(rundir, f"{prefix}64")
    d32 = os.path.join(rundir, f"{prefix}32")
    if os.path.isdir(d64) and os.path.isdir(d32):
        for label, d in [("Float64", d64), ("Float32", d32)]:
            print(f"\n--- {label}: {d} ---")
            saves = read_pvd_times(d)
            series[label] = compute_series(saves)
        break
else:
    # Try single directory (rundir itself or first vtk-* subdir)
    candidates = [rundir] + sorted(glob.glob(os.path.join(rundir, "vtk-*")))
    for d in candidates:
        if glob.glob(os.path.join(d, "*.pvd")):
            label = os.path.basename(d) or "data"
            print(f"\n--- {label}: {d} ---")
            saves = read_pvd_times(d)
            series[label] = compute_series(saves)
            break
    else:
        sys.exit(f"No VTK output found in {rundir}")


# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

colors = {"Float64": "C0", "Float32": "C1"}

for label, (days, mass, energy) in series.items():
    c = colors.get(label, "C2")

    # Relative change from initial
    dm = (mass - mass[0]) / abs(mass[0])
    de = (energy - energy[0]) / abs(energy[0])

    axes[0].plot(days, dm, "o-", color=c, markersize=3, label=label)
    axes[1].plot(days, de, "o-", color=c, markersize=3, label=label)

axes[0].set_ylabel("$(M - M_0) / |M_0|$")
axes[0].set_title("Total Mass Conservation")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

axes[1].set_ylabel("$(E - E_0) / |E_0|$")
axes[1].set_title("Total Energy Conservation")
axes[1].set_xlabel("Time (days)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

fig.suptitle("Baroclinic Instability: Conservation Diagnostics", fontsize=13, fontweight="bold")

out = os.path.join(rundir, "baro_conservation.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved {out}")
plt.close()
