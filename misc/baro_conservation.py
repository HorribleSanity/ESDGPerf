"""Baroclinic instability: energy and mass conservation time series.

Parses "Conservation step=... time=... mass=... energy=..." lines from
simulation log files (written by the DG-quadrature-weighted integrals
in baroclinicinstability.jl).  Plots relative change from t=0.

Usage:
    python baro_conservation.py <run-directory>
"""
import glob, os, re, sys
import numpy as np
import matplotlib.pyplot as plt

rundir = sys.argv[1] if len(sys.argv) > 1 else "."
SECS_PER_DAY = 86400

CONSERVATION_RE = re.compile(
    r"Conservation\s+step=(\d+)\s+time=([^\s]+)\s+mass=([^\s]+)\s+energy=([^\s]+)"
)


def parse_log(path):
    """Parse conservation lines from a log file.  Returns (days, mass, energy)."""
    steps, times, masses, energies = [], [], [], []
    with open(path) as f:
        for line in f:
            m = CONSERVATION_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                times.append(float(m.group(2)))
                masses.append(float(m.group(3)))
                energies.append(float(m.group(4)))
    assert times, f"No Conservation lines found in {path}"
    days = np.array(times) / SECS_PER_DAY
    return days, np.array(masses), np.array(energies)


# --- Discover log files ---
series = {}  # label -> (days, mass, energy)

for label in ["Float64", "Float32"]:
    path = os.path.join(rundir, f"{label}.log")
    if os.path.isfile(path):
        print(f"Reading {path}")
        series[label] = parse_log(path)

if not series:
    # Try any .log file in the run directory
    for path in sorted(glob.glob(os.path.join(rundir, "*.log"))):
        label = os.path.splitext(os.path.basename(path))[0]
        print(f"Reading {path}")
        series[label] = parse_log(path)

if not series:
    sys.exit(f"No log files with Conservation data found in {rundir}")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

colors = {"Float64": "C0", "Float32": "C1"}

for label, (days, mass, energy) in series.items():
    c = colors.get(label, "C2")

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
