"""Compute norms of the difference between two runs at a target day."""
import glob, re, sys
import numpy as np
import pyvista as pv

# --- Settings ---
run1 = sys.argv[1]
run2 = sys.argv[2]
target_day = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0

# --- Find saves sorted by step number ---
step_of = lambda f: int(re.search(r"step(\d+)", f).group(1))

def find_day(rundir, td):
    saves = sorted(glob.glob(f"{rundir}/vtk/*.pvtu"), key=step_of)
    spd = step_of(saves[1]) - step_of(saves[0])
    day_of = lambda f: step_of(f) / spd * 0.5
    pvtu = min(saves, key=lambda f: abs(day_of(f) - td))
    return pvtu, day_of(pvtu)

f1, d1 = find_day(run1, target_day)
f2, d2 = find_day(run2, target_day)
print(f"Run 1: {f1}  (day {d1:.2f})")
print(f"Run 2: {f2}  (day {d2:.2f})")

# --- Read meshes ---
m1 = pv.read(f1)
m2 = pv.read(f2)

# Verify same grid
assert m1.n_points == m2.n_points, f"Point count mismatch: {m1.n_points} vs {m2.n_points}"
pt_err = np.max(np.abs(m1.points - m2.points))
print(f"Max point coordinate difference: {pt_err:.2e}")

# --- Compute norms ---
for name in ["pressure", "temperature"]:
    q1 = m1[name]
    q2 = m2[name]
    diff = q1 - q2
    l2_abs = np.sqrt(np.mean(diff**2))
    l2_rel = l2_abs / np.sqrt(np.mean(q1**2))
    linf_abs = np.max(np.abs(diff))
    linf_rel = linf_abs / np.max(np.abs(q1))
    print(f"\n{name}:")
    print(f"  L2   abs={l2_abs:.6e}  rel={l2_rel:.6e}")
    print(f"  Linf abs={linf_abs:.6e}  rel={linf_rel:.6e}")

