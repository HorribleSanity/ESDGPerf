#!/bin/bash
# Verify every kernel file is correct.
# For each step: show the git commit, what changed, and diff against our file.
# Human-readable output — read it and convince yourself.

KDIR="benchmarks/roofline/kernels"
PATCH='s/@cuStaticSharedMem/CuStaticSharedArray/g; s/VolumeKernel{1}/VolumeKernel{:final}/g; s/@fastmath //g'

verify() {
    local step=$1 commit=$2 file=$3 label=$4
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP $step: $label"
    echo "  Commit: $(git log --oneline -1 $commit)"
    echo "  File:   $file"

    # What changed in this commit?
    echo "  Git diff summary:"
    git diff --stat ${commit}^..$commit -- src/kernels/volume_kernel_v1.jl src/kernels/volume_kernel_vfinal.jl 2>/dev/null | sed 's/^/    /'

    # Does our file match git + patches?
    git show ${commit}:src/kernels/volume_kernel_v1.jl 2>/dev/null \
        | sed "$PATCH" > /tmp/_verify.jl 2>/dev/null
    if [ ! -s /tmp/_verify.jl ]; then
        # Try vfinal path (steps 12-13)
        git show ${commit}:src/kernels/volume_kernel_vfinal.jl \
            | sed 's/@cuStaticSharedMem/CuStaticSharedArray/g' > /tmp/_verify.jl
    fi

    if diff -q "$KDIR/$file" /tmp/_verify.jl > /dev/null 2>&1; then
        echo "  Match:  ✓ EXACT"
    else
        echo "  Match:  ✗ MISMATCH"
        diff "$KDIR/$file" /tmp/_verify.jl | head -10
    fi
    echo ""
}

verify 00 1055c70 step00-baseline.jl      "Initial v1 (geometry preprocessing)"
verify 01 dae4515 step01-prim.jl          "Primitive variables"
verify 02 3d44b12 step02-logave.jl        "Fast logave"
verify 03 74c0441 step03-precomp-logs.jl  "Precompute logs"
verify 04 69789b2 step04-dq-not-flux.jl   "dq not flux"
verify 05 5a00ba3 step05-aux-shared.jl    "Aux in shared memory"
verify 06 ed04f58 step06-fused.jl         "Fused kernel"
verify 07 6f02047 step07-symmetry.jl      "Symmetry"
verify 08 afc3ab8 step08-inv-logave.jl    "Inv logave"
verify 09 69f47d2 step09-no-prefetch.jl   "No prefetch q2/aux2"
verify 10 3441ab9 step10-wip-agg.jl       "WIP aggregate"
verify 12 a02aafb step12-tendency-opt.jl  "Tendency optimization"
verify 13 29a5fa4 step13-ldg-const.jl     "Read-only cache (Const)"

# Step 11 is special: same kernel as step 10
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 11: contract nsz (same kernel as step 10, only contract.jl changes)"
if diff -q "$KDIR/step11-contract.jl" "$KDIR/step10-wip-agg.jl" > /dev/null; then
    echo "  Match:  ✓ IDENTICAL to step10"
else
    echo "  Match:  ✗ DIFFERS from step10"
fi
echo ""

rm -f /tmp/_verify.jl
echo "Done."
