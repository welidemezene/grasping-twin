#!/usr/bin/env bash
# Sweep a checkpoint across cube offsets, one container per offset.
#
# One offset per container because Isaac Lab's simulation context is a singleton
# and hangs if a second scene is built in the same process. Uses `docker run
# --rm` WITHOUT -d, which is the form that works here -- -d aborted at Vulkan
# init with exit 255 three times running in week 3.
#
# Vulkan errors in these logs are ALWAYS present and are harmless in headless
# mode. They are not a signal. Two week-3 launch failures looked like config
# problems and were both transient: retry before debugging.
#
# Usage:  ./sweep_shift.sh <checkpoint-path-inside-container> [out_prefix]
# e.g.    ./sweep_shift.sh checkpoints/stage20_final s20
#         ./sweep_shift.sh ../week3/week3/checkpoints/stage16_final s16_baseline

set -u

CKPT=${1:-checkpoints/stage20_final}
PREFIX=${2:-s20}
REPO=~/grasping_twin
IMAGE=grasping-twin-isaaclab:latest

run_in_container() {
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" "$@"
}

# x  y  label
#   0.000 is the CONTROL -- week 3's fixed point. It must not regress.
#   0.034 is the diagonal that made stage16_final grasp air on Isaac Lab 3.0.
#   0.050 sits at the training edge; 0.070 is past it, and is expected to fail.
#     A policy that passes 0.070 has learned to look; one that passes only 0.000
#     has learned a trajectory.
OFFSETS="
0.000 0.000 control
0.024 0.024 diagonal_34mm
0.040 0.000 x_40mm
0.000 -0.040 y_neg_40mm
0.035 0.035 diagonal_50mm_edge
0.050 0.050 diagonal_70mm_beyond
"

echo "sweeping $CKPT -> ${PREFIX}_sweep.csv"
echo

printf '%s\n' "$OFFSETS" | while read -r X Y LABEL; do
    [ -z "${LABEL:-}" ] && continue
    echo "=== $LABEL  (x $X, y $Y) ==="
    run_in_container eval_shift.py "$CKPT" \
        --out_prefix "$PREFIX" --shift_x "$X" --shift_y "$Y" --hold 5 2>&1 \
        | grep -vE 'Warning|\[Error\]|deprecated' | tail -15
    echo
done

echo "=== SWEEP TABLE ==="
cat "${PREFIX}_sweep.csv"
