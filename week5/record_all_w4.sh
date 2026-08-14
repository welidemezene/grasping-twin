#!/usr/bin/env bash
# Record every policy the viewer needs, at two cube offsets.
#
# ONE CONTAINER PER RECORDING. Isaac Lab's simulation context is a singleton and
# hangs if a second scene is built in the same process, so a loop inside one
# container does not work. This is the same shape as week4/sweep_shift.sh.
#
# TWO OFFSETS, and the pair is the argument:
#
#   control (0, 0)        week 3's fixed point. Every policy should look fine
#                         here -- stage 16 scores 99.6%. A viewer who only sees
#                         this offset learns nothing, which is exactly why the
#                         old grid videos were unconvincing.
#   5 cm diagonal         where stage 16 collapses to 41.0% and stage 22 holds
#   (0.035, 0.035)        93.2%. Same cube position, both policies. This is the
#                         shot that shows one arm going where the cube USED to be.
#
# FOUR POLICIES, in the order the story runs:
#   stage16  perfect and blind        (week 3's result)
#   stage20  reads the cube, no lift  (the cost)
#   stage22  reads the cube AND lifts (week 4's result)
#   stage23  the 3M peak; its _final grasps 0/512 and is deliberately not here
#
# Usage:  ./record_all_w4.sh            # both offsets, all four policies
#         ./record_all_w4.sh control    # one offset only
#
# Vulkan errors in the logs are always present in headless mode and are not a
# signal. Retry a failed launch once before debugging it.

set -u

REPO=~/grasping_twin
IMAGE=grasping-twin-isaaclab:latest
OUTDIR=/workspace/week5/motion
ONLY=${1:-all}

run_in_container() {
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" "$@"
}

# label|checkpoint path inside the container|short name for the viewer
#
# stage 16 lives under week3/week3/ -- the checkpoint directory is doubled in
# this repo, and week4/sweep_shift.sh uses the same doubled path.
POLICIES="
s16|../week3/week3/checkpoints/stage16_final|stage 16
s20|checkpoints/stage20_final|stage 20
s22|checkpoints/stage22_final|stage 22
s23|checkpoints/stage23_2995200_steps|stage 23 @ 3M
"

# tag|shift_x|shift_y
OFFSETS="
control|0.000|0.000
diag50|0.035|0.035
"

mkdir -p "$REPO/week5/motion"

printf '%s\n' "$OFFSETS" | while IFS='|' read -r TAG X Y; do
    [ -z "${TAG:-}" ] && continue
    [ "$ONLY" != "all" ] && [ "$ONLY" != "$TAG" ] && continue

    printf '%s\n' "$POLICIES" | while IFS='|' read -r NAME CKPT LABEL; do
        [ -z "${NAME:-}" ] && continue
        OUT="$OUTDIR/${NAME}_${TAG}.json"
        echo "=== $LABEL  @  $TAG (x $X, y $Y) ==="
        run_in_container ../week5/record_motion_w4.py "$CKPT" "$OUT" \
            --shift_x "$X" --shift_y "$Y" --hold 5 --label "$LABEL" 2>&1 \
            | grep -vE 'Warning|\[Error\]|deprecated|Vulkan' | tail -14
        echo
    done
done

echo "=== RECORDED ==="
ls -la "$REPO/week5/motion/" 2>/dev/null | grep '\.json$'
echo
echo "Reminder: these are single episodes. Every success RATE comes from"
echo "week4/*_sweep.csv, 512 trials each. Do not caption a clip with a number"
echo "the sweep does not say."
