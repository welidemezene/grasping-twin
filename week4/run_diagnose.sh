#!/usr/bin/env bash
# The matched-pair diagnosis: why does +x 4 cm fail when -y 4 cm does not?
#
# Three runs, ~2 minutes each, and they must be read TOGETHER:
#
#   control    shift 0      — stage16_final at home. Establishes what a healthy
#                             trajectory looks like in these numbers.
#   y_neg_40   -4 cm in y   — 99.6% success. Same displacement MAGNITUDE as the
#                             failing case, so anything that differs between
#                             this run and the next is the cause.
#   x_pos_40   +4 cm in x   — 41.8% success. The run under investigation.
#
# Reading one of these alone would be the project's signature mistake for the
# fifth time: concluding from a comparison that moved more than one thing. The
# pair is the experiment; the +x run alone is an anecdote.
#
# Same container discipline as sweep_shift.sh: one offset per `docker run --rm`
# (no -d), because Isaac Lab's simulation context is a singleton. Vulkan errors
# are always present in these logs and are not a signal.
#
# Usage:  ./run_diagnose.sh [checkpoint-inside-container] [out_prefix]

set -u

CKPT=${1:-../week3/week3/checkpoints/stage16_final}
PREFIX=${2:-diag}
REPO=~/grasping_twin
IMAGE=grasping-twin-isaaclab:latest

run_in_container() {
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" "$@"
}

OFFSETS="
0.000 0.000 control
0.000 -0.040 y_neg_40mm_WORKS_99.6
0.040 0.000 x_pos_40mm_FAILS_41.8
"

echo "diagnosing $CKPT -> ${PREFIX}_summary.txt + per-step trace CSVs"
echo

printf '%s\n' "$OFFSETS" | while read -r X Y LABEL; do
    [ -z "${LABEL:-}" ] && continue
    echo "=== $LABEL  (x $X, y $Y) ==="
    run_in_container diagnose_reach.py "$CKPT" \
        --out_prefix "$PREFIX" --shift_x "$X" --shift_y "$Y" --hold 5 2>&1 \
        | grep -vE 'Warning|\[Error\]|deprecated' | tail -20
    echo
done

echo "=== ALL THREE VERDICTS ==="
grep -A1 'VERDICT' "${PREFIX}_summary.txt"
