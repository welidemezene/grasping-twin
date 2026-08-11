#!/usr/bin/env bash
# Where does the time go: stage 16 vs stage 22, phase by phase.
#
# The matched pair is the experiment. Stage 22 shakes worst (0.2759 arm delta
# per step vs stage 16's 0.1211), but the whole-episode number cannot say WHERE
# the shake lives, and the footage suggests it is concentrated at the grab. Two
# runs of probe_phases.py answer that:
#
#   s16   the fixed-spawn policy, boolean _held. Week 3's result.
#   s22   the randomized-spawn policy, smooth _held. Week 4's result.
#
# Both are replayed under FrankaLiftStage20Cfg, so the SCENE is identical and
# the only difference is the policy. Reading s22 alone would say "it dwells 30
# steps" without the one fact that makes that a finding or a non-finding:
# whether stage 16 dwelt 30 steps too. Same mistake this project has made four
# times; do not make it a fifth.
#
# ~2 minutes per run on existing checkpoints. NOTHING IS TRAINED HERE.
#
# One `docker run --rm` per policy (no -d): Isaac Lab's simulation context is a
# singleton, so two probes cannot share a container. Vulkan errors are always
# present in these logs and are not a signal.
#
# Usage:  ./run_phases.sh

set -u

REPO=~/grasping_twin
IMAGE=grasping-twin-isaaclab:latest

run_in_container() {
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" "$@"
}

# label -> checkpoint. Note the doubled week3 path: that is where week 3's
# checkpoints actually live in the WSL copy, not a typo.
POLICIES="
s16 ../week3/week3/checkpoints/stage16_final
s22 checkpoints/stage22_final
"

printf '%s\n' "$POLICIES" | while read -r LABEL CKPT; do
    [ -z "${LABEL:-}" ] && continue
    echo "=== $LABEL  ($CKPT) ==="
    run_in_container probe_phases.py "$CKPT" "$LABEL" --hold 5 2>&1 \
        | grep -vE 'Warning|\[Error\]|deprecated' | tail -30
    echo
done

echo "=== SIDE BY SIDE ==="
for f in phases_s16.txt phases_s22.txt; do
    [ -f "$f" ] && grep -E 'DWELL|LIFT_LAG|median|exactly 1' "$f" | sed "s|^|$f  |"
done
