#!/usr/bin/env bash
# Find the knee: is there a stage 20 checkpoint that clears the gate?
#
# lift_trajectory.sh established that stage 20's lift falls MONOTONICALLY with
# training while reliability rises, at the control offset:
#
#     2.6M   74.0%   0.1029 m   39.4 deg
#     5.0M   96.5%   0.0358 m   29.6 deg
#     7.6M  100.0%   0.0226 m   20.9 deg
#      10M  100.0%   0.0257 m   20.3 deg
#
# So the run was not undertrained, it was overtrained, and ep_rew_mean climbed
# 66 -> 96.5 across exactly that span: the reward PAYS for the collapse. Training
# time is itself the tradeoff dial.
#
# That leaves a concrete question the sweep cannot answer. The gate wants >90%
# AND >10 cm. At 2.6M there is lift but not reliability (74%, 10.3 cm); by 5.0M
# there is reliability but no lift (96.5%, 3.6 cm). If the two curves cross
# anywhere, the crossing is between them, and a checkpoint that already exists
# on disk passes week 4 with no further training at all.
#
# Four checkpoints across 3.0M-4.2M, control offset only -- same discipline as
# lift_trajectory.sh, since the question is about training time and adding
# offsets would put a second variable in a probe built to isolate one. If a
# candidate emerges it gets the FULL six-offset sweep before anything is
# claimed; passing at shift 0 is not passing.
#
# Checkpoint names are the real ones on disk (stride 199,680), not round
# numbers -- guessing cost a container start once already.
#
# Usage:  ./knee_sweep.sh [out_prefix]

set -u

PREFIX=${1:-s20_knee}
REPO=/home/$USER/grasping_twin
IMAGE=grasping-twin-isaaclab:latest

for CK in 2995200 3394560 3793920 4193280; do
    echo "=== stage20_${CK}_steps ==="
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" \
        eval_shift.py "checkpoints/stage20_${CK}_steps" \
        --out_prefix "$PREFIX" --shift_x 0 --shift_y 0 --hold 5 2>&1 \
        | grep -vE 'Warning|\[Error\]|deprecated' | tail -12
    echo
done

echo "=== THE KNEE (control offset) ==="
echo "gate: >90% AND >0.10 m. stage16_final for reference: 99.4%, 0.1345 m"
cat "${PREFIX}_sweep.csv"
