#!/usr/bin/env bash
# Is stage 20's collapsed lift UNDERTRAINING or the tradeoff reasserting itself?
#
# Stage 20 solved generalization outright -- +x 4 cm went 41.8% -> 98.6%, the
# 5 cm diagonal 41.0% -> 99.8% -- and simultaneously dropped the median lift
# from stage 16's 0.1345 m to 0.0257 m, at every offset INCLUDING the unshifted
# control. So the warm start did not merely fail to extend the lift; it unlearned
# it at the very point stage 16 had mastered.
#
# Two explanations survive the sweep, and they are not distinguishable from it:
#
#   UNDERTRAINED -- ep_rew_mean was still climbing steeply at the 10M cutoff
#       (84 -> 96.5 over the last 2M). Stage 13 had this same signature. If the
#       lift is on its way back, more steps are the cheapest possible fix.
#
#   THE TRADEOFF -- randomizing the spawn makes a low, square, guaranteed hold
#       the safe earner, exactly the mechanism week 3 documented for the
#       cube_upright term: it pays at ANY height while lifting_progress needs
#       height, so hovering-at-rest beats gambling on a lift. Stage 20's numbers
#       (100% success, 2.6 cm, 20 deg) are stage 18/19's signature to the digit.
#
# The discriminator is the SHAPE OF THE LIFT ACROSS TRAINING, at the control
# offset where stage 16's 0.1345 m is the known reference:
#
#   rising   -> undertrained. Continue the run; do not touch the reward.
#   falling  -> the tradeoff. More steps make it worse, and the next stage has
#               exactly one suspect. Do NOT spend 45 minutes finding that out.
#   flat low -> it never had the lift after the warm start, which points at the
#               learning rate or ent_coef disturbing stage 16's solution early.
#
# Control offset only. The question is about the lift over time, not about
# generalization -- that one the sweep already answered, and adding offsets here
# would be a second variable in a probe that exists to isolate one.
#
# Usage:  ./lift_trajectory.sh [out_prefix]

set -u

PREFIX=${1:-s20_traj}
REPO=/home/$USER/grasping_twin
IMAGE=grasping-twin-isaaclab:latest

# Roughly every 2.5M steps. The checkpoint stride is 199,680 (save_freq
# 200000/512, rounded), which does not divide evenly into these targets, so the
# nearest existing filename is resolved by pattern -- guessing one costs a
# container start to discover it was wrong.
for TARGET in 2500000 5000000 7500000; do
    CK=$(ls "$REPO"/week4/checkpoints/ \
         | sed 's/stage20_\([0-9]*\)_steps.zip/\1/' \
         | grep -E '^[0-9]+$' \
         | awk -v t="$TARGET" 'BEGIN{best=-1;d=1e18}
                               {x=$1-t; if(x<0)x=-x; if(x<d){d=x;best=$1}}
                               END{print best}')
    echo "=== stage20_${CK}_steps  (target ${TARGET}) ==="
    docker run --rm --gpus all -v "$REPO":/workspace -w /workspace/week4 "$IMAGE" \
        eval_shift.py "checkpoints/stage20_${CK}_steps" \
        --out_prefix "$PREFIX" --shift_x 0 --shift_y 0 --hold 5 2>&1 \
        | grep -vE 'Warning|\[Error\]|deprecated' | tail -12
    echo
done

echo "=== LIFT ACROSS TRAINING (control offset) ==="
echo "reference — stage16_final: 99.4%, 0.1345 m | stage20_final: 100.0%, 0.0257 m"
cat "${PREFIX}_sweep.csv"
