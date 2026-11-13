#!/bin/bash
# Week 9 exam: Diffusion 100k checkpoint, seeds 1000/2000/3000 x 100 episodes (same exam as week 8).
# Usage:  bash week9/eval.sh          (round 1: default n_action_steps = 32)
#         bash week9/eval.sh 48       (round 2: the setting chosen on seed 4000)
#         RUN=b24_seed0 STEP=033333 bash week9/eval.sh   (round 3: the batch-24 checkpoint)
set -e
NAS=${1:-}                     # n_action_steps; empty = policy default
source $HOME/grasping_twin/lerobot/bin/activate
export MUJOCO_GL=egl
RUN=${RUN:-human_seed0}; STEP=${STEP:-100000}   # which training run and checkpoint to test
CKPT=$HOME/diff_runs/$RUN/checkpoints/$STEP/pretrained_model
if [ -n "$NAS" ]; then EXTRA="--policy.n_action_steps=$NAS"; OUT=$HOME/diff_runs/eval_nas$NAS; else EXTRA=""; OUT=$HOME/diff_runs/eval; fi
[ "$RUN" != "human_seed0" ] && OUT=${OUT}_$RUN
for SEED in 1000 2000 3000; do
  echo "=== diffusion seed=$SEED n_action_steps=${NAS:-default} ($(date +%H:%M)) ==="
  lerobot-eval --policy.path=$CKPT --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=100 --eval.batch_size=1 --eval.use_async_envs=false --seed=$SEED \
    --policy.device=cuda $EXTRA --output_dir=$OUT/seed$SEED
done
python3 $HOME/grasping_twin/week8/tools/count_failures.py $OUT/*/eval_info.json
echo "=== done ($(date +%H:%M)) ==="
