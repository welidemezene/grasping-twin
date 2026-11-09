#!/bin/bash
# Week 9 exam: Diffusion 100k checkpoint, seeds 1000/2000/3000 x 100 episodes (same exam as week 8).
# (written by Claude, at Woldemedihn's request)
set -e
source $HOME/grasping_twin/lerobot/bin/activate
export MUJOCO_GL=egl
CKPT=$HOME/diff_runs/human_seed0/checkpoints/100000/pretrained_model
for SEED in 1000 2000 3000; do
  echo "=== diffusion seed=$SEED ($(date +%H:%M)) ==="
  lerobot-eval --policy.path=$CKPT --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=100 --eval.batch_size=1 --eval.use_async_envs=false --seed=$SEED \
    --policy.device=cuda --output_dir=$HOME/diff_runs/eval/seed$SEED
done
python3 $HOME/grasping_twin/week8/tools/count_failures.py $HOME/diff_runs/eval/*/eval_info.json
echo "=== done ($(date +%H:%M)) ==="
