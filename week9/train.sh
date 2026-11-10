#!/bin/bash
# Week 9: Diffusion Policy, same data/steps/batch/seed as week 8 ACT. Only the policy type changes.
# Usage: nohup bash week9/train.sh > ~/diff_runs/train.log 2>&1 &
set -e
source $HOME/grasping_twin/lerobot/bin/activate
export MUJOCO_GL=egl
lerobot-train --policy.type=diffusion \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps=100000 --batch_size=8 --seed=0 \
  --env_eval_freq=20000 --eval.n_episodes=20 --eval.batch_size=1 --eval.use_async_envs=false \
  --save_freq=20000 --log_freq=200 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir=$HOME/diff_runs/human_seed0
