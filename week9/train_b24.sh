#!/bin/bash
# Week 9 round 3: same as train.sh, but batch 24 instead of 8.
# Steps cut to 33,333 so the network still sees the data ~40 times (33,333 x 24 = 800k = 100k x 8).
# Practice checks are off (--env_eval_freq=0): batch 24 uses 10.5 of ~11.5 GB, and they could run out of memory.
# Usage: nohup bash week9/train_b24.sh > ~/diff_runs/train_b24.log 2>&1 &
set -e
source $HOME/grasping_twin/lerobot/bin/activate
export MUJOCO_GL=egl
lerobot-train --policy.type=diffusion \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps=33333 --batch_size=24 --seed=0 \
  --env_eval_freq=0 \
  --save_freq=10000 --log_freq=200 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir=$HOME/diff_runs/b24_seed0
