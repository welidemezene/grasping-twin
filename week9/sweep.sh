#!/bin/bash
# Week 9 round 2: does committing longer (executing more of each 64-step plan) reduce reach failures?
# Same Diffusion 100k checkpoint, no retraining; only n_action_steps changes between runs.
# Max n_action_steps = horizon - n_obs_steps + 1 = 64 - 2 + 1 = 63.
# Usage:  bash week9/sweep.sh EPISODES SEED "LIST"
#   bash week9/sweep.sh 100 4000 "32 16 48 63"   (choose the setting on a seed the exam never uses)
#   bash week9/sweep.sh 2 4000 "32 63"           (dry run, 2 episodes)
set -e

N=${1:-100}                    # episodes per setting
SEED=${2:-4000}                # eval seed = which cube positions
LIST=${3:-"32 16 48 63"}       # n_action_steps values to try

CKPT=$HOME/diff_runs/human_seed0/checkpoints/100000/pretrained_model
OUT=$HOME/diff_runs/sweep_n${N}_seed$SEED
source $HOME/grasping_twin/lerobot/bin/activate
export MUJOCO_GL=egl

for n in $LIST; do
  echo "=== seed=$SEED n_action_steps=$n  ($(date +%H:%M)) ==="
  lerobot-eval --policy.path=$CKPT --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=$N --eval.batch_size=1 --eval.use_async_envs=false \
    --seed=$SEED --policy.device=cuda --policy.n_action_steps=$n \
    --output_dir=$OUT/nas_$n
done

# Success rate and failure stages of every run in this sweep
python3 $HOME/grasping_twin/week8/tools/count_failures.py $OUT/nas_*/eval_info.json
for f in $OUT/nas_*/eval_info.json; do
  echo "$f: $(grep -o '"pc_success": [0-9.]*' $f | tail -1)"
done
echo "=== done ($(date +%H:%M)) ==="
