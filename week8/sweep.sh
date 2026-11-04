#!/bin/bash
# Week 8 sweep: does re-planning more often fix the failures?
# Same 100k checkpoint; only n_action_steps changes between runs.
# Usage:  bash week8/sweep.sh EPISODES SEED "LIST"
#   bash week8/sweep.sh 100 1000 "100 50 25 10"   (first sweep, Nov 4, + ensemble)
#   bash week8/sweep.sh 100 2000 "100 50"         (confirmation run)
#   bash week8/sweep.sh 2 2000 "100 50"           (dry run, 2 episodes)
# (written by Claude, at Woldemedihn's request)

set -e                         # stop at the first error

N=${1:-100}                    # episodes per config   (1st argument, default 100)
SEED=${2:-1000}                # eval seed = which cube positions (2nd argument)
LIST=${3:-"100 50 25 10"}      # n_action_steps values to try (3rd argument)

CKPT=$HOME/act_runs/transfer_cube_seed0/checkpoints/100000/pretrained_model
OUT=$HOME/act_runs/sweep_n${N}_seed$SEED
EVAL=$HOME/grasping_twin/lerobot/bin/lerobot-eval
export MUJOCO_GL=egl

COMMON="--policy.path=$CKPT --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=$N --eval.batch_size=1 --eval.use_async_envs=false \
  --seed=$SEED --policy.device=cuda"

# The robot looks again every n steps
for n in $LIST; do
  echo "=== seed=$SEED n_action_steps=$n  ($(date +%H:%M)) ==="
  $EVAL $COMMON --policy.n_action_steps=$n --output_dir=$OUT/nas_$n
done

# Temporal ensemble: only part of the first sweep (seed 1000)
if [ "$SEED" = "1000" ]; then
  echo "=== temporal ensemble  ($(date +%H:%M)) ==="
  $EVAL $COMMON --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
    --output_dir=$OUT/ensemble
fi

# Print the result of every run in this sweep
for f in $OUT/*/eval_info.json; do
  echo "$f: $(grep -o '"pc_success": [0-9.]*' $f | tail -1)"
done
echo "=== done ($(date +%H:%M)) ==="
