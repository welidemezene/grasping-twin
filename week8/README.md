# Week 8: ACT on ALOHA sim (transfer cube)

![Success (left) vs failure (right) at 6 s](media/success_vs_failure_6s.png)

*Left: success, the cube is passed in mid-air. Right: failure, the cube is lifted, then dropped; no handover.*

Gate, written before training:
- Success = the left gripper holds the cube in the air (env reward 4). *Clarified after training; the gate said "Success = pass".*
- Attempts = 50, because it is hard to decide the behavior of traning with the less attempts 
- Pass if success rate >= 80 % at 100k steps, because this time we use all the data so it should pass minimum 80% of what it trying to do in first traning 

## Setup

| | |
|---|---|
| Policy | ACT (LeRobot 0.6.1, unmodified: [modeling_act.py](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/modeling_act.py)), chunk_size 100, n_action_steps 100 (re-plans every 2 s) |
| Data | `lerobot/aloha_sim_transfer_cube_human`, 50 human demos, 20,000 frames, one top camera |
| Env | `gym_aloha` `AlohaTransferCube-v0`, 400 steps = 8 s at 50 fps |
| Training | 100k steps, batch 8, seed 0, RTX 5070 Ti laptop (12 GB), ~5 h |
| Eval | 50 episodes per checkpoint |

How it was run:

```
MUJOCO_GL=egl lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --env.type=aloha --env.task=AlohaTransferCube-v0 --steps=100000 --batch_size=8 --env_eval_freq=20000 --eval.n_episodes=50 --save_freq=20000 --seed=0 --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --output_dir=$HOME/act_runs/transfer_cube_seed0
```

The 20k eval crashed (`NamespaceNotFound: gym_aloha` in the async eval workers); training was resumed from the 20k checkpoint with `--eval.use_async_envs=false`.

## Result

| Checkpoint | Success (50 trials) |
|---|---|
| 40k | 74% (37/50) |
| 60k | 76% (38/50) |
| 80k | 66% (33/50) |
| 100k | **78% (39/50)** |

**Verdict: FAIL** (78% < 80%, by one attempt).

12 s test (diagnostic, after training): 100k checkpoint, `--env.episode_length=600` instead of 400, 50 trials: **72% (36/50)**. More time did not help, so the failures are stuck, not slow.

i just think first times it should pass 80% but we did not reach there after first 40k runs it does not improve the process so for me am taking this as fail and also the other things are i watch a video it is good but it fails sometimes (11 of 50) so we need to impove it i do check it does not have a loop issue it is some issue why it is not updating after some steps we may use more demos focusing on what points are our trained model failling.

so i belive more loop helps after we update or add more demos focusing where our trained model are missing things 

Next test: add demos at the failure positions, same 100k steps. If success is still rising at 100k, then test longer training.

so to identify which one cause to have a better trained model we need to do it in sequence otherwise it gona be hard to know and understand what causes the model trained good.

## Tools

`tools/eval_angle_cam.py` (written by Claude): runs `lerobot-eval` but records the video from the `angle` camera with recolored arms. The policy still sees the original `top` camera, so its behaviour is unchanged (checked: same 10 outcomes as the plain run).

```
MUJOCO_GL=egl python week8/tools/eval_angle_cam.py --policy.path=<checkpoint>/pretrained_model --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=10 --eval.batch_size=1 --eval.use_async_envs=false --output_dir=<out>
```
