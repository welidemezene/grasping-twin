# grasping_twin

Self-study physical AI course: teaching a simulated Franka arm to grasp and lift a cube, from scratch, with the bugs left visible.

## Status

- **Week 1 (done)**: PPO on Reacher-v5 (MuJoCo/Gymnasium).
- **Week 2 (done)**: Isaac Lab + Isaac Sim 4.5 via Docker in WSL2 (RTX 5070 Ti, glibc workaround). Franka Lift training at up to 4096 parallel envs.
- **Week 3 (CLOSED 2026-08-06)**: result = `stage16_final` — **100% grasp, 13.5 cm median lift, 99.6% grasped-and-airborne over 512 trials** (`check_airborne.py`, cube-corner geometry, not height). Stages 17–19 tried to reduce the 40° carry tilt via a tighter hold gate and a heavier `cube_upright` weight; both levers independently destroy the lift, so the 2×2 is complete and stage 16 stands.
- **Week 4: starting.**

## What Week 3 actually taught (the bugs are the learning)

1. **Reward hacking**: the stock lift reward paid for cube height without a grasp — the policy batted and hovered while ep_rew read 88.
2. **Exploration collapse**: the gripper action head drifted to "always open" (mean +2.07, close a 4.1-sigma event) — PPO cannot learn an action it never samples. Found by one instrumented replay after nine stages of reward-guessing.
3. **Per-step gripper dither**: fixed with a sticky gripper (action held 5 frames per decision).
4. **A hidden curriculum**: the stock task multiplies movement penalties 1000× mid-run, silently killing every long training run's second half.

Standing rule: never declare a result from one replay or a height test — 512 trials, corner geometry, and watch the video.

## Layout

- `week1/`, `week2/`, `week3/` — code, per-stage training scripts, replay/eval tooling, and measurement logs.
- `viewer/` — browser-based motion replay viewer.
