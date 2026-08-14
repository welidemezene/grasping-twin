# grasping_twin

Self-study physical AI course: teaching a simulated Franka arm to grasp and lift a cube, from scratch, with the bugs left visible.

## Status

- **Week 1 (done)**: PPO on Reacher-v5 (MuJoCo/Gymnasium).
- **Week 2 (done)**: Isaac Lab + Isaac Sim 4.5 via Docker in WSL2 (RTX 5070 Ti, glibc workaround). Franka Lift training at up to 4096 parallel envs.
- **Week 3 (CLOSED 2026-08-06)**: result = `stage16_final` — **100% grasp, 13.5 cm median lift, 99.6% grasped-and-airborne over 512 trials** (`check_airborne.py`, cube-corner geometry, not height). Stages 17–19 tried to reduce the 40° carry tilt via a tighter hold gate and a heavier `cube_upright` weight; both levers independently destroy the lift, so the 2×2 is complete and stage 16 stands.
- **Week 4 (CLOSED 2026-08-11)**: result = `stage22_final` — **99.8% grasped-and-airborne, 14.4 cm median lift, 4 of 6 offsets pass the gate**. Stage 16 lifted 512/512 without ever reading the cube's position (41.8% once it moved 4 cm); stage 20's randomized spawn fixed that (**→ 98.6%**) but destroyed the lift; stage 21 falsified the obvious cause; stage 22 traced it to the **boolean hold gate** and recovered the lift by smoothing it. Cost, measured and unfixed: 65–81° carry tilt, unexplained, and 2× the shake of stage 16. Stage 23 (spawn box ±5 → ±12 cm) peaked at 3M steps then unlearned itself to 0/512 by `_final` — recorded, not hidden.
- **Week 5: starting** — leaving simulation for the SO-101 via LeRobot. No weights carry over; the measurement discipline does.

## What Week 3 actually taught (the bugs are the learning)

1. **Reward hacking**: the stock lift reward paid for cube height without a grasp — the policy batted and hovered while ep_rew read 88.
2. **Exploration collapse**: the gripper action head drifted to "always open" (mean +2.07, close a 4.1-sigma event) — PPO cannot learn an action it never samples. Found by one instrumented replay after nine stages of reward-guessing.
3. **Per-step gripper dither**: fixed with a sticky gripper (action held 5 frames per decision).
4. **A hidden curriculum**: the stock task multiplies movement penalties 1000× mid-run, silently killing every long training run's second half.

Standing rule: never declare a result from one replay or a height test — 512 trials, corner geometry, and watch the video.

## Layout

- `week1/`, `week2/`, `week3/` — code, per-stage training scripts, replay/eval tooling, and measurement logs.
- `viewer/` — browser-based motion replay viewer.
