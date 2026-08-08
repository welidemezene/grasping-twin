# Week 4 — make the grasp survive the world moving

Week 3 ended with `stage16_final`: a Franka that grasps a cube 512/512 times and
lifts it 13.5 cm on 99.6% of 512 trials. Then the render pipeline rebuilt the
scene on Isaac Lab 3.0, the cube landed 3.4 cm from where it used to, and the
same checkpoint closed its fingers on empty air.

That is the whole of Week 4. Not a better lift — a lift that still happens when
the cube is not where the policy remembers.

## The mechanism, stated before any training

`inspect_obs.py` recorded the policy's observation group in Week 3:

```
object_position   ->   object_position_in_robot_root_frame
```

The cube's position has been an input the entire time. It was also **constant
for every stage from 1 to 19**, because `curriculum_lift_cfg.py` pins the spawn
range to `(0, 0, 0)` — with a comment saying randomization is "a later stage".
A constant input is indistinguishable from a bias: varying it never changed the
optimal action, so no gradient ever asked the network to attend to it. Twenty
stages of PPO learned an excellent **open-loop trajectory to one point**.

So this is a training-distribution problem. Not the reward, not the architecture.
The observation was always sufficient; the distribution never made it necessary.

## Stage 20: exactly one variable moves

`stage20_cfg.py` sets the cube spawn range to ±5 cm in x and y. Stage 16's
rewards are carried over **untouched** — every weight, the 0.030 hold gate, and
the `cube_upright` weight of 4.

That restraint is a direct lesson from Week 3. The completed 2×2 showed the hold
gate and the `cube_upright` weight each independently destroy the lift, and
stage 18 moved both at once, so it isolated nothing — stage 19 had to be run
purely to undo the wrong conclusion. **Do not declare a hypothesis dead from a
run that changed two variables.** If randomization alone recovers the lift, that
is a clean result. If it does not, the next stage has exactly one suspect.

Warm-started from `stage16_final` at a low learning rate: the arm, the grasp and
the lift are solved and there is nothing to gain by paying for them again.
`ent_coef` is raised 0.002 → 0.004, because a nearly-deterministic policy will
not stumble into a behaviour conditioned on an input it has been ignoring.

## The gate, fixed in advance

`eval_shift.py` is Week 3's `check_airborne.py` measurement — the height of the
cube's **lowest corner** from its full pose, across all 512 environments at once
— swept over fixed offsets by `sweep_shift.sh`, one container per offset
(Isaac Lab's simulation context is a singleton and hangs on a second scene).

**PASS = above 90% grasped-and-airborne AND above 10 cm median lift, at every
held-out offset.**

The offset is identical across all 512 envs for a given run. Randomizing it
per-env would average easy and hard placements into a number that describes no
particular displacement.

| offset | why it is in the sweep |
|---|---|
| 0.000 | control — Week 3's fixed point, must not regress |
| 0.034 diagonal | the displacement that made `stage16_final` grasp air |
| 0.040 x, −0.040 y | single-axis, both directions |
| 0.050 diagonal | at the training edge |
| 0.070 diagonal | past the training range — expected to fail |

A policy that passes 0.070 has learned to look. One that passes only 0.000 has
learned a trajectory.

Standing rules still apply: never from one replay, never from cube height, use
medians (one flung frame at 340 mm moved a mean by 1.5 mm), and Vulkan errors in
these logs are always present and never a signal.

## Baseline, measured 2026-08-08 — and it corrected the premise

`stage16_final` was swept before stage 20 was written, so the improvement would
have something honest to be measured against. The control reproduced Week 3 at
99.4% / 0.1345 m, confirming the new harness agrees with `check_airborne.py`.

Then the sweep disagreed with the story this page was built on.

| shift | success | median lift | verdict |
|---|---|---|---|
| 0.000 control | **99.4%** | 0.1345 m | pass |
| 0.034 diagonal (+x, +y) | **95.9%** | 0.1387 m | pass |
| 0.040 −y | **99.6%** | 0.1274 m | pass |
| 0.040 +x | **41.8%** | 0.057 m | fail |
| 0.050 diagonal (+x, +y) | **41.0%** | 0.1318 m | fail |
| 0.070 diagonal (+x, +y) | **0.0%** | — | fail |

**The 3.4 cm shift does not break it.** 95.9% of 512 trials, with a median lift
of 13.9 mm *higher* than the control. So the Isaac Lab 3.0 failure that motivated
this entire week was not caused by the displacement alone — something else about
that build differed, and attributing it to 3.4 cm was a conclusion drawn from a
change that moved more than one variable. That is the third time this project
has made that specific mistake.

**The brittleness is real, but it is directional, not radial.** 4 cm in −y costs
nothing (99.6%). The same 4 cm in +x collapses to 41.8%. A radius-based story
cannot produce that asymmetry, so "the policy memorised a point" is too crude —
it has a workspace that is generous in one direction and sharply bounded in
another, which points at reach or arm configuration, not purely at perception.

What survives: at 7 cm the policy fails completely (0/512, though it still
*touches* the cube in 6 trials), so there is a genuine generalization ceiling to
push. Stage 20 stands, and the gate stands. What changes is the claim it is
allowed to make — the honest framing is "extend a directional 4 cm envelope to a
uniform 7 cm one", not "teach a blind policy to see".

Open question for the first diagnostic, before stage 20 burns 45 minutes: is the
+x failure a *perception* failure or a *reach* failure? Log whether the arm gets
near the cube at +x 4 cm and misses, or never travels there at all. The two have
different fixes and the sweep cannot tell them apart.

## Files

- `stage20_cfg.py` — randomized spawn (training) and fixed-offset (eval) configs
- `train_stage20.py` — warm start from `stage16_final`, one variable changed
- `eval_shift.py` — 512-trial corner-geometry verdict at one fixed offset
- `sweep_shift.sh` — runs the offset table, one container each
