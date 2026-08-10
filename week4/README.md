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

## The diagnostic, 2026-08-09 — neither perception nor kinematics

`diagnose_reach.py` ran three matched 512-trial replays of `stage16_final`,
tracking per step the end-effector's distance to the **real** cube and to the
**ghost** (Week 3's fixed spawn point), plus arm-joint saturation.

| run | approach | shift compensated | joints pinned | grasped | success |
|---|---|---|---|---|---|
| control | 4.6 mm | — | 0 | 99.8% | 99.6% |
| −y 4 cm | 9.9 mm | 75% | 0 | 100% | 98.6% |
| **+x 4 cm** | 21.1 mm | **47%** | 0 | 62.5% | 38.7% |

**Not perception.** The arm moves toward the real cube and away from the ghost —
only 25% of envs ever visit the ghost, against 99.6% reaching the cube. The
position channel is not being ignored.

**Not kinematics.** Zero arm joints came within 5% of a limit at any step of any
run, and +x cubes that *do* get grasped lift 14.0 cm — higher than the control.
Everything downstream of the grasp is intact.

**Verdict: partial tracking.** An under-powered, directional position→motion
gain. The arm goes most of the way and stops 21 mm out, which is too far to close
on. Randomization supplies exactly that gain, so stage 20 was the right run.

Two method notes, both of which cost something to learn:

- The first pass judged "arrived" against `eval_shift.py`'s 30 mm grasp band and
  reported +x as arriving — while the control arrives at 4.6 mm. **A band that
  cannot separate 4.6 from 30.4 cannot separate a grasp from a miss.** A pass/fail
  threshold borrowed from a different question answers the wrong question. It now
  reports a continuous compensation fraction.
- **Noise floor, measured:** +x 4 cm scored 41.8 / 41.2 / 38.7 across three
  identical configs. PhysX is not bit-deterministic. **Differences under ~3
  percentage points are not results**, anywhere on this page.

## Stage 20 — generalization solved, and the lift paid for it

`stage20_final`, six offsets × 512 trials (`s20_sweep.csv`):

| offset | stage 16 | **stage 20** | median lift |
|---|---|---|---|
| control | 99.4% | **100%** | 0.0257 m |
| 0.034 diagonal | 95.9% | **99.8%** | 0.0268 m |
| **0.040 +x** | 41.8% | **98.6%** | 0.0306 m |
| 0.040 −y | 99.6% | **99.6%** | 0.0265 m |
| **0.050 diagonal** | 41.0% | **99.8%** | 0.0308 m |
| 0.070 diagonal | 0.0% | **29.1%** | 0.0283 m |

**Directional brittleness is gone.** The under-reaching diagnosis was right and
randomization supplied the gain. **But the median lift fell 0.1345 → 0.0257 m at
every offset including the control**, and tilt 40° → 20°. Every row fails the
gate on lift.

### The reward pays for the collapse

`lift_trajectory.sh` and `knee_sweep.sh` probe existing checkpoints — no
retraining — at the control offset:

| steps | success | median lift |
|---|---|---|
| 2.6M | 74.0% | **0.1029 m** |
| 3.39M | 94.1% | 0.0778 m |
| 5.0M | 96.5% | 0.0358 m |
| 7.6M | 100% | 0.0226 m |
| 10M | 100% | 0.0257 m |

**Overtrained, not undertrained.** Reliability was bought and lift and tilt paid,
continuously, across the whole run. And `ep_rew_mean` climbed **66 → 96.5 over
exactly that span** — reward up while the task got worse. Fifth instance of
reward hacking in this project, and the first caught by an instrument rather than
by watching a video.

No checkpoint closes Week 4: the curves cross *below* the gate, and the best joint
point is 3.39M at 94.1% / 7.8 cm, short by 2.2 cm. One caveat — the 2.6M–3.4M
region is genuinely unstable (2.99M scores 42.6% / 4.6 cm, worse than both
neighbours), so "monotonic" is too strong. The direction holds across the run;
the early trajectory does not.

### What this says about Week 3

`stage16_final` was never the best policy its reward could produce. It was a
policy *in transit* to this same attractor, and the 10M cutoff happened to land
somewhere good. Three interventions — a tighter hold gate, a heavier
`cube_upright`, a randomized spawn — all land on ~100% / ~2 cm / ~20°. The Week 3
2×2 was not measuring two levers. It was measuring how fast different configs
slide down one hill.

## Stage 21 — the obvious explanation, tested and falsified

One suspect was named in advance: `cube_upright_in_hand` pays for squareness at
**any** height while `lifting_progress_in_hand` requires height, so once the
spawn moves and a lift is a gamble, a square hold just above the table is the
guaranteed earner. Stage 21 removes that term — weight 4 → **0**, not 2, because
a half-step leaves both explanations alive — warm-started from `stage16_final`,
which is stage 20's *start* rather than its result, so one term is the only
difference between the two runs.

**Both halves of the registered prediction were wrong.**

| | predicted | measured |
|---|---|---|
| lift | returns toward 10 cm | **2.7 cm** — unchanged |
| tilt | regresses toward stage 15's 79° | **19.9°** — slightly better than stage 20 |

The trajectory (`s21_traj_sweep.csv`) is 83.6% / 0.0617 at 2.6M → 99.6% / 0.0302
at 5.0M → 100% / 0.0267 at 7.6M: the same slide on the same schedule, starting
*lower* than stage 20 did. That is precisely the falsification condition written
down before the run.

The tilt result is the more important one. **Deleting the squareness term outright
left the tilt at ~20°, so that term was not producing the squareness credited to
it.** The mechanism story carried since Week 3 is disproven on its own terms, and
stage 19's result — weight 4 → 10 dropping the lift 124.3 → 15.3 mm — is once
again unexplained.

It also cost generalization: control 83.2%, +x 4 cm **2.9%**, 5 cm diagonal
**1.4%**, all far outside the noise floor. `stage21_final` is worse than its own
7.6M checkpoint (83.2% vs 100%), so the run destabilized late — judge stage 21 by
the trajectory, not the final.

**Stage 20 remains Week 4's result, strictly better on every axis.**

## Where Week 4 lands

**The finding:** domain randomization converts an open-loop replay into a
closed-loop grasp — +x 4 cm from 41.8% to 98.6% — and the lift is the price. That
price is *not* attributable to the squareness term, because that was the obvious
explanation and it was tested and eliminated.

**The gate is not met.** Stage 20 gives ~100% success at 2.6 cm; the gate wanted
>90% *and* >10 cm. The honest frontier point is stage 20's 3.39M checkpoint at
94.1% / 7.8 cm.

**The next suspect, untested:** `_held` is a **boolean** multiplying five reward
terms at once. A fast, high lift shifts the cube between the fingers, trips the
gate, and the wanted behaviour switches its own reward off. Three separate
single-term interventions have now landed on ~100% / ~2 cm / ~20°, which points
at the gating *structure* rather than any one weight.

**Why this mattered for hardware.** `stage16_final` grasped 512/512 without ever
attending to the cube's position — a policy that would have closed on air in
front of a real camera, every time. Stage 20 is the first policy in this project
that reads its sensor. That, not the 13.5 cm lift, is the part that transfers.

## Files

- `stage20_cfg.py` — randomized spawn (training) and fixed-offset (eval) configs
- `train_stage20.py` — warm start from `stage16_final`, one variable changed
- `eval_shift.py` — 512-trial corner-geometry verdict at one fixed offset
- `sweep_shift.sh` — runs the offset table, one container each
- `diagnose_reach.py` / `run_diagnose.sh` — real-cube vs ghost tracking, joint saturation
- `lift_trajectory.sh` — lift across checkpoints; takes `[out_prefix] [stage]`
- `knee_sweep.sh` — finer resolution on the 2.6M–5M region
- `stage21_cfg.py` / `train_stage21.py` — the falsified single-term hypothesis
