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

## Stage 22 — the gate was a cliff, and the lift came back over it

Stage 21 eliminated the single-term explanation, which left the one written down
above as the next suspect: `_held` is a **boolean** multiplying five reward terms
at once. Stage 22 changes exactly that and nothing else.

```
near    = sigmoid((0.030 - d) / 0.004)
stopped = sigmoid((bump - 0.5) / 0.08)
_held   = near * stopped            # was: (d < 0.030) & (bump > 0.5)
```

The widths are deliberately tight. The load report checks smooth against boolean
on the same states: clearly held reads 0.970 versus 1, clearly not held reads
0.001 versus 0. **So the gate's definition is unchanged to about 2 mm — only the
cliff is gone.** Warm start is `stage20_final`, because generalization was the
thing worth keeping.

**Both registered predictions were confirmed — the first confirmed hypothesis in
this project after two falsified ones.**

**The lift trajectory reversed.** Stage 20 fell 0.1029 → 0.0257 over this span;
stage 22 climbs: 2.6M 0.0320 → 5.0M 0.0297 → 7.6M 0.0658 → final **0.1440 m**.

**Tilt rose 20.3° → 64.8–80.9°**, exactly as predicted. That is the price, and it
is worse than stage 16's 40°.

| offset | success | lift | gate |
|---|---|---|---|
| control | 99.8% | 0.144 | PASS |
| 3.4 cm diagonal | 99.0% | 0.139 | PASS |
| **+x 4 cm** | **87.7%** | 0.147 | FAIL |
| −y 4 cm | 100% | 0.153 | PASS |
| 5 cm diagonal | 93.2% | 0.126 | PASS |
| 7 cm diagonal | 62.7% | 0.120 | FAIL |

**Four of six rows pass >90% and >10 cm.** +x at 87.7% is 2.3 pp under the bar,
which is **inside the ~3 pp noise floor measured on 2026-08-09** — unresolved, not
failed, and it needs a repeat rather than an explanation.

**Why this survived when `cube_upright` didn't.** It explains the fact stage 21
could not: that randomizing the spawn costs the lift at *every* offset, including
the unshifted control the policy handles perfectly. A fixed spawn makes every
grasp nearly identical, so the boolean flag is stable. A varied grip makes it
flicker, and it flickers hardest under upward acceleration — so a low, still hold
is the only way to keep all five channels switched on.

## Smoothness — the arm shakes, and it always has

The vibration is visible in the stage 22 footage, so it got measured rather than
argued about. `probe_smoothness.py`, 512 envs × 250 steps, deterministic replay.

| | arm delta/step | reversals/step | joint speed |
|---|---|---|---|
| stage 16 | 0.1211 | 0.701 | 0.3750 |
| stage 20 | 0.1669 | 0.502 | 0.5432 |
| **stage 22** | **0.2759** | **0.777** | 0.5114 |

Action channels are in [−1, 1], so stage 22 swings **13.8% of the full command
range every control step**, and each arm channel reverses direction on 78% of
steps. That is oscillation, not travel.

**Not exploration noise** — replays run `deterministic=True`, so nothing is
sampled. What shakes is the policy reacting to observations.

**The cause is that nothing pays for smoothness.** `curriculum_lift_cfg.py` kills
the stock curriculum because it escalates `action_rate` and `joint_vel` from
−1e-4 to −1e-1 at ~5.1M steps, which collapsed stages 10 and 11 and froze every
long run. Killing it left both terms at −1e-4 — **four orders of magnitude under
task rewards that run 2 to 12.** The penalty has existed in name only since stage
12. Every policy in this project is shaky; the video only made it visible.

**Instrument bug found and fixed mid-probe**: reversals were accumulated into a
numpy **bool** array, and `+` on bools is a logical OR, so the counter saturated
at True and reported 0.004 for every policy regardless of the motion. Cast to
int64. The tell was a metric returning the same number for three visibly
different behaviours.

## Phases — where the time goes, and where the shake lives

The episode-wide median above cannot see what the footage suggests: that the hand
arrives, hesitates, and re-grips before lifting. `probe_phases.py` cuts each
episode into APPROACH / DWELL / LIFT_LAG / CARRY, reusing `eval_shift.py`'s grasp
and airborne definitions verbatim so the numbers stay comparable to the sweeps.
First episode only — a mid-run reset costs the "ever" flags nothing but destroys
a phase clock.

| | stage 16 | stage 22 |
|---|---|---|
| reached the cube | 434/512 | **497/512** |
| grasped | 404/512 | **486/512** |
| grasped and lifted | 373/512 | **475/512** |
| APPROACH steps | 24 | 25 |
| **DWELL** arrival → grasp | **11** | **11** |
| **LIFT_LAG** grasp → airborne | **7** | **2** |
| clean single grasp | 90.6% (max 10) | **99.4%** (max 2) |
| arm delta, approach | 0.1390 | 0.1957 |
| arm delta, dwell | 0.2403 | 0.3272 |
| arm delta, carry | 0.1356 | **0.2772** |

**The hesitation is real and chronic, not stage 22's doing.** The hand sits at
the cube for 11 steps before the grasp condition fires — *identically* in both
policies. Nothing has ever paid to shorten it.

**The pause before the lift got better, not worse**: 7 steps → **2**, max 25 → 6.
That is the smooth gate doing precisely its job. The boolean punished committing
to a lift, so the policy waited; without the cliff it goes.

**Re-gripping is stage 16's problem.** It drops and recatches up to ten separate
times in one episode, with only 90.6% of envs managing a single clean grasp.
Stage 22 is at 99.4%.

**The shake is uniform, not concentrated at the grab.** Stage 22 is worse in
every phase — 1.41× approach, 1.36× dwell, and **2.04× on the carry**, the worst
ratio of the three. So the hypothesis that the smooth gate pays for
micro-correction *at the grab specifically* is **falsified**: the excess is not
where that mechanism would put it. It belongs to the −1e-4 penalty, which is
chronic and applies to every policy here. Worth noting that DWELL is the shakiest
phase in *both* policies: the arm is least stable exactly while it is trying to
close on a cube.

## The footage — and a near-miss with a 9-trial anecdote

Neither existing video shows the week 4 finding. `stage22_grid25.mp4` uses the
randomized spawn: ±5 cm on a 4.2 cm cube across 25 arms is about two cube-widths
in a wide shot, so a viewer cannot see the displacement — and every arm succeeds,
so there is no failure in frame to compare against. The result that matters,
41.0% versus 93.2%, is measured with **all 512 envs at the same fixed offset**,
which is exactly what a randomized render never shows.

`replay_shifted.py` films `FrankaLiftShiftedCfg` instead: every environment at
one fixed shift, the way the sweep measures. The 5 cm diagonal was chosen from
the sweep rows, not for looks — the 7 cm diagonal is stage 16's most dramatic
number (0 of 512) but stage 22 fails 37% of the time there too, so both policies
would be dropping cubes on screen.

**The first render was 9 envs and it nearly produced a false result.** Stage 16
came back 7/9 — 78%, against its recorded 41.0%. Taken at face value that would
have said the sweep was wrong, or made a clip where both arms mostly succeed
look like a demonstration. Checked instead of published:

| | stage 16 |
|---|---|
| recorded sweep, 512 | 41.0% |
| re-run `eval_shift.py`, 512 | 35.7% |
| re-run `eval_shift.py`, 9 | 33.3%, then 22.2% |
| render, 64 envs | **42.2%** |
| render, 9 envs | 78% ← the anecdote |

**The render script was right and nine trials was the problem.** At 64 envs it
reproduces the sweep to about a point. The 9-env run was a ~2% draw from a 41%
distribution, which is precisely what the project's standing rule exists to
catch: never declare a result from a small sample. Same rule that retracted week
3 the first time.

The shipped pair is 25 envs each, same offset, same seed:

| | on screen | 512-trial sweep |
|---|---|---|
| `shift50_s16_grid25.mp4` | 11/25 lifted | 41.0% |
| `shift50_s22_grid25.mp4` | **21/25 lifted** | 93.2% |

Fourteen of twenty-five arms fail in the stage 16 clip; four fail in stage 22's.
The on-screen counts are printed by the render itself and stated here because 25
trials cannot restate a 512-trial number — the sweeps remain the evidence and the
video is the illustration.

### The solo cut — one arm, seven cube positions, 60 seconds

The two grid videos fail as public artefacts for opposite reasons. The
randomized grid hides the displacement (25 arms, wide frame, two cube-widths of
motion). The fixed-offset pair shows the failure honestly but compares two
policies at *one* offset — it proves the point to someone reading the sweep, and
shows nothing about handling variety.

A solo close-up inverts the ratio: the camera is on one arm, so ±5 cm is a large
fraction of the frame. `replay_solo_random.py` films one environment through
seven consecutive episodes, **straight through the resets** as week 3's 42-second
cut was. The reset is visible on purpose — the viewer watches the cube jump to a
new place and the same arm go get it. Hiding it would make the clip look edited,
which is the exact suspicion the footage exists to answer.

```
 ep   spawn x    spawn y   grasped  lifted
  1   +0.4899   +0.0017      yes      yes
  2   +0.5089   +0.0324      yes      yes
  3   +0.4604   +0.0405      yes      yes
  4   +0.4923   -0.0269      NO       NO
  5   +0.5101   +0.0065      yes      yes
  6   +0.5248   +0.0498      yes      yes
  7   +0.4986   +0.0243      yes      yes

 7 episodes, 6 lifted
 spawn spread: x 6.4 cm, y 7.7 cm   (the cube is 4.2 cm wide)
```

The spread is larger than the cube itself in both axes, which is what makes the
shot legible where the grid was not. **Episode 4 fails and stays in the cut** —
the sweep says 93.2% at a 5 cm diagonal, so a clip with no failure in it would
be less honest than the number it illustrates, not more impressive.

`stage22_solo_random_episodes.txt` records the table, so a caption can be written
from measurements rather than from watching. A handful of episodes cannot restate
`s22_sweep.csv`'s 512 trials; the video is the illustration and the sweep is the
evidence.

## Where Week 4 lands

**The finding:** domain randomization converts an open-loop replay into a
closed-loop grasp — +x 4 cm from 41.8% to 98.6% — and the lift is the price. That
price is *not* the squareness term, because that was the obvious explanation and
stage 21 tested and eliminated it. It is the **boolean hold gate**, and stage 22
recovers the lift by making that gate smooth while leaving its definition intact
to ~2 mm.

**The gate is met on four of six offsets.** `stage22_final` holds ~14 cm at
99–100% out to a 5 cm diagonal. The two failures are +x 4 cm at 87.7%, inside the
noise floor and unresolved, and the 7 cm diagonal at 62.7% — where stage 16
scored 0.0%.

**Week 4's result is `stage22_final`**, superseding `stage20_final`.

**What it cost:** tilt at 65–81°, worse than any policy in the project, and a
carry that shakes twice as hard as stage 16's. Both are measured, neither is
fixed. `cube_upright` is the term for the tilt, and stage 21 showed that removing
it changes little — so the tilt is not currently explained.

**Carried forward, not fixed:** the smoothness penalty at −1e-4. The fix is cheap
and obvious, but these weights transfer to nothing — the SO-101 is a different
body — and shake matters far more on real hardware than in a simulator that never
wears out. It is a week 5 problem, deliberately.

**Why this mattered for hardware.** `stage16_final` grasped 512/512 without ever
attending to the cube's position — a policy that would have closed on air in
front of a real camera, every time. Stage 20 is the first policy in this project
that reads its sensor, and stage 22 keeps that while getting the lift back. That,
not the 14 cm, is the part that transfers.

## Files

- `stage20_cfg.py` — randomized spawn (training) and fixed-offset (eval) configs
- `train_stage20.py` — warm start from `stage16_final`, one variable changed
- `eval_shift.py` — 512-trial corner-geometry verdict at one fixed offset
- `sweep_shift.sh` — runs the offset table, one container each
- `diagnose_reach.py` / `run_diagnose.sh` — real-cube vs ghost tracking, joint saturation
- `lift_trajectory.sh` — lift across checkpoints; takes `[out_prefix] [stage]`
- `knee_sweep.sh` — finer resolution on the 2.6M–5M region
- `stage21_cfg.py` / `train_stage21.py` — the falsified single-term hypothesis
- `stage22_cfg.py` / `train_stage22.py` — the smooth hold gate; week 4's result
- `replay_stage22.py` — films under `FrankaLiftStage20Cfg`, so every env draws its own cube
- `probe_smoothness.py` — how hard the arm shakes, one number per policy
- `probe_phases.py` / `run_phases.sh` — where the time goes and where the shake lives
- `replay_shifted.py` — films one fixed offset across all envs, the way the sweep measures
- `replay_solo_random.py` — one arm through consecutive random-spawn episodes; logs each episode's spawn and outcome
- `media/` — `shift50_s16_grid25.mp4` / `shift50_s22_grid25.mp4`, the same 5 cm diagonal for both policies; plus `stage22_hero.mp4` and the randomized-spawn grids
