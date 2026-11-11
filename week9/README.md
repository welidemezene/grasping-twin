# Week 9: Diffusion Policy on ALOHA sim (transfer cube)

**Question:** week 8 showed that human demos (many styles) score lower than scripted demos (one style) with the same ACT model: 87.7% vs 94.7%. Diffusion Policy is built for "many right answers". On the **same human demos**, does it beat ACT?

## Gate (written before training)

- **Trained:** Diffusion Policy (LeRobot 0.6.1 defaults), `lerobot/aloha_sim_transfer_cube_human`, **100k steps**, batch 8, training seed 0. Same steps, batch, seed and data as week 8 ACT.
- **One variable:** only the policy type changes (ACT → Diffusion). `crop_shape` stays at the default (off), because ACT had no augmentation either.
- **Success:** the left gripper holds the cube in the air (env reward 4), the same definition as week 8.
- **Exam:** the 100k checkpoint, seeds 1000 / 2000 / 3000 × 100 episodes, **first run of each only**, pooled over 300 episodes. Diffusion's own default re-planning (predict 64, execute 32).
- **Pass if ≥ 93%.** Why 93: ACT on the same data scored 87.7%. The noise on a 300 vs 300 gap is about ±2.6 points, and a real win needs about 2× that, so 87.7 + ~5 ≈ 93.
- **If it fails:** at default settings, Diffusion is not a drop-in fix for the human-data gap. Then decide whether **one** tuning run (one change, e.g. crop on) is worth it before moving on to collecting targeted demos on the SO-101.

- **Early stop (decided before any progress numbers):** run to 100k. Stop early **only if the run is broken**: the 40k progress check is 0/20 **and** the training loss is not falling. A plateau is **not** a reason to stop: 20-episode checks move about ±10 points by chance (week 8 dipped to 66% at 80k, then reached 78% at 100k). Every 20k checkpoint is saved, so any of them can be examined later.

Limit: one training run (seed 0); training-seed variance is not measured.

## Setup facts (smoke test, 300 steps, 2026-11-09)

| | ACT (week 8) | Diffusion |
|---|---|---|
| Parameters | 52M | 263M |
| Training speed | ~8.5 steps/s | 3.85 steps/s |
| 100k steps | ~5 h | ~7.2 h |
| GPU memory | — | 6.2 GB of 12 |
| Eval, per episode | ~12 s | ~26 s (100 denoising steps per chunk) |

Needed `uv pip install 'lerobot[diffusion]==0.6.1'` (adds `diffusers`; LeRobot and torch unchanged).

## Plan

1. [ ] Train 100k: `bash week9/train.sh` (progress check every 20k on 20 episodes; this is a signal, not the exam)
2. [ ] Exam: `bash week9/eval.sh` (3 seeds × 100, ~2.2 h)
3. [ ] Count failure stages: `python3 week8/tools/count_failures.py ~/diff_runs/eval/*/eval_info.json`
4. [ ] Watch failures; compare with ACT's (mostly carry)
5. [ ] Result + verdict vs 93%, in my own words
6. [ ] Next action + what I can now do unaided


YES i think yes it would beat which we have in a ACT otherwise what is the point of having diffustion

a litter better than 93%


## Result (2026-11-09)

Diffusion, default settings, 100k checkpoint, 300 episodes:

| seed 1000 | seed 2000 | seed 3000 | pooled |
|---|---|---|---|
| 67% | 68% | 63% | 198/300 = 66.0% |

Failures: reach 23, grasp 15, carry 64, handover 0.

**Gate (≥ 93%):** did not pass. We did not improve it by looking at its weakest parts first, but at default settings it did not pass.

**My prediction:** wrong. 66% is less than I predicted ("a little better than 93%").

**Compared with ACT's default (79.5%), I can say:** I expected the Diffusion default to be better than the ACT default, but it was not. 66% vs 79.5% is a big gap.

**What I can't say yet:** that ACT is better than Diffusion. Diffusion was untuned, so we need to tune it before comparing. It also has many more parameters (263M vs 52M), so it may need more demos than ACT; that is a guess, not tested.

**What I saw in the videos:** a lot of passes and failures. Mostly, when it fails, it does not notice whether it picked up the cube or not.

**Why I think reach failures appeared (ACT at execute 50/100 had 0, at 25 had 7, at 10 had 63; Diffusion executes 32):** the demos are not the problem. ACT learned from the same 50 demos and had no reach failures. It is like two friends going to the same gym: one gets stronger and the other does not, so the difference is in the person, not the gym.

My guess is that Diffusion re-plans too fast, every 0.64 s, and every re-plan starts from scratch (new random noise). So it can go to the left, then re-plan and go to the right, and never reach the cube. Two things together: it re-plans often, and each re-plan can choose a different side. To check it, make it commit longer before re-planning, close to ACT's 1 s (round 2 below).

## Round 2 gate: does committing longer fix the reach failures? (written before running, 2026-11-11)

**Idea being tested:** Diffusion re-plans every 32 steps (0.64 s), more often than ACT's good settings (1–2 s), and every re-plan starts from new random noise, so it can switch between left and right routes and never reach the cube. If that is true, executing more steps per plan should reduce reach failures, and executing fewer should increase them.

- **Same** 100k checkpoint, no retraining. **Only** `n_action_steps` changes.
- **Choosing step, seed 4000** (never used by the exam), 100 episodes each: execute **32** (default, baseline on this seed), **16**, **48**, **63** (maximum: 64 predicted − 2 observation frames + 1). Script: [sweep.sh](sweep.sh).
- **Idea supported if**, on seed 4000: reach failures at the better of 48/63 are **at most half** of those at 32, **and** 16 has **more** reach failures than 32.
- **Best setting** = highest success rate on seed 4000 (ties → longer commit).
- **Exam:** the best setting on seeds 1000 / 2000 / 3000 × 100, first run only, pooled 300. **Pass if ≥ 85%**: beats ACT's default (79.5%) by more than noise and comes close to tuned ACT (87.7%).

**My prediction:** 48 wins on seed 4000 with about 85% success. ACT's best was re-planning every 1 s, and 48 (0.96 s) is closer to 1 s than 63 (1.26 s). I expect the reach idea to be supported, for the reasons in my explanation above.
