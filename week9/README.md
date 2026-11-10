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

