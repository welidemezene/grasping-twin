# Week 6 — the ladder that was missing

**The goal, in one sentence:** build an instrument that shows a training run's
envelope coverage *while it is running*, prove the instrument works against a run
whose answer is already known, and only then use it on a new one.

## Why this week

Stage 23 widened the spawn box from &plusmn;5 cm to &plusmn;12 cm. That was the
right hypothesis and it is still not falsified — at 3M steps it reached **43.4%**
envelope coverage, the only thing in this project that has ever moved that number.

Then it ran 7M more steps and destroyed itself. `stage23_final` grasps **0 of 512**
at dead centre.

The idea was not the failure. **Nothing was watching** was the failure. The run
was evaluated once, at `_final`, by which point the good policy had been
overwritten fifty times. A single number at the end cannot tell you that the
middle was better.

So week 6 does not start by training anything.

## The gate, fixed before any code

The same rule as weeks 3, 4 and 5.

**For the instrument (phase 2):** `eval_ladder.py` run over stage 23's existing
checkpoints must reproduce the numbers already in `week4/README.md`:

| checkpoint | coverage | joints pinned on failures | verdict |
| --- | --- | --- | --- |
| `stage22_final` | 36.3% | 0.01 | POLICY WALL |
| `stage23_2995200` | 43.4% | 0.16 | POLICY WALL |
| `stage23_6988800` | 25.8% | 1.12 | KINEMATIC WALL |
| `stage23_final` | 0.0% | — | collapse |

If the ladder does not reproduce those, the ladder is wrong. Fix it before
spending GPU time on a run it would misreport.

**For the experiment (phase 3):** stage 24 must beat **36.3%** envelope coverage
**and** hold median lift at the control offset **above 10 cm**.

Both halves. Stage 23's peak passed the first and failed the second — it bought
+7 points of coverage by dropping centre lift from 14.4 cm to 3.1 cm. Coverage
alone is not a win, and a gate with one clause is a gate that gets gamed.

## The design principle for the week

**Build the instrument first, and validate it on data whose answer you already
know.**

Fifty stage 23 checkpoints are sitting in the WSL copy. The curve they describe
is known: up to 43.4% at 3M, down to 25.8% at 7M, zero at the end. That makes
them a test fixture — an experiment where the right answer is already written
down, so a wrong instrument is caught immediately.

Building the tool and the experiment at the same time means a surprising result
has two possible causes, and no way to tell them apart.

## Three phases, and who does what

**Phase 1 — `eval_ladder.py`. His.**
Python, measurement, no Isaac Lab internals. This is the identified curriculum
gap: evaluation, sampling, knowing when a number means something.

**Phase 2 — validate against stage 23. His.**
Run the ladder over the existing checkpoints. Check the four rows above.

**Phase 3 — stage 24. Shared.**
`stage24_cfg.py`, the training script and the Docker launcher are AI-written —
Isaac Lab plumbing is hard, is not the learning target, and `CLAUDE.md` says he
may lean on it there. Running the ladder against the live run, reading the curve,
and calling the peak are his.

## Phase 1 spec — what `eval_ladder.py` must do

**Input:** a checkpoint directory and a stage prefix, e.g. `stage23`.

**For each checkpoint, in step order:** run the existing
`week4/probe_envelope.py` on it, one Docker container each — Isaac Lab's
simulation context is a singleton and hangs if a second scene is built in the
same process.

**Parse from the probe's output** (see `week4/envelope_s23_2995200.txt`):

```
  overall lifted: 222 / 512  (43.4%)          -> coverage
     on successes 0.01     on failures 0.16   -> joints pinned
  VERDICT: POLICY WALL, not kinematics.       -> POLICY or KINEMATIC
```

**Write one CSV:** `step, coverage, joints_pinned_failures, verdict`.

**Print a summary that answers the question stage 23 could not:**

- which checkpoint was best, and at what step
- how many steps the run continued *past* that peak
- a peak rule stated in advance — e.g. coverage has stayed below `best * 0.9`
  for N consecutive checkpoints

That last line is the whole point. A number per checkpoint is data; **"it peaked
1.5M steps ago and is now 40% worse"** is a decision.

## What counts as a result

Stage 24 failing is a result and gets written up in full, the way stage 23 was.
The instrument is the deliverable this week; the run is a use of it.

If stage 24 does not beat the gate, the ladder still tells us *where* it peaked
and *what it traded away* — which is more than stage 23 produced from ten million
steps.

## Files

- `eval_ladder.py` — the checkpoint ladder (phase 1)
- `ladder_s23.csv` — the validation run over stage 23 (phase 2)
- `stage24_cfg.py`, `train_stage24.py`, `run_stage24.sh` — the experiment (phase 3)

## Notes

Docker writes files as **root**; `chown` them back. There is no `python3` on
`PATH` in the Isaac image — the interpreter is `/isaac-sim/python.sh`. Vulkan
errors in headless logs are always present and are never a signal.

The full checkpoint ladder lives only in the WSL compute copy
(`~/grasping_twin/week4/checkpoints/`), about 200 files. The Windows git copy
holds three.
