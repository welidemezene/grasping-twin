# Week 5 — the policies, in the browser

**The goal, in one sentence:** one URL where anyone can replay week 4's policies,
compare two side by side on the same cube position, and see the 512-trial numbers
that back them.

## Why this week, and not the SO-101

Week 4 closed naming week 5 as "SO-101 via LeRobot." That needs an SO-101, and
there isn't one. Ordering an arm, shipping it, and assembling it is not a
one-week result, and the honest thing is to say so rather than let the plan drift
while nothing ships.

Two more reasons this is the better week, not just the available one:

**Week 4's evidence is currently unwatchable.** The result exists as 64 MB MP4
files. A reviewer cannot scrub them, cannot compare two policies frame-by-frame,
and cannot check a number against the file it came from. `viewer/` already
replays a policy in the browser from a joint-angle JSON — and it stopped at
stage 19. It has no idea week 4 happened.

**Reinforcement learning does not transfer to that arm anyway.** Weeks 1–4 are RL
in simulation: millions of attempts against a reward function. A real SO-101 is
driven by imitation learning — you teleoperate it, record ~50 demonstrations, and
a policy copies you. Different body, different technique, and no reward signal
off-sim. `stage22_final` cannot be uploaded to it. What transfers is the
measurement discipline, and that is worth strengthening before it moves house.

## The gate, fixed before any code

The same rule as weeks 3 and 4: the bar is written down first, so passing it
cannot be renegotiated afterwards.

1. **Four policies replay** — `stage16_final`, `stage20_final`, `stage22_final`,
   `stage23_2995200`. That is the whole week 4 argument: blind → sees but cannot
   lift → sees and lifts → overreached and collapsed.
2. **Two play side by side, on the same cube position, scrubbing in sync.** This
   is the reason the tool exists. An MP4 cannot do it, and it is the only way to
   *see* that one arm goes where the cube used to be while the other goes where
   it is.
3. **Every number on screen traces to a committed file.** No figure typed in by
   hand. A hand-typed number drifts from its source.
4. **three.js is vendored, not loaded from a CDN.** `viewer/index.html` currently
   pulls r128 from cdnjs; the day that URL moves, the portfolio piece is a black
   screen.
5. **Loads and runs on a phone.**
6. **Deployed and publicly linkable.**

## What is honest to claim, and what is not

Each recording is **one episode at one cube offset**. It illustrates; it does not
measure. Every success *rate* comes from `week4/*_sweep.csv` — 512 trials each.

This is not a stylistic preference. Twice during week 4 a small sample nearly
shipped a false result: a 9-environment render read 78% for a policy the sweep
puts at 41%, and a 9-episode comparison "ranked" two policies whose real scores
differ by 7 points. The viewer must therefore show the sweep number next to the
clip, and must never compute a rate from what is on screen.

## The two offsets, and why the pair is the argument

| offset | stage 16 | stage 22 | what it shows |
| --- | --- | --- | --- |
| control, 0 cm | 99.6% | 99.8% | both look fine — this offset alone proves nothing |
| 5 cm diagonal | 41.0% | 93.2% | stage 16 closes on air; stage 22 tracks the cube |

The 7 cm diagonal is more dramatic for stage 16 (0.0%) but stage 22 also fails
37% of the time there, so both arms drop cubes and the story muddies. The 5 cm
diagonal is where one policy nearly always fails and the other nearly always
works, which is the honest version of the same point.

## Progress

| # | Step | State |
| --- | --- | --- |
| 1 | Record the four policies to JSON at two fixed offsets | done — `motion/`, 8 files |
| 2 | **Viewer accepts the `{meta, frames}` format** | **in progress** |
| 3 | Label and offset shown on screen, read from `meta` | not started |
| 4 | Two policies side by side, scrubbing in sync | not started |
| 5 | Vendor three.js instead of the CDN | not started |
| 6 | Deploy to a public URL | not started |

Step 4 is the one an MP4 cannot do, and the reason the tool is worth building.
Steps 2 and 3 exist to make it possible.

## Session log

Kept because the documented failure mode on this project is not difficulty, it is
**long gaps between bursts** — and the cost of a gap is paid in reorientation. A
specific next action costs minutes to resume from; "continue the viewer" costs
days.

**2026-08-14 → 15.** Week 4 closed out in the README, including stage 23's
collapse. Week 5 opened: `record_motion_w4.py` and `record_all_w4.sh` written,
eight recordings produced in Docker, and a bad `lifted` threshold caught and
moved into `derive_flags.py` — it tested `cube_z > 0.035` while the cube rests at
`0.0521`, so all eight files had claimed a held lift, one of them while moving the
cube −3.5 mm. `CLAUDE.md` written: he types, the AI explains.

> **Next action.** In `viewer/index.html`, `load()` at line 447: add a `meta`
> variable to the state list on line 445, and unwrap at the top of the function —
> if `data` is not an array but `data.frames` is, keep `data.meta` and continue
> with `data.frames`. Done when `week5/motion/s22_diag50.json` opens **and**
> `viewer/motion_s16_final.json` still opens.

> **What he can do unaided that he could not before.** Explain why nine episodes
> cannot rank two policies and 512 can; state the difference between training and
> evaluation; and read a JavaScript failure by tracing execution order rather than
> trusting the error message — found by putting a `console.log` after a `throw`
> and reasoning out why it never printed. The `load()` change itself is not
> written yet.

## Files

- `record_motion_w4.py` — exports one arm's trajectory to JSON at a **fixed**
  cube offset (`FrankaLiftShiftedCfg`), with the sticky gripper training used.
  Emits `{"meta": …, "frames": […]}` so the viewer can caption itself from the
  file instead of from a typed-in string
- `record_all_w4.sh` — the four policies at both offsets, one container each
  (Isaac Lab's simulation context is a singleton and hangs on a second scene)
- `motion/` — the recordings, plus a `_summary.txt` per recording

## Note on the two repo copies

`~/grasping_twin` in WSL is the **compute** copy: it holds the full checkpoint
ladder (every 200k steps for stages 20–23, ~200 files) and is where Docker runs.
Its git history stops at week 3. The Windows copy is the **git** copy and holds
only the checkpoints worth versioning. Scripts are written on Windows, copied
into WSL to run, and committed on Windows.
