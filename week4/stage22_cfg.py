"""Stage 22 — the hold gate is a cliff, and a lift is what walks off it.

Stage 20 solved generalization (+x 4 cm 41.8% -> 98.6%, 5 cm diagonal 41.0% ->
99.8%) and lost the lift doing it (0.1345 -> 0.0257 m at every offset, control
included). Stage 21 tested the obvious explanation under single-variable
conditions and FALSIFIED it: with cube_upright_in_hand removed outright the lift
stayed at 2.7 cm, and the tilt -- predicted to regress toward stage 15's 79 deg
-- came out at 19.9 deg, marginally better than stage 20's. The term was not
producing the squareness credited to it, and it was not the price of the lift.

The knee probe closed off the remaining cheap fix. The 3.39M checkpoint's
94.1%/7.8 cm was measured at the control offset only; swept properly it scores
28.9% at +x 4 cm and lifts 1 mm there, and 3.79M scores 4.7%. Success at +x 4 cm
across the run reads 41.8% (stage 16) -> 28.9% -> 4.7% -> 98.6%: the policy
passes through a valley where the old open-loop trajectory is gone and the gain
that replaces it does not exist yet, while the lift decays from the first
checkpoint onward. No checkpoint in the run has both abilities. There is nothing
left to select; something has to be changed.

THE SUSPECT: _held IS A BOOLEAN, AND IT MULTIPLIES FIVE TERMS AT ONCE.

    near = distance < hold_distance            # hard threshold at 0.030
    stopped_by_cube = bump > 0.5               # hard threshold
    return near & stopped_by_cube

object_lifted, cube_upright, lifting_progress and object_goal_distance are each
multiplied by that boolean. So the instant a cube shifts a couple of millimetres
between the fingers, five reward channels go to zero together -- not reduced,
zero -- and they come back only when the grip re-seats.

WHY THIS EXPLAINS WHAT cube_upright COULD NOT. The cube_upright story never
accounted for the one fact that most needs accounting for: why RANDOMIZING THE
SPAWN specifically triggers the loss, at every offset including the control
where nothing changed. The boolean does. With a fixed spawn every grasp is very
nearly the same grasp, so the flag is stable and a lift rarely trips it. Widen
the spawn and the grip varies episode to episode, the flag starts flickering,
and it flickers hardest during exactly the motion that disturbs a grip most --
accelerating upward. A low, still hold is then not merely a good earner, it is
the only way to keep all five channels switched on at once. That is a policy
being paid to stop lifting, which is what the ep_rew_mean 66 -> 96.5 climb
across the collapse already told us was happening.

THE ONE CHANGE: the same two conditions, made continuous.

    near        -> sigmoid((hold_distance - d) / NEAR_WIDTH)
    stopped     -> sigmoid((bump - 0.5) / BUMP_WIDTH)
    held        -> product, in [0, 1]

The widths are deliberately tight (4 mm, 0.08 of bump) so that a clearly-held
cube still scores ~0.97 and a clearly-unheld one ~0.00. The gate is NOT being
loosened -- what it counts as held is unchanged to within a couple of
millimetres. What is removed is the cliff at the boundary. A lift that shifts
the cube slightly now costs a fraction of the reward instead of all of it, so
the gradient at the boundary points back toward re-seating the grip rather than
toward never going near the boundary.

This deliberately does NOT add hysteresis, which is the other half of the idea
and would need per-env state carried across steps. That would be a second
variable, and this week has already paid twice for runs that moved two things.
If the smooth gate works, hysteresis can be tried on top of it, on evidence.

WARM START FROM stage20_final, NOT stage16_final. Stage 21 started from stage
16's weights because the question was what a reward term costs from a common
origin. The question here is different: generalization is the thing worth
keeping, stage 20 is the only policy that has it, and the run is asking whether
the lift can be recovered ON TOP of it. Starting from stage 16 would be asking
whether a smooth gate reaches the same place, which is a slower question.

REGISTERED PREDICTIONS, BEFORE THE RUN:

  1. If the boolean is the mechanism, the lift trajectory should stop decaying.
     A stage 22 whose lift merely falls MORE SLOWLY than stage 20's has not
     confirmed anything -- that is the trap stage 20's own final numbers set,
     and it is why lift_trajectory.sh is not optional here.
  2. Tilt should rise somewhat. A grip is disturbed by lifting, and paying
     partial credit for a disturbed grip is precisely what this change does.
  3. FALSIFICATION: the lift still decays to ~2-3 cm across training. Three
     explanations would then have failed -- the squareness weight, the gate
     structure, and checkpoint selection -- and the honest conclusion is that
     the trade is intrinsic to this reward family under a randomized spawn.
     Ship stage 20 and stop buying the lift back.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week3"))

import torch
from isaaclab.utils import configclass

import grasp_reward
from stage20_cfg import FrankaLiftStage20Cfg

# Softening widths. Tight on purpose: these reproduce the boolean's verdict
# everywhere except within ~2 sigma of its own thresholds, so the run tests the
# CLIFF and not a looser definition of "held".
NEAR_WIDTH = 0.004   # metres of end-effector distance
BUMP_WIDTH = 0.08    # units of _on_cube_bump, whose threshold is 0.5

_HELD_BOOLEAN = grasp_reward._held


def _held_smooth(env, hold_distance, object_cfg, ee_frame_cfg, robot_cfg):
    """_held with the two step functions replaced by sigmoids of the same
    thresholds. Returns a float tensor in [0, 1]; every call site multiplies by
    `.float()`, which is a no-op on a float tensor, so no caller changes.

    Reference points, against the boolean it replaces:
        d = 15 mm, bump 0.90  ->  0.97   (boolean 1)
        d = 30 mm, bump 0.50  ->  0.25   (boolean 0, at the cliff edge)
        d = 31 mm, bump 0.48  ->  0.19   (boolean 0)
        d = 45 mm, bump 0.20  ->  0.00   (boolean 0)
    """
    robot = env.scene[robot_cfg.name]
    distance = grasp_reward._ee_cube_distance(env, object_cfg, ee_frame_cfg)
    bump = grasp_reward._on_cube_bump(grasp_reward._finger_sum(robot))
    near = torch.sigmoid((hold_distance - distance) / NEAR_WIDTH)
    stopped = torch.sigmoid((bump - 0.5) / BUMP_WIDTH)
    return near * stopped


@configclass
class FrankaLiftStage22Cfg(FrankaLiftStage20Cfg):
    """Stage 20 exactly -- randomized spawn, every weight, the 0.030 hold
    distance -- with the hold gate made continuous instead of boolean.

    The change is applied by rebinding grasp_reward._held rather than by editing
    week3/grasp_reward.py, so stages 16 and 20 remain reproducible from this
    tree and eval_shift.py is untouched. The rebind is module-global and
    therefore reaches all five gated terms at once, which is the intended single
    variable: the question is about the gate's SHAPE, not about any one term.
    """

    def __post_init__(self):
        super().__post_init__()
        grasp_reward._held = _held_smooth


def restore_boolean_gate():
    """Undo the rebind. Nothing in the training path needs this -- it exists so
    an interactive session that imports this module cannot silently leave the
    smooth gate installed for a later stage-20 reproduction."""
    grasp_reward._held = _HELD_BOOLEAN


# NO SEPARATE EVAL CONFIG, for stage 21's reason: eval_shift.py replays the
# policy deterministically and judges cube geometry, so reward terms take no
# part in the measurement. Stage 20, 21 and 22 are therefore all scored by the
# same instrument, which is the only way their numbers can be compared.
