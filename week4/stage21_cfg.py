"""Stage 21 — remove the term that pays for hovering.

Stage 20 solved generalization (+x 4 cm 41.8% -> 98.6%, 5 cm diagonal 41.0% ->
99.8%) and lost the lift doing it: 0.1345 m -> 0.0257 m at every offset, the
unshifted control included. Two probes on checkpoints already on disk showed
this was not undertraining but a continuous trade made THROUGHOUT the run:

    2.6M    74.0%   0.1029 m   39.4 deg
    3.39M   94.1%   0.0778 m   33.4 deg
    5.0M    96.5%   0.0358 m   29.6 deg
     10M   100.0%   0.0257 m   20.3 deg

Lift and tilt fall together as reliability rises, and ep_rew_mean climbed
66 -> 96.5 across exactly that span. The reward went UP while the task got
worse, so the policy was not drifting into a low square hold -- it was being
paid to.

THE SUSPECT, NAMED IN ADVANCE. cube_upright_in_hand pays for keeping a face
level WHILE HELD, at any height whatsoever. lifting_progress_in_hand requires
height. Once the spawn moves and a lift becomes a gamble, holding the cube
square just above the table is a guaranteed earner that beats gambling, and the
tilt curve falling in lockstep with the lift curve is that trade made visible.

This mechanism is not new -- week 3's stage 19 established it by isolating the
weight (4 -> 10 dropped the lift 124.3 mm -> 15.3 mm with the hold gate
untouched). What is new is that randomizing the spawn appears to strengthen it
enough that weight 4, which stage 16 survived, no longer is survivable.

WEIGHT 4 -> 0, NOT 4 -> 2. A half-step would leave both explanations alive if
the lift only partly recovers. Zero completes the axis week 3 started (4 and 10
are measured; 0 is not) and gives an unambiguous answer about what the term is
costing. If the lift comes back, the term is the price and the right weight can
be searched for afterwards, on evidence.

ONE VARIABLE FROM STAGE 20, which is now the baseline -- the spawn range stays
at +-5 cm, every other reward keeps stage 16's weights, the hold gate stays at
0.030. Stage 18 moved two and had to be undone by stage 19; that has now cost
this project a run once and will not cost it another.

REGISTERED PREDICTION, so it cannot become a post-hoc story: tilt WILL regress.
Stage 15, with no squareness term at all, tilted up to 79 degrees. If the lift
returns at the cost of tilt, that is the term doing exactly what it was added
for, and the result is a frontier to price rather than a failure. What would
falsify the whole explanation is the lift staying at ~2.6 cm with the term gone
-- in which case something other than cube_upright is paying for the hover, and
the next suspect is the hold gate's boolean multiplying five terms at once.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week3"))

from isaaclab.utils import configclass

from stage20_cfg import FrankaLiftStage20Cfg


@configclass
class FrankaLiftStage21Cfg(FrankaLiftStage20Cfg):
    """Stage 20 exactly -- randomized spawn and all -- minus the squareness term."""

    def __post_init__(self):
        super().__post_init__()

        # Removing the term outright rather than setting weight 0.0, so it costs
        # nothing to evaluate and cannot quietly contribute through a params
        # change later. The attribute is defined on FrankaLiftStage16Cfg.
        self.rewards.cube_upright = None


# NO SEPARATE EVAL CONFIG, deliberately. Stage 21 is scored with the existing
# eval_shift.py and its FrankaLiftShiftedCfg, unchanged: the policy is replayed
# deterministically and judged on cube geometry, so the reward terms play no
# part in the measurement, and the scene, observation space and action space are
# identical either way. Forking the instrument for one run would mean stage 20
# and stage 21 were scored by two different scripts -- which is precisely the
# kind of quiet second variable this week has been paying to avoid.
