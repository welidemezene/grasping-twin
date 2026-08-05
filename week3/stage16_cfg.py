"""Stage 16 config: lift it HIGH, and carry it SQUARE.

Stage 15 measured honestly (check_airborne.py, 512 parallel trials, cube pose
rather than cube height):

    ever grasped            512/512  (100%)
    grasped AND airborne    507/512  (99.0%)
    median lift while held  0.0099 m
    cube tilt               up to 79 degrees

So the hard part is solved -- it grasps every time and gets the cube off the
table almost every time. What is left is that the lift is minimal and ugly.
Two causes, one fix each:

1. THE RAMP SATURATES. lifting_progress was tanh(h / 0.02), which is ~0.96 by
   4 cm, so a 15 cm lift paid barely more than a 2 cm one. With nothing to gain
   above the minimum, the policy settled at the minimum: a 9.9 mm median.
   Now most of the weight sits on a far ramp, tanh(h / 0.10), which keeps
   paying to ~20 cm. The near ramp survives at lower weight so the fine
   gradient that first found the lift is not thrown away.

2. NOTHING CARED ABOUT ORIENTATION. Tilting the cube 79 degrees cost nothing,
   so a corner pinch scored the same as a clean grip. cube_upright_in_hand pays
   for keeping a face level while held, scaled so a corner-balance is worth
   zero and a square carry is worth full marks.

Kept additive rather than multiplied. Stages 1-9 taught that lesson the hard
way: multiplying two channels together made progress in one punishable by the
other, and every long run retreated into the valley it created.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

import grasp_reward
from curriculum_lift_cfg import FrankaLiftStage1Cfg, HOLD_DISTANCE

REST_HEIGHT = 0.0210


@configclass
class FrankaLiftStage16Cfg(FrankaLiftStage1Cfg):
    """Stage 15's rewards, with height that keeps paying and a squareness term."""

    def __post_init__(self):
        super().__post_init__()

        # Height, now with a far ramp carrying most of the weight. Weight up
        # from 8 to 12 because this is the behaviour still missing.
        self.rewards.lifting_progress = RewTerm(
            func=grasp_reward.lifting_progress_in_hand,
            params={
                "rest_height": REST_HEIGHT,
                "std": 0.02,        # near ramp: fine gradient off the table
                "far_std": 0.10,    # far ramp: still paying at 20 cm
                "far_mix": 0.6,
                "hold_distance": HOLD_DISTANCE,
            },
            weight=12.0,
        )

        # Carry it square. Deliberately smaller than the height term -- lifting
        # badly must still beat not lifting, or the robot will simply stop
        # trying rather than risk the penalty.
        self.rewards.cube_upright = RewTerm(
            func=grasp_reward.cube_upright_in_hand,
            params={"hold_distance": HOLD_DISTANCE},
            weight=4.0,
        )
