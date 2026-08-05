"""Stage 15 config: give the lift a ramp instead of a cliff.

Stage 14's grasp is real -- fingers stop at 0.0429 against a cube-stop point of
0.0420 -- and it carries the cube 13.2 mm off the table. It earns nothing for
that. `object_is_lifted` is a step function at 0.035 m, so from the robot's
point of view 1 mm and 13 mm of lift are identical: zero. There is no gradient
pulling upward, and the only way to discover the lift is to stumble over the
whole threshold at once.

That is exactly the bug that kept the gripper shut for nine stages, moved one
layer up. The fix is the same in spirit: make the wanted behaviour reachable by
small improvements rather than by luck.

So `lifting_progress` ramps continuously from the resting height while the cube
is held, and the original step bonus stays on top as the prize worth aiming
for. Everything else is inherited unchanged from the stage 10-14 config.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

import grasp_reward
from curriculum_lift_cfg import FrankaLiftStage1Cfg, LIFT_HEIGHT, HOLD_DISTANCE

REST_HEIGHT = 0.0210   # cube centre when it sits flat on the table


@configclass
class FrankaLiftStage15Cfg(FrankaLiftStage1Cfg):
    """Stage 14's rewards plus a continuous ramp under the lift threshold."""

    def __post_init__(self):
        super().__post_init__()

        # The ramp. tanh(h / 0.02) means the first millimetre already pays and
        # 13 mm -- stage 14's best -- is worth ~0.57 of the term instead of 0.
        # Weight 8 sits below the 15 of the step bonus, so crossing the gate is
        # still clearly better than hovering under it.
        self.rewards.lifting_progress = RewTerm(
            func=grasp_reward.lifting_progress_in_hand,
            params={
                "rest_height": REST_HEIGHT,
                "std": 0.02,
                "hold_distance": HOLD_DISTANCE,
            },
            weight=8.0,
        )
