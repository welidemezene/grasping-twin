"""Stage 3: push from grip to LIFT. Easier gate, cheaper grip, lift stays the big prize."""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

import grasp_reward

# a cube resting flat has its centre at 0.0210 m; tipped onto a corner that
# becomes 0.0210 * sqrt(3) = 0.0364, which clears a bare 0.035 height check —
# so height alone is never enough to call something a lift
LIFT_HEIGHT = 0.035
HOLD_DISTANCE = 0.05


@configclass
class FrankaLiftStage1Cfg(FrankaCubeLiftEnvCfg):
    """reach (small) -> grip (medium, cheaper now) -> LIFT (big prize)."""

    def __post_init__(self):
        super().__post_init__()

        # Cube spawns in the SAME spot every episode
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }

        # Reaching stays a small breadcrumb
        self.rewards.reaching_object.weight = 2.0

        # Change 2: grip pays less so holding-forever isn't the smart move  8.0 -> 4.0
        self.rewards.grasping_object = RewTerm(
            func=grasp_reward.object_is_grasped,
            params={"grasp_distance": 0.03},
            weight=4.0,
        )

        # The three height-gated terms below are worth 36 between them, and the
        # stock versions pay for height however it happens. A replay showed the
        # policy earning them by batting the cube into a tumble with the hand
        # wide open, which pays far better than grasping and is far easier to
        # find. Each now requires the cube to be held.
        self.rewards.lifting_object = RewTerm(
            func=grasp_reward.object_lifted_in_hand,
            params={"minimal_height": LIFT_HEIGHT, "hold_distance": HOLD_DISTANCE},
            weight=15.0,
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=grasp_reward.object_goal_distance_in_hand,
            params={
                "std": 0.3,
                "minimal_height": LIFT_HEIGHT,
                "command_name": "object_pose",
                "hold_distance": HOLD_DISTANCE,
            },
            weight=16.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=grasp_reward.object_goal_distance_in_hand,
            params={
                "std": 0.05,
                "minimal_height": LIFT_HEIGHT,
                "command_name": "object_pose",
                "hold_distance": HOLD_DISTANCE,
            },
            weight=5.0,
        )
