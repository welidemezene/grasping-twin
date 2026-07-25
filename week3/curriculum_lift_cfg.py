"""Stage 3: push from grip to LIFT. Easier gate, cheaper grip, lift stays the big prize."""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

import grasp_reward


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

        # Change 1: easier lift gate  2.5cm -> 2.0cm
        self.rewards.lifting_object.params["minimal_height"] = 0.020
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.020
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.020

        # Reaching stays a small breadcrumb
        self.rewards.reaching_object.weight = 2.0

        # Change 2: grip pays less so holding-forever isn't the smart move  8.0 -> 4.0
        self.rewards.grasping_object = RewTerm(
            func=grasp_reward.object_is_grasped,
            params={"grasp_distance": 0.03},
            weight=4.0,
        )

        # Change 3: lifting is already weight 15 by default — the biggest prize. Left as is.
