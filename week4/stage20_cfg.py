"""Stage 20 config — the cube stops being where the policy remembers it.

Week 3's result, stage16_final, lifts a cube 13.5 cm on 99.6% of 512 trials.
Then the render pipeline moved the scene 3.4 cm on Isaac Lab 3.0 and the same
checkpoint closed on empty air. Every stage from 1 to 19 trained against this
line in curriculum_lift_cfg.py:

    "Cube spawns in the SAME spot every episode (randomization is a later
     stage - first the grasp itself has to exist)"

The grasp exists. So the honest reading of the 3.4 cm failure is that twenty
stages taught a very good open-loop trajectory to one fixed point, and nothing
in the training distribution ever asked the policy to look at the cube. This is
not a reward bug -- it is the reward being satisfiable without perception.

THE ONLY CHANGE HERE IS THE SPAWN DISTRIBUTION. Rewards are stage 16's,
untouched, weights and all. That is deliberate:

  - Week 3 ended with a complete 2x2 over the hold gate and the cube_upright
    weight. Both levers independently destroy the lift. There is no untested
    cell worth running, and re-opening either while ALSO randomizing the spawn
    would repeat stage 18's mistake -- two variables moved at once, so the run
    can never isolate either. Stage 19 had to be run to undo that confusion.
  - If randomization alone recovers the lift, that is a clean result about
    generalization. If it does not, the next stage has exactly one suspect.

TRAIN RANGE vs TEST RANGE. The train range below is +-5 cm in x and y. The
success test (eval_shift.py) evaluates at shift values the policy never trained
on, including the 3.4 cm diagonal that broke it. Evaluating inside the training
distribution would measure memorisation with extra steps -- which is the same
class of error as judging a lift by cube height, and this project has already
paid for that one twice.

Z is left alone. The cube rests on a table; lifting it off a floating spawn
height is a different task, not a harder version of this one.
"""

import os
import sys

# week3 holds the base config, the reward module and stage 16's config. They are
# imported by bare name (grasp_reward, curriculum_lift_cfg), so week3 has to be
# importable, not just reachable by path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week3"))

from isaaclab.utils import configclass

from stage16_cfg import FrankaLiftStage16Cfg

# +-5 cm in the table plane. Chosen so the 3.4 cm diagonal that broke stage 16
# sits INSIDE the span the policy has seen variation across, while the specific
# test offsets stay held out. Wider than this and the cube starts leaving the
# arm's comfortable workspace, which confounds "cannot generalize" with "cannot
# reach".
SPAWN_RANGE = 0.05


@configclass
class FrankaLiftStage20Cfg(FrankaLiftStage16Cfg):
    """Stage 16's rewards exactly. The cube now moves."""

    def __post_init__(self):
        super().__post_init__()

        # This overrides the fixed (0,0,0) spawn that every stage 1-19 used.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-SPAWN_RANGE, SPAWN_RANGE),
            "y": (-SPAWN_RANGE, SPAWN_RANGE),
            "z": (0.0, 0.0),
        }


@configclass
class FrankaLiftShiftedCfg(FrankaLiftStage16Cfg):
    """Evaluation config: the cube sits at ONE fixed offset, chosen by the caller.

    Not a randomized config. To ask "does this survive a 3.4 cm shift" the shift
    has to be the same for all 512 environments, otherwise the 512 trials
    average over easy and hard offsets and the number means nothing in
    particular. eval_shift.py sets shift_x / shift_y before instantiating.
    """

    shift_x: float = 0.0
    shift_y: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"] = {
            "x": (self.shift_x, self.shift_x),
            "y": (self.shift_y, self.shift_y),
            "z": (0.0, 0.0),
        }
