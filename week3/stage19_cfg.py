"""Stage 19: keep the gate that lifts, fix only the tilt.

THE MEASURED RESULT OF THE BRACKET (512 trials each, check_airborne.py):

    hold gate   lift median   tilt median   grasped+airborne
    0.030          124.3 mm      40.2 deg       99.6%   <- stage 16
    0.020           23.1 mm      25.2 deg      100.0%   <- stage 18
    0.015           21.5 mm      72.7 deg       97.7%   <- stage 17

The bracket is refuted. There is no good middle: tightening the hold gate AT
ALL destroys the lift, and the relationship is monotonic in the gate, not
U-shaped. 0.02 lifted 23 mm, barely better than 0.015's 21 mm, against 0.03's
124 mm.

Stage 18 also killed the explanation I had for stage 17's collapse. I argued
cube_upright at weight 10 paid a guaranteed reward for hovering at rest and so
out-competed lifting. Stage 18 put that weight back to 4 and the lift did NOT
return. The weight was not the driver; the gate is doing essentially all of it.

The likely mechanism, consistent with all three runs: a fast, high lift makes
the cube shift and swing between the fingers, which pushes cube-centre-to-
fingertip distance past a tight gate. `_held` is a BOOLEAN multiplying five
reward terms, so the moment it drops the policy earns nothing -- the exact
behaviour we want is the one that switches its own reward off. A loose gate
tolerates that movement; a tight one punishes it.

So stop touching the gate. Stage 16's 0.03 is the only setting that produces a
real lift, and it is kept here unchanged.

What stage 16 is genuinely missing is squareness: 40.2 deg median tilt. Stage
17 raised cube_upright to 10 to fix that, and it worked in the only sense we
can measure it -- stage 18, at weight 4, tilted 25.2 deg while stage 17, at
weight 10, tilted 72.7. But BOTH of those ran with a tightened gate, so the
tilt weight has never been tested against the gate that actually lifts.

That is the one untested cell, and this is it:

    gate 0.03 (stage 16, untouched)  +  cube_upright weight 10

If tilt is the only real complaint about stage 16, fix tilt directly and leave
the gate alone. If this run keeps the ~12 cm lift and brings tilt under 30 deg,
week 3 is finished. If the lift collapses again, then cube_upright at 10 IS
harmful after all, stage 16 is the bankable result, and week 3 ends anyway.

Either outcome ends week 3. This is the last experiment.
"""

from isaaclab.utils import configclass

from stage16_cfg import FrankaLiftStage16Cfg

UPRIGHT_WEIGHT = 10.0   # the only change; hold_distance stays at stage 16's 0.03


@configclass
class FrankaLiftStage19Cfg(FrankaLiftStage16Cfg):
    """Stage 16 exactly, with squareness made to cost something."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.cube_upright.weight = UPRIGHT_WEIGHT

        # Print the gates so the log proves nothing else moved.
        gates = {}
        for name in dir(self.rewards):
            if name.startswith("_"):
                continue
            params = getattr(getattr(self.rewards, name, None), "params", None)
            if isinstance(params, dict) and "hold_distance" in params:
                gates[name] = params["hold_distance"]
        print("[stage19] cube_upright weight -> %.1f" % UPRIGHT_WEIGHT)
        print("[stage19] hold_distance UNCHANGED: %s" % gates)
