"""Stage 23 — the envelope is a band 6 cm wide in x. Widen the box.

`probe_envelope.py` spread 512 cubes over a +-15 cm box and binned the outcome by
where each one landed. stage22_final scored 36.3% overall, and the shape of the
failure is the finding:

         -0.129 -0.086 -0.043  0.000  0.043  0.086  0.129
  +0.129      0%     0%     0%    33%   100%    29%     0%
  +0.086      0%     0%    10%    70%    86%    11%     0%
  +0.043      0%     0%    38%    91%    69%     0%     0%
  +0.000      0%     0%   100%    92%   100%     0%     0%
  -0.043      0%    17%   100%   100%    62%     0%     0%
  -0.086      0%     0%   100%    92%    22%    10%     0%
  -0.129     20%    67%    82%   100%    67%     0%     0%

A VERTICAL BAND, NOT A DISC. It holds the full +-13 cm in y at centre x --
including 100% at y = -0.129, more than twice the training range, so y
generalized well past what it was taught -- and collapses to zero past about
|x| = 0.06 in BOTH directions.

FAILING NEAR AND FAR ALIKE IS THE POINT. "Cannot reach that far" cannot produce a
failure at x = -0.13, where the cube is 13 cm CLOSER to the base. And the joint
numbers agree: 0.01 arm joints pinned at a limit on failures, identical to
successes. Nothing is anywhere near a limit anywhere on that map.

  => POLICY WALL, NOT KINEMATIC. The arm can reach every square it fails in.

That is the same under-powered x-tracking gain diagnose_reach.py measured in
stage 16 (+x 4 cm: 47% compensation, 21 mm short, zero joints pinned), surviving
into stage 22 at a larger radius. Randomization strengthened the gain enough for
5 cm and no further -- which is exactly what training on a +-5 cm box should be
expected to buy.

THE ONE CHANGE: SPAWN_RANGE 0.05 -> 0.12.

Everything else is stage 22's: the smooth hold gate, every reward weight, the
0.030 hold distance, ent_coef, learning rate, sticky hold. Stage 20 established
that widening the spawn distribution alone is a clean, interpretable change; this
is the same change one notch further out, and it inherits FrankaLiftStage22Cfg so
the gate rebind still happens in super().__post_init__().

WHY 0.12 AND NOT 0.15. The probe's outermost bins are past the point where the
table's own geometry starts to matter, and a cube spawned at 15 cm in both axes
is 21 cm diagonally from home -- three times any distance this policy has ever
handled. 0.12 doubles the diagonal reach requirement, which is already a large
ask for one run, and it keeps every spawn comfortably on the table. If it works,
0.15 is the next notch and it is cheap to test with the same probe.

WHY NOT WIDEN ONLY X, since y is nearly free. Two reasons. It would make the
train and test distributions differently shaped, so a y regression would be
invisible until something else caught it. And a square box is the same knob stage
20 turned, which keeps this run comparable to that one rather than being a new
kind of intervention.

PREDICTIONS REGISTERED BEFORE THE RUN. This project's rule, after two falsified
hypotheses and one confirmed:

  1. THE X BAND WIDENS toward +-0.12. That is the whole point; if it does not
     move, widening the distribution is not sufficient and the x gain needs a
     different mechanism than more examples.
  2. THE LIFT DROPS AGAIN, at least early. Going from a fixed spawn to +-5 cm
     cost 13.5 cm -> 2.6 cm because a varied grip makes the hold gate flicker.
     More variety should push the same way.
  3. IF THE SMOOTH GATE IS THE REAL MECHANISM, the lift RECOVERS by the end of
     the run, as it did in stage 22 (0.032 -> 0.030 -> 0.066 -> 0.144). Judge the
     SHAPE across checkpoints, not the final number alone.
  4. FALSIFIED IF the lift decays and stays down while the band widens. The
     smooth gate would then be a fix that holds only at +-5 cm, and the honest
     statement becomes that this reward family trades lift for envelope at a
     fixed exchange rate -- ship stage 22 and stop widening.
  5. Y SHOULD NOT REGRESS. It already works to +-13 cm. If it gets worse while x
     improves, the policy is trading one axis for the other and the capacity,
     not the distribution, is the binding constraint.

JUDGE IT WITH, in this order:
  1. probe_envelope.py stage23_final s23_wide --range 0.15
     Same instrument, same box, directly comparable to the map above. This is
     the primary verdict: does the band widen.
  2. sweep_shift.sh checkpoints/stage23_final s23
     The six fixed offsets, >90% and >10 cm. Comparable to stage 22's row.
  3. lift_trajectory.sh s23_traj stage23
     The shape of the lift across checkpoints, for predictions 2 and 3.

A NOTE ON WHAT PASSING WOULD MEAN. Nothing here transfers to the SO-101 as
weights -- different body, ~6 outputs against this one's 8. What transfers is the
finding: an envelope has a shape, the shape is measurable in one run, and the
wall type decides whether retraining is worth starting at all.
"""

from isaaclab.utils import configclass

from stage22_cfg import FrankaLiftStage22Cfg

SPAWN_RANGE = 0.12          # stage 20 and 22 used 0.05


@configclass
class FrankaLiftStage23Cfg(FrankaLiftStage22Cfg):
    """Stage 22 exactly -- smooth hold gate, every weight -- with a wider spawn box.

    Inherits from the stage 22 config rather than copying it, so the gate rebind
    in FrankaLiftStage22Cfg.__post_init__ still happens. A copy that missed the
    rebind would train stage 20's boolean gate under stage 23's name, and the two
    would then be compared as though the gate were held constant.
    """

    def __post_init__(self):
        super().__post_init__()
        # LAST, so it overrides stage 20's 0.05 rather than being overridden by
        # it. super() walks Stage22 -> Stage20, and Stage20's __post_init__ sets
        # this key; assigning before the super() call would be silently undone.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-SPAWN_RANGE, SPAWN_RANGE),
            "y": (-SPAWN_RANGE, SPAWN_RANGE),
            "z": (0.0, 0.0),
        }


# NO SEPARATE EVAL CONFIG, for stage 21 and 22's reason: eval_shift.py replays
# the policy deterministically and judges cube geometry, so reward terms take no
# part in the measurement. Stages 20 through 23 are all scored by the same
# instrument, which is the only way their numbers can be compared.
