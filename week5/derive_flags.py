"""Per-frame grasp/lift flags for the viewer — one definition, one place.

Split out of record_motion_w4.py so the recorder and the browser agree, and so a
threshold can be corrected without spending 12 minutes of GPU re-recording. The
raw fields (joints, cube, ee) are physics output and never change; these flags
are derived, so they can be recomputed from any existing recording:

    python3 derive_flags.py motion/*.json

WHY THIS FILE EXISTS AT ALL — A BUG WORTH KEEPING WRITTEN DOWN. The first
version tested "lifted" as cube z > 0.035 m, carried over from week 3's summary
script. The cube RESTS at z = 0.0521 m. So the test was true while the cube sat
untouched on the table, and the first eight recordings all reported a held lift,
including stage20_final whose actual displacement was MINUS 3.5 mm. A summary
line read "HELD LIFT: 205 frames, max -0.0035 m" and that contradiction is the
only reason it got caught.

The lesson is the same one week 4 kept re-learning: a threshold in absolute world
coordinates is a threshold whose meaning depends on where the table is. Lift is
therefore measured RELATIVE to the cube's own resting height in frame 0.

NOT THE AUTHORITATIVE AIRBORNE TEST. week4/eval_shift.py decides "airborne" from
cube-corner geometry over 512 trials, and that is what every published rate in
this project comes from. These flags exist to label ONE episode for a viewer, and
a 2 cm margin on the cube's centre is a deliberately conservative stand-in — half
the cube's own width, so a cube being shoved along the table cannot pass.
"""
import json
import sys

# A grasp is fingers stopped BY the cube, not a shut fist. An empty fist sums to
# about 0.004, so week 3's original `finger_sum < 0.005` test could only ever
# detect closing on nothing.
GRASP_FINGER_SUM = 0.042
GRASP_TOLERANCE = 0.012
GRASP_MAX_GAP = 0.030

# Relative to the cube's resting height in frame 0 — never an absolute z.
LIFT_MARGIN_M = 0.020


def annotate(frames):
    """Add finger_sum, gap, grasped, lifted to each frame. Returns an episode dict."""
    if not frames:
        raise ValueError("no frames")

    rest_z = frames[0]["cube"][2]

    for fr in frames:
        fr["finger_sum"] = fr["joints"][7] + fr["joints"][8]
        fr["gap"] = sum((fr["ee"][i] - fr["cube"][i]) ** 2 for i in range(3)) ** 0.5
        fr["lift"] = fr["cube"][2] - rest_z
        fr["grasped"] = bool(abs(fr["finger_sum"] - GRASP_FINGER_SUM) < GRASP_TOLERANCE
                             and fr["gap"] < GRASP_MAX_GAP)
        fr["lifted"] = bool(fr["grasped"] and fr["lift"] > LIFT_MARGIN_M)

    grasp_frames = [i for i, fr in enumerate(frames) if fr["grasped"]]
    lift_frames = [i for i, fr in enumerate(frames) if fr["lifted"]]
    worst = min(range(len(frames)), key=lambda i: frames[i]["finger_sum"])

    return {
        "rest_z_m": round(rest_z, 4),
        "grasped": bool(grasp_frames),
        "lifted": bool(lift_frames),
        "first_grasp_frame": grasp_frames[0] if grasp_frames else None,
        "first_lift_frame": lift_frames[0] if lift_frames else None,
        "max_lift_m": round(max(fr["lift"] for fr in frames), 4),
        "max_lift_while_held_m": round(max((frames[i]["lift"] for i in lift_frames), default=0.0), 4),
        "min_gap_m": round(min(fr["gap"] for fr in frames), 4),
        "empty_fist": bool(frames[worst]["finger_sum"] < 0.01),
        "lift_margin_m": LIFT_MARGIN_M,
        "note": "One episode. Not a success rate -- see week4/*_sweep.csv for the 512-trial verdict.",
    }


def summary_text(meta):
    """The human-readable summary, written from meta so the two cannot disagree."""
    ep = meta["episode"]
    out = [
        f"checkpoint {meta['checkpoint']}",
        f"offset x {meta['shift_x']:+.3f} y {meta['shift_y']:+.3f} "
        f"(diagonal {meta['shift_diagonal_m']:.4f} m)   hold {meta['hold']}   frames {meta['frames']}",
        f"cube rests at z {ep['rest_z_m']:.4f} m; lift is measured from there",
        f"gripper-cube gap: min {ep['min_gap_m']:.4f} m",
    ]
    if ep["grasped"]:
        out.append(f"GRASP: first at frame {ep['first_grasp_frame']}")
        if ep["lifted"]:
            out.append(f"HELD LIFT: first at frame {ep['first_lift_frame']}, "
                       f"max {ep['max_lift_while_held_m']:.4f} m while held")
        else:
            out.append(f"NO HELD LIFT: grasped, but never cleared {ep['lift_margin_m']:.3f} m "
                       f"above rest (best {ep['max_lift_m']:+.4f} m)")
    else:
        out.append("NO GRASP")
        if ep["empty_fist"]:
            out.append("  -> empty fist: it closed on nothing")
    out.append("")
    out.append("One episode, one offset. The verdict at this offset is in week4/*_sweep.csv.")
    return "\n".join(out) + "\n"


def main(paths):
    for path in paths:
        with open(path) as f:
            doc = json.load(f)

        # Accept both shapes: week 3 wrote a bare list, week 5 writes an object.
        if isinstance(doc, list):
            doc = {"meta": {}, "frames": doc}

        meta = doc["meta"]
        meta["episode"] = annotate(doc["frames"])
        meta["frames"] = len(doc["frames"])
        meta.setdefault("checkpoint", path)
        for k, default in (("shift_x", 0.0), ("shift_y", 0.0), ("hold", 5)):
            meta.setdefault(k, default)
        meta.setdefault("shift_diagonal_m",
                        round((meta["shift_x"] ** 2 + meta["shift_y"] ** 2) ** 0.5, 4))

        with open(path, "w") as f:
            json.dump(doc, f)
        with open(path.replace(".json", "_summary.txt"), "w") as f:
            f.write(summary_text(meta))

        ep = meta["episode"]
        verdict = ("lift %+.4f m" % ep["max_lift_while_held_m"]) if ep["lifted"] else (
            "GRASPED, NO LIFT (best %+.4f m)" % ep["max_lift_m"] if ep["grasped"] else "NO GRASP")
        print(f"{path:44s} {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
