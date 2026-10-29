"""Turns raw_imu_log.csv into a labeled dataset of step-candidate peaks.

Run on a computer (not the board): python3 build_step_dataset.py

How the labels come from data we already have: the raw log is tagged with
the SAME activity label as the existing calibration recordings (standing /
walking / running / biking). During walking and running, a real footstrike
happens on every detected peak, so every candidate from those sessions is
labeled a real step (1). During standing and biking there is no footstrike
at all - any peak the detector fires on there is, by definition, a false
positive (noise, a shift in the strap, the pedaling motion) - so those
candidates are labeled not-a-step (0). No extra manual step-by-step tagging
needed; we're reusing the activity labels you already recorded.
"""
import csv
import json
from pathlib import Path

from steps import StepPeakDetector, STEP_FEATURE_KEYS

RAW_LOG_PATH = Path(__file__).resolve().parent / "data" / "raw_imu_log.csv"
OUT_PATH = Path(__file__).resolve().parent / "data" / "step_candidates.csv"

RAW_SAMPLE_HZ = 20  # matches RAW_INTERVAL_SECONDS in calibration.py
# "step"/"non_step" are dedicated step-training recordings (see main.py);
# walking/running/standing/biking are the original activity recordings,
# reused here as weak supervision since they're still valid extra data.
STEP_LABELS = {"walking", "running", "step"}            # every detected peak here = a real step
NON_STEP_LABELS = {"standing", "biking", "non_step"}     # every detected peak here = a false positive


def load_sessions():
    """Splits the raw log back into individual recording sessions.

    Each Record/Stop click writes its own contiguous block with its own
    `t` starting near 0 - so a new session starts wherever `t` goes
    backwards (or the label changes). Keeping sessions separate matters:
    feeding two unrelated recordings into one continuous detector would
    create a fake, huge time gap "peak" right at the seam.
    """
    with open(RAW_LOG_PATH, newline="") as f:
        rows = [
            {**r, "t": float(r["t"]), "ax": float(r["ax"]), "ay": float(r["ay"]), "az": float(r["az"]),
             "gx": float(r["gx"]), "gy": float(r["gy"]), "gz": float(r["gz"])}
            for r in csv.DictReader(f)
        ]

    sessions = []
    current = []
    for r in rows:
        if current and (r["label"] != current[-1]["label"] or r["t"] < current[-1]["t"]):
            sessions.append(current)
            current = []
        current.append(r)
    if current:
        sessions.append(current)
    return sessions


def build_candidates(sessions):
    """Runs the peak detector over each session and labels every candidate
    it finds. Used by both the CLI script below and the board's live
    /retrain_step_net endpoint, so they can never drift apart."""
    candidates = []
    skipped_labels = set()
    for session in sessions:
        label = session[0]["label"]
        if label in STEP_LABELS:
            is_step = 1
        elif label in NON_STEP_LABELS:
            is_step = 0
        else:
            skipped_labels.add(label)
            continue

        detector = StepPeakDetector(sample_hz=RAW_SAMPLE_HZ)
        for r in session:
            c = detector.feed(r["t"], r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"])
            if c is not None:
                c["is_step"] = is_step
                c["source_label"] = label
                candidates.append(c)
    return candidates, skipped_labels


def main():
    sessions = load_sessions()
    print(f"Found {len(sessions)} recording sessions in {RAW_LOG_PATH.name}")

    candidates, skipped_labels = build_candidates(sessions)
    if skipped_labels:
        print(f"Skipped unrecognized label(s): {skipped_labels}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_FEATURE_KEYS + ["is_step", "source_label"])
        writer.writeheader()
        writer.writerows(candidates)

    n_pos = sum(c["is_step"] for c in candidates)
    n_neg = len(candidates) - n_pos
    print(f"Wrote {len(candidates)} candidate peaks to {OUT_PATH.name}: {n_pos} real steps, {n_neg} non-steps")
    by_label = {}
    for c in candidates:
        by_label[c["source_label"]] = by_label.get(c["source_label"], 0) + 1
    print("By source activity:", json.dumps(by_label, indent=2))


if __name__ == "__main__":
    main()
