"""Logs test-walk trials (how many steps the counter said vs. how many you
actually took) and computes summary error stats - the uncertainty analysis
for the writeup, built from real trials instead of one claimed number.
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LOG_PATH = DATA_DIR / "step_test_results.csv"
FIELDS = ["predicted", "actual", "error", "abs_error", "pct_error"]


def log_trial(predicted: int, actual: int):
    DATA_DIR.mkdir(exist_ok=True)
    is_new = not LOG_PATH.exists()
    error = predicted - actual
    pct_error = (abs(error) / actual * 100) if actual else None
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "predicted": predicted, "actual": actual,
            "error": error, "abs_error": abs(error), "pct_error": pct_error,
        })


def load_trials():
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["predicted"] = int(r["predicted"])
        r["actual"] = int(r["actual"])
        r["error"] = int(r["error"])
        r["abs_error"] = int(r["abs_error"])
        r["pct_error"] = float(r["pct_error"]) if r["pct_error"] not in (None, "", "None") else None
    return rows


def summary():
    """Mean absolute error and mean percent error across every trial
    logged so far - the two numbers that answer "how good is it?"."""
    trials = load_trials()
    if not trials:
        return {"n_trials": 0, "mean_abs_error": None, "mean_pct_error": None, "recent_trials": []}

    abs_errors = [t["abs_error"] for t in trials]
    pct_errors = [t["pct_error"] for t in trials if t["pct_error"] is not None]
    return {
        "n_trials": len(trials),
        "mean_abs_error": sum(abs_errors) / len(abs_errors),
        "mean_pct_error": (sum(pct_errors) / len(pct_errors)) if pct_errors else None,
        "recent_trials": trials[-10:],
    }
