"""Linux-side entry point for the Activity Tracker app.

Talks to the MCU over the Bridge to read the MPU6050 (see ../sketch/sketch.ino),
keeps a rolling window of samples, runs both classifiers, and serves a phone
web page for calibration + live predictions.
"""
import json
import time
import threading
from collections import deque
from pathlib import Path

from arduino.app_utils import App, Logger, brick, call
from arduino.app_bricks.web_ui import WebUI

from features import extract_features, single_axis_peaks
from models import Model1Threshold, Model2KNN
from calibration import CalibrationManager
from steps import StepPeakDetector, STEP_FEATURE_KEYS
from step_net_inference import StepNetInference
from build_step_dataset import load_sessions, build_candidates, RAW_LOG_PATH
import step_net_train_numpy as step_train

logger = Logger("ActivityTracker")

SAMPLE_HZ = 50
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLE_HZ * WINDOW_SECONDS)

# MPU6050 default full-scale ranges: +-2g for accel, +-250 deg/s for gyro.
ACCEL_SCALE = 1.0 / 16384.0  # g per raw LSB
GYRO_SCALE = 1.0 / 131.0     # deg/s per raw LSB

LABELS = ["standing", "walking", "running", "biking"]


# Each of these calls the matching function the sketch registered with
# Bridge.provide() (see sketch/sketch.ino) and blocks for its return value.
@call()
def read_ax() -> int: ...
@call()
def read_ay() -> int: ...
@call()
def read_az() -> int: ...
@call()
def read_gx() -> int: ...
@call()
def read_gy() -> int: ...
@call()
def read_gz() -> int: ...


@brick
class ImuSampler:
    """Polls the MCU at SAMPLE_HZ and keeps the most recent WINDOW_SIZE
    samples, converted to physical units."""

    def __init__(self, window_size):
        self.buffer = deque(maxlen=window_size)
        self._unprocessed = []  # samples the step counter hasn't drained yet (see drain_new)
        self._lock = threading.Lock()
        self.connected = True

    def loop(self):
        try:
            sample = (
                read_ax() * ACCEL_SCALE, read_ay() * ACCEL_SCALE, read_az() * ACCEL_SCALE,
                read_gx() * GYRO_SCALE, read_gy() * GYRO_SCALE, read_gz() * GYRO_SCALE,
            )
            self.connected = True
        except Exception as e:
            if self.connected:
                logger.error(f"Lost contact with the sketch over the Bridge: {e}")
            self.connected = False
            time.sleep(0.5)
            return
        with self._lock:
            self.buffer.append(sample)
            self._unprocessed.append((time.time(), sample))
        time.sleep(1.0 / SAMPLE_HZ)

    def window(self):
        with self._lock:
            return list(self.buffer)

    def drain_new(self):
        """Returns every sample added since the last call, in order, then
        clears that list - so a slower-running consumer (the step counter)
        sees each sample exactly once instead of repeating or skipping any,
        no matter how its own loop timing drifts relative to this one.
        """
        with self._lock:
            new = self._unprocessed
            self._unprocessed = []
            return new


sampler = ImuSampler(WINDOW_SIZE)
model1 = Model1Threshold()
model2 = Model2KNN(k=5)
calib = CalibrationManager(sampler, model1, model2, WINDOW_SIZE)

state = {
    "prediction1": "collecting data...",
    "prediction2": "collecting data...",
    "loo_accuracy": None,
}


@brick
class LiveClassifier:
    """Twice a second, classifies the current window with both models -
    unless we're recording calibration data, in which case stay out of the
    way so the recording loop gets the CPU."""

    def loop(self):
        time.sleep(0.5)
        if calib.is_recording:
            return
        window = sampler.window()
        if not sampler.connected:
            state["prediction1"] = state["prediction2"] = "sketch not responding"
            return
        if len(window) < WINDOW_SIZE:
            state["prediction1"] = state["prediction2"] = "collecting data..."
            return
        peaks = single_axis_peaks(window)
        feats = extract_features(window)
        state["prediction1"] = model1.predict(peaks)
        state["prediction2"] = model2.predict(feats)


live_classifier = LiveClassifier()

step_weights_path = Path(__file__).resolve().parent / "data" / "step_net_weights.json"
step_net = StepNetInference(step_weights_path) if step_weights_path.exists() else None


@brick
class StepCounter:
    """Drains new raw samples from ImuSampler, runs them through the peak
    detector, and asks StepNet whether each detected peak was a real step.
    Keeps a running total that only resets when you ask it to."""

    def __init__(self):
        self.detector = StepPeakDetector(sample_hz=SAMPLE_HZ)
        self.count = 0
        # Timestamps (time.time()) of recently counted steps, so the page can
        # draw a marker at the right spot on the live accel/gyro graphs.
        self.recent_step_times = deque(maxlen=50)

    def loop(self):
        time.sleep(0.02)
        if step_net is None:
            return
        for t, (ax, ay, az, gx, gy, gz) in sampler.drain_new():
            candidate = self.detector.feed(t, ax, ay, az, gx, gy, gz)
            if candidate is not None and step_net.is_step(candidate):
                self.count += 1
                self.recent_step_times.append(t)

    def reset(self):
        self.detector = StepPeakDetector(sample_hz=SAMPLE_HZ)
        self.count = 0
        self.recent_step_times.clear()


step_counter = StepCounter()

# The app's files land at /app/<app-name's local layout>, so our python/
# folder is /app/python, not /app - WebUI's /app/assets default doesn't
# match our layout, so point it at the folder's actual location.
web_ui = WebUI(assets_dir_path="/app/python/assets")


def get_status():
    return {
        "connected": sampler.connected,
        "prediction1": state["prediction1"],
        "prediction2": state["prediction2"],
        "recording": calib.is_recording,
        "recording_label": calib.recording_label,
        "rows_recorded_this_session": calib.rows_recorded_this_session,
        "labels": LABELS,
        "counts_by_label": calib.counts_by_label(),
        "loo_accuracy": state["loo_accuracy"],
        "step_count": step_counter.count,
        "step_net_loaded": step_net is not None,
        "step_times": list(step_counter.recent_step_times),
    }


def get_sample():
    """Most recent raw accel+gyro reading, for the live graphs on the page.

    Kept separate from get_status() so the page can poll it faster without
    re-fetching the (slower-changing) calibration counts and predictions.
    """
    window = sampler.window()
    if not window:
        return {"connected": sampler.connected, "sample": None, "t": None}
    return {"connected": sampler.connected, "sample": list(window[-1]), "t": time.time()}


def start_recording(label: str):
    if label not in LABELS:
        return {"ok": False, "error": f"unknown label '{label}'"}
    started = calib.start_recording(label)
    return {"ok": started}


def stop_recording():
    calib.stop_recording()
    return {"ok": True}


def start_step_recording():
    """Dedicated step-training recording - walk/run however you like.
    Logs only the raw trace (not a Model 1/2 feature snapshot)."""
    return {"ok": calib.start_recording("step")}


def start_nonstep_recording():
    """Dedicated step-training recording for anything that ISN'T a step:
    pivoting, tapping your foot, shaking or bumping the sensor, etc."""
    return {"ok": calib.start_recording("non_step")}


def train():
    n_rows = calib.train()
    state["loo_accuracy"] = model2.leave_one_out_accuracy()
    return {"ok": True, "rows": n_rows, "loo_accuracy": state["loo_accuracy"]}


def erase_calibration():
    calib.erase()
    state["prediction1"] = "collecting data..."
    state["prediction2"] = "collecting data..."
    state["loo_accuracy"] = None
    return {"ok": True}


web_ui.expose_api("GET", "/status", get_status)
web_ui.expose_api("GET", "/sample", get_sample)
web_ui.expose_api("POST", "/start_recording", start_recording)
web_ui.expose_api("POST", "/stop_recording", stop_recording)
web_ui.expose_api("POST", "/start_step_recording", start_step_recording)
web_ui.expose_api("POST", "/start_nonstep_recording", start_nonstep_recording)
web_ui.expose_api("POST", "/train", train)
web_ui.expose_api("POST", "/erase_calibration", erase_calibration)


def reset_steps():
    step_counter.reset()
    return {"ok": True}


web_ui.expose_api("POST", "/reset_steps", reset_steps)


def retrain_step_net():
    """Rebuilds the step-candidate dataset from raw_imu_log.csv and retrains
    StepNet right here on the board, using a hand-written gradient descent
    (step_net_train_numpy.py) instead of PyTorch - PyTorch itself is too
    heavy to install on this board's ARM Linux. train_step_net.py (run on a
    computer) does the same training with real PyTorch for the writeup;
    this is a lighter from-scratch reimplementation of the same math, for
    quick in-the-field recalibration from the phone page.
    """
    global step_net
    import numpy as np

    if not RAW_LOG_PATH.exists():
        return {"ok": False, "error": "no step training data recorded yet"}

    sessions = load_sessions()
    candidates, _ = build_candidates(sessions)
    if len(candidates) < 10:
        return {"ok": False, "error": f"only {len(candidates)} candidate peaks found - record more data first"}

    X = np.array([[c[k] for k in STEP_FEATURE_KEYS] for c in candidates])
    y = np.array([[c["is_step"]] for c in candidates], dtype=float)

    # A shuffled 80/20 holdout for a quick, honest accuracy estimate - not a
    # full k-fold like the Mac script, to keep this fast enough to run live.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    split = max(1, int(len(idx) * 0.8))
    train_idx, val_idx = idx[:split], idx[split:]

    train_mean, train_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    train_std[train_std == 0] = 1.0
    holdout_weights = step_train.train((X[train_idx] - train_mean) / train_std, y[train_idx])
    val_accuracy = None
    if len(val_idx):
        X_val_n = (X[val_idx] - train_mean) / train_std
        val_accuracy = step_train.accuracy(holdout_weights, X_val_n, y[val_idx])

    # Final model for deployment, trained on everything.
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    final_weights = step_train.train((X - mean) / std, y)

    payload = step_train.to_export_payload(final_weights, mean, std)
    with open(step_weights_path, "w") as f:
        json.dump(payload, f, indent=2)

    step_net = StepNetInference(step_weights_path)  # hot-swap the live model
    return {"ok": True, "candidates": len(candidates), "holdout_accuracy": val_accuracy}


web_ui.expose_api("POST", "/retrain_step_net", retrain_step_net)

# WebUI is itself a brick (like ImuSampler and LiveClassifier above), so
# App.run() starts its web server automatically - no manual web_ui.start().
logger.info(f"Web UI will be available at {web_ui.url}")

App.run()
