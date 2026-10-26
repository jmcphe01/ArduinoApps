"""Linux-side entry point for the Activity Tracker app.

Talks to the MCU over the Bridge to read the MPU6050 (see ../sketch/sketch.ino),
keeps a rolling window of samples, runs both classifiers, and serves a phone
web page for calibration + live predictions.
"""
import time
import threading
from collections import deque

from arduino.app_utils import App, Logger, brick, call
from arduino.app_bricks.web_ui import WebUI

from features import extract_features, single_axis_peaks
from models import Model1Threshold, Model2KNN
from calibration import CalibrationManager

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
        time.sleep(1.0 / SAMPLE_HZ)

    def window(self):
        with self._lock:
            return list(self.buffer)


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
    }


def start_recording(label: str):
    if label not in LABELS:
        return {"ok": False, "error": f"unknown label '{label}'"}
    started = calib.start_recording(label)
    return {"ok": started}


def stop_recording():
    calib.stop_recording()
    return {"ok": True}


def train():
    n_rows = calib.train()
    state["loo_accuracy"] = model2.leave_one_out_accuracy()
    return {"ok": True, "rows": n_rows, "loo_accuracy": state["loo_accuracy"]}


web_ui.expose_api("GET", "/status", get_status)
web_ui.expose_api("POST", "/start_recording", start_recording)
web_ui.expose_api("POST", "/stop_recording", stop_recording)
web_ui.expose_api("POST", "/train", train)

# WebUI is itself a brick (like ImuSampler and LiveClassifier above), so
# App.run() starts its web server automatically - no manual web_ui.start().
logger.info(f"Web UI will be available at {web_ui.url}")

App.run()
