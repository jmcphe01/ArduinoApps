"""Step (foot-strike) detection from the raw accelerometer/gyro stream.

Two-stage design, same idea as Model 1/Model 2 in models.py but for a
different problem:

  1. A classic signal-processing peak detector (this file) finds CANDIDATE
     step events - local maxima in acceleration magnitude - the same simple
     technique any basic pedometer uses. It's cheap and fast but easily
     fooled: pedaling, bumps, and arm swings all produce accel peaks too.
  2. A small neural net (step_net.py) looks at a few numbers describing each
     candidate (how tall the peak was, how much rotation was happening, how
     long since the last one) and decides whether it's a REAL step or a
     false positive. That's the part a hand-written threshold struggles
     with, and what we need the net for.

This same StepPeakDetector class is used both offline (to build the
training set from raw_imu_log.csv) and live on the board (one sample at a
time, in main.py) - using the same code in both places means the model
is being trained on exactly the signal it will see live.
"""
import math
from collections import deque

# What the neural net sees for each candidate peak, in a fixed order - this
# list IS the net's input vector definition (see step_net.py).
STEP_FEATURE_KEYS = ["peak_mag", "peak_prominence", "gyro_mag_at_peak", "interval_since_last_peak"]

# StepNet's layer sizes - lives here (not step_net.py) so step_net_train_numpy.py
# can use them without importing torch, which isn't installed on the board.
STEP_NET_INPUT_DIM = len(STEP_FEATURE_KEYS)
STEP_NET_HIDDEN1 = 8
STEP_NET_HIDDEN2 = 4

HISTORY_SECONDS = 2.0       # how far back we look to judge "is this tall for recent motion"
REFRACTORY_SECONDS = 0.25   # minimum gap between peaks - no human steps faster than 4/sec
PROMINENCE_FLOOR = 0.03     # ignore peaks barely above the local baseline (sensor noise)


class StepPeakDetector:
    """Feed it samples one at a time (in order); it returns a candidate
    feature dict whenever the sample it saw *previous* call turns out to
    have been a local peak, or None otherwise.

    (We only know a point was a peak once we've also seen the point after
    it, hence the one-sample lag - this works identically for offline
    replay and for a live stream.)
    """

    def __init__(self, sample_hz):
        self.history = deque(maxlen=max(int(sample_hz * HISTORY_SECONDS), 1))
        self.prev_mag = None
        self.prev_prev_mag = None
        self.prev_gyro_mag = None
        self.last_peak_t = None

    def feed(self, t, ax, ay, az, gx, gy, gz):
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)

        candidate = None
        if self.prev_mag is not None and self.prev_prev_mag is not None and len(self.history) >= 2:
            is_local_max = self.prev_mag > self.prev_prev_mag and self.prev_mag > mag
            if is_local_max:
                baseline = sum(self.history) / len(self.history)
                prominence = self.prev_mag - baseline
                far_enough = self.last_peak_t is None or (t - self.last_peak_t) >= REFRACTORY_SECONDS
                if prominence > PROMINENCE_FLOOR and far_enough:
                    interval = (t - self.last_peak_t) if self.last_peak_t is not None else 1.0
                    candidate = {
                        "peak_mag": self.prev_mag,
                        "peak_prominence": prominence,
                        "gyro_mag_at_peak": self.prev_gyro_mag,
                        "interval_since_last_peak": interval,
                    }
                    self.last_peak_t = t

        self.history.append(mag)
        self.prev_prev_mag = self.prev_mag
        self.prev_mag = mag
        self.prev_gyro_mag = gyro_mag
        return candidate
