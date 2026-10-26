# Activity Tracker

Wearable activity classifier for the Uno Q + MPU6050 (worn on the ankle).
Classifies **standing still / walking / running / biking** using two models,
built without neural nets, and lets you calibrate + watch live predictions
from a phone browser.

## How it works

- `sketch/sketch.ino` (MCU, C++): reads the MPU6050 over I2C continuously and
  hands the latest raw values to Python on request, over the Bridge.
- `python/main.py` + `python/features.py` + `python/models.py` +
  `python/calibration.py` (Linux side): buffers samples into 2-second
  windows, computes features, and runs:
  - **Model 1**: a threshold rule using only the peak-to-peak acceleration of
    one accelerometer axis (chosen automatically during training).
  - **Model 2**: a hand-rolled k-nearest-neighbors classifier using features
    from all 6 IMU channels (3 accelerometer + 3 gyroscope).
- `python/assets/index.html`: the phone page. Served at `http://<board-ip>:7000`.
  Lets you record calibration data per activity, train both models, and watch
  their live predictions update every second.

## Using it

1. Deploy: `python3 tools/deploy.py activity-tracker` (or Cmd+Shift+B in VS
   Code with a file from this app open → "Run on UNO Q").
2. Open the printed URL (`http://<board-ip>:7000`) on your phone, on the same
   Wi-Fi as the board.
3. For each activity, tap "Record \<activity\>", do that activity for ~20-30s,
   then tap "Stop Recording". Repeat for standing, walking, running, biking.
4. Tap "Train Models". The live predictions below update automatically from
   then on.

Calibration data is stored in `python/data/calibration.csv` (created on first
recording) and is reloaded automatically if the app restarts.
