# Activity Tracker

Wearable activity classifier + step counter for the Uno Q + MPU6050 (worn on
the ankle). The base classifier distinguishes **standing still / walking /
running / biking** using two models built without neural nets; on top of
that, a small PyTorch-trained neural net counts steps in real time.

## How the base classifier works

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

## How the step counter works

Step counting is a separate two-stage pipeline, built on top of the same raw
IMU stream:

1. **Peak detection** (`python/steps.py`): a classic signal-processing step,
   the same idea any basic pedometer uses - find local maxima in
   acceleration magnitude that are tall enough and far enough apart to
   plausibly be a footstrike. This catches every *candidate* step, but it's
   easily fooled: pedaling, bumps, and pivoting your foot all produce
   similar-looking peaks.
2. **StepNet**, a small neural net (`python/step_net.py`), looks at 4 numbers
   describing each candidate peak (how tall it was, how much it stood out
   from the recent baseline, how much rotation was happening, how long since
   the last one) and decides: real step, or false positive?

Two independent implementations train the exact same network shape:

- `python/train_step_net.py` - trains with **real PyTorch** (autograd,
  `.backward()`, `Adam`) on a computer, with k-fold cross-validation for an
  honest accuracy estimate. This is the canonical training pipeline.
- `python/step_net_train_numpy.py` - a from-scratch reimplementation of the
  same training, with every gradient written out by hand via the chain rule
  instead of PyTorch's autograd. PyTorch itself is too large to install on
  this board (its only wheel here pulls in a multi-gigabyte NVIDIA/CUDA
  stack this ARM board has no use for), so this powers the **Retrain Step
  Net** button on the phone page for quick in-the-field recalibration.

Either way, the trained weights are just 3 small matrices + biases, saved to
`python/data/step_net_weights.json`. At inference time
(`python/step_net_inference.py`) the board runs the forward pass with plain
numpy - a handful of matrix multiplies and two simple functions (ReLU,
sigmoid) - so nothing about what runs live needs PyTorch or is a black box.

## How the neural net is working (in plain terms)

StepNet takes 4 numbers describing one candidate peak - how tall it is, how
much it stands out above the recent baseline, how much rotation the gyro saw
at that moment, and how long since the last candidate - and passes them
through two small hidden layers (8 neurons, then 4) before a single output
neuron squashed to a number between 0 and 1 with a sigmoid function: the
estimated probability that this peak was a real step. Each neuron just
computes a weighted sum of its inputs plus a bias, then applies a simple
nonlinear function (ReLU, which is 0 for negative inputs and passes positive
ones through unchanged) - stacking a few of these is what lets the network
learn a decision boundary more flexible than a single straight-line
threshold, which is exactly where our old hand-tuned Model 1 struggled to
tell pedaling from walking.

Training starts with random weights, so the network's first guesses are
close to random. Backpropagation is how it improves: after a forward pass
produces a prediction, we compare it to the true label (step or not-a-step)
with a loss function (binary cross-entropy) that's small when the
prediction is confident and correct, large when it's confidently wrong.
Backprop then works backward through the network one layer at a time,
using the chain rule from calculus to compute exactly how much each
individual weight contributed to that error - literally "how much would the
loss change if I nudged this one number slightly." Gradient descent then
nudges every weight a small step in the direction that reduces the loss.
PyTorch's `.backward()` call does this chain-rule bookkeeping automatically;
our from-scratch numpy version (used for on-device retraining) writes out
those same derivative formulas by hand, layer by layer, which is a good way
to confirm we actually understand what autograd is doing under the hood
rather than just trusting it.

Repeating this thousands of times (forward pass, measure the error,
backprop the blame to every weight, nudge them all slightly) is what turns
those random starting weights into ones that actually separate real steps
from false positives. We measure how well it generalizes with k-fold
cross-validation rather than just training accuracy - with our current
dataset, the network separates steps from non-steps correctly on data it
wasn't trained on around 89-90% of the time, which is honest enough to
quote in a writeup, and noticeably better than the simple threshold rule
could do on its own, especially at telling pedaling and foot-pivoting apart
from real footstrikes.

## Using it

1. Deploy: `python3 tools/deploy.py activity-tracker` (or Cmd+Shift+B in VS
   Code with a file from this app open → "Run on UNO Q").
2. Open the printed URL (`http://<board-ip>:7000`) on your phone, on the same
   Wi-Fi as the board.
3. **Base classifier (optional, not needed for the step counter)**: for each
   activity, tap "Record \<activity\>", do that activity for ~20-30s, then
   tap "Stop Recording". Repeat, then tap "Train Models".
4. **Step counter training**: use the "Step Training Data" card - "Record
   Steps" while walking/running/jogging, "Record Non-Step Motion" while
   doing anything that *isn't* a step (pivoting, tapping your foot, shaking
   or bumping the sensor, standing still). The more varied non-step motion
   you capture, the better it learns to reject false positives.
5. Retrain with real PyTorch on your computer (`python3
   build_step_dataset.py && python3 train_step_net.py`, then redeploy) for
   the official numbers, or tap **Retrain Step Net** on the page for a quick
   on-device recalibration.
6. The **Steps** card shows the live count, with a gold marker on the
   acceleration/gyro graphs at the moment each step was detected.

Calibration data is stored in `python/data/calibration.csv`; raw IMU traces
in `python/data/raw_imu_log.csv`; step-candidate training data in
`python/data/step_candidates.csv`. All reload automatically on restart.

## Uncertainty analysis (how good is it?)

The **Test Walk** card on the page is the uncertainty analysis tool: reset
the step count, walk a known number of steps while counting them yourself,
then enter that number. It logs `predicted` vs. `actual` to
`python/data/step_test_results.csv` and shows the running mean absolute
error and mean percent error across every trial so far - do several trials
at different paces (slow walk, normal walk, jog) for a real picture of where
it's accurate and where it isn't, rather than a single cherry-picked number.

## Reflection questions (starting points - personalize these for your writeup)

- **Where does the step counter fail most?** Likely answer based on how it
  was built: very slow, shuffling steps (peak may not clear the detection
  threshold) and fast pivoting/direction changes early in training before
  enough non-step examples were collected - check your own Test Walk trials
  and the per-activity candidate counts in `step_candidates.csv` to confirm.
- **Why use a neural net here instead of another hand-tuned threshold?** A
  single threshold on peak height can't separate "a gentle step" from "an
  aggressive foot pivot" when they produce similar peak heights - the net
  can combine multiple signals (height, prominence, rotation, timing)
  nonlinearly in a way one threshold can't.
- **What would most improve accuracy with more time?** More and more varied
  training data (different shoes, different surfaces, faster/slower paces),
  and potentially feeding the net a short raw waveform snippet around each
  peak instead of just 4 summary numbers.
- **What's the tradeoff between the PyTorch training script and the
  on-device numpy retrain button?** The PyTorch version gets proper k-fold
  cross-validation and is the "ground truth" training process; the on-device
  version trades some rigor (a single 80/20 split, fewer epochs worth of
  patience) for the convenience of recalibrating without a computer.

## Video / animation

[How StepNet Learns](https://claude.ai/artifact/1GVGsp2qk3jWkYEZF8zrtq) is an
animated, 6-scene walkthrough of the whole pipeline - peak detection, the
network's forward pass, the loss, backpropagation, and training - built to
be screen-recorded into your submission video (hit Play and let it run, or
step through with Next). Source is in `docs/stepnet_explainer.html`. For
your own hardware demo video: show the Test Walk card logging a real trial,
and the gold step markers lining up with your actual footsteps on the live
graphs.
