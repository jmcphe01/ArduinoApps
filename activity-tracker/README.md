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

StepNet takes four numbers describing one candidate peak - its height, how
far it stands above the recent baseline, how much rotation the gyro saw at
that instant, and how long since the last candidate - and runs them through
two hidden layers of 8 and 4 neurons before a single output neuron. Each
neuron computes a weighted sum: it multiplies every input it receives by its
own learned weight, adds them all up along with one more learned number (the
bias), and that single sum is the neuron's raw output. On its own that's
just a linear combination, no more powerful than a weighted average, so
every hidden neuron passes its sum through ReLU, which outputs the sum
unchanged if it's positive and zero if it's negative. That one simple cutoff
is what makes the network nonlinear - without it, stacking layers would
collapse mathematically into one big linear equation, no more expressive
than Model 1's single threshold. The final neuron uses sigmoid instead of
ReLU, squeezing its output into a number between 0 and 1, which we read as
"probability this peak is a real step."

Right after creation, every weight and bias is a small random number, so the
network's first guesses are essentially noise. Training is the process of
correcting those numbers using labeled examples, and it has three parts.
First, the forward pass above produces a prediction. Second, we score it
with a loss function (binary cross-entropy) that returns a small number
when the prediction matches the true label confidently and a large number
when it doesn't. Third, backpropagation computes the gradient - for every
single weight and bias in the network, a number that says "if I increased
this one value by a tiny amount, the loss would change by roughly this
much, in this direction." It gets those numbers by applying the chain rule
from calculus starting at the output and moving backward: the output
layer's gradient is cheap to compute directly from the loss, and each
earlier layer's gradient is built from the gradient the layer after it
already produced, which is why it's called backpropagation and not just
"propagation." PyTorch's `.backward()` performs that entire chain-rule
computation automatically; the from-scratch numpy version we wrote for
on-device retraining does the identical arithmetic by hand, layer by layer,
which is really the only way to be sure you understand what autograd is
doing rather than trusting it as a black box.

Gradient descent is what actually uses those gradients: every weight gets
nudged a small step in the opposite direction of its gradient, since the
gradient points toward increasing loss and we want to decrease it. The size
of that step is the learning rate - too large and the weights overshoot and
bounce around; too small and training crawls. One update barely moves the
needle, but repeating the full cycle (forward pass, compute loss,
backpropagate, update every weight) several hundred times is what actually
separates "random numbers" from "a network that distinguishes a footstep
from a pedal stroke." The honest way to check whether that worked is k-fold
cross-validation - training on most of the data and testing on a held-out
slice it never saw, repeated across several different slices - rather than
trusting accuracy on the same data it trained on, which a small network can
simply memorize without learning anything general. By that more honest
measure, ours lands around 89-90%.

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
