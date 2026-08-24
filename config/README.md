# Configuration Files

The YAML files under this directory are **bundled defaults**, shipped with the package. The
CLI never reads or writes them directly at runtime — the first time any command actually needs
a given config file, it's copied to `~/.nano-explorer/config/` (same relative layout, e.g.
`config/models/slam.yaml` → `~/.nano-explorer/config/models/slam.yaml`), and every command reads
and writes that copy from then on. This is what you should actually edit to select models, tune
thresholds, adjust camera parameters, or store a calibrated motor trim.

The split exists so a `pip install --upgrade` — which only ever touches the installed package's
own copy of this directory — can never silently wipe something you tuned or calibrated. Once a
file has been seeded into `~/.nano-explorer/config/`, it's never overwritten automatically; only
a missing file triggers a (one-time) copy. See `lib/settings.py`'s `user_config_path()` /
`ensure_user_config()` for the mechanism.

## config/models/

| File                 | Used by command              | Purpose |
|----------------------|------------------------------|---------|
| `detection.yaml`     | `vision detect`              | Object detection backend and model name |
| `face.yaml`          | `vision faces`               | Face/people detector backend |
| `segmentation.yaml`  | `vision segment`             | Segmentation model name |
| `pose.yaml`          | `vision pose`, `vision gesture` | trt_pose weights and engine paths |

### Common fields

```yaml
backend:    # "jetson-inference" | "opencv-dnn" | "haar" | "dnn"
model:      # Model name string or path to weights file
threshold:  # Confidence cutoff [0.0–1.0]
```

## config/motors/

| File         | Used by command                          | Purpose |
|--------------|-------------------------------------------|---------|
| `trim.yaml`  | every command that drives the motors       | Per-wheel power trim (`left_trim`/`right_trim`, 1.0 = no correction) |

Written by `tools/calibrate_motors.py`; see that script's docstring for the calibration
procedure. Safe to edit by hand too — just two floats.

## config/slam/

| File                  | Purpose |
|-----------------------|---------|
| `nano_camera.yaml`    | ORB-SLAM2 monocular camera intrinsics and ORB extractor parameters |

### Calibrating the camera

The default intrinsics are approximate.  For better SLAM quality, calibrate
your specific camera with a checkerboard:

```bash
# Using OpenCV (run on a computer with display)
python3 -c "
import cv2
import numpy as np
# … standard checkerboard calibration procedure
# See: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"
```

Then update `Camera.fx`, `Camera.fy`, `Camera.cx`, `Camera.cy`
and the distortion coefficients in `nano_camera.yaml`.
