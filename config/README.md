# Configuration Files

All runtime configuration lives here. Edit these YAML files to select
models, tune thresholds, and adjust camera parameters — no source code
changes needed.

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
