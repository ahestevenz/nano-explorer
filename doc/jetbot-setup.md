# 🤖 Waveshare JetBot Setup — Jetson Nano 4GB (Maxwell)

A complete initialisation guide for the Waveshare JetBot on a Jetson Nano 4GB (Maxwell) running JetPack 4.6.

---

## Prerequisites

- Jetson Nano 4GB (Maxwell architecture)
- JetPack 4.6 (provides CUDA 10.2, Python 3.6)
- Waveshare JetBot kit with CSI camera
- Internet connection on the Nano

---

## Step 1 -- Flash JetPack 4.6.1

1. Download the SD card image from NVIDIA:
   `jetson-nano-jp461-sd-card-image.zip` from https://developer.nvidia.com/jetson-nano-sd-card-image
2. Flash to a ≥ 64 GB microSD card using **balenaEtcher** or `dd`.
3. Insert the card, connect a display + keyboard, and boot.
4. Complete first-boot setup (create user, set timezone, etc.).
5. Confirm your L4T version:
   ```bash
   head -1 /etc/nv_tegra_release
   # Expected: R32 (release), REVISION: 7.1 ...
   ```

---

## Step 2 — Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-smbus \
    libfreetype6-dev \
    python3-pil \
    libi2c-dev \
    i2c-tools \
    cmake \
    curl \
    libopenblas-base \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev
```

---

## Step 3 — Configure I2C & User Permissions

Add your user to the required groups and enable the I2C kernel module:
```bash
sudo usermod -aG i2c $USER
sudo usermod -aG video $USER

# Load the i2c-dev module for this session
sudo modprobe i2c-dev

# Persist across reboots
echo "i2c-dev" | sudo tee -a /etc/modules
```

> **Note:** Log out and back in (or reboot) for group changes to take effect.

---

## Step 4 — Install PyTorch for JetPack 4.6

Download and install the pre-built PyTorch 1.8.0 wheel for CUDA 10.2 / Python 3.6:
```bash
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
     -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

---

## Step 5 — Build & Install torchvision from Source

> ⚠️ The PyPI `torchvision` package won't work on Jetson — it must be compiled locally:

```bash
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0
sudo python3 setup.py install
cd ..
```

---

## Step 6 — Install JetBot

Use the [ahestevenz fork](https://github.com/ahestevenz/jetbot), which includes the TensorRT 8.x `FlattenConcat` compatibility fix already applied — no manual patching required.
```bash
cd ~
git clone https://github.com/ahestevenz/jetbot
cd jetbot
sudo python3 setup.py install
```
---

## Step 7 — Install jetson-inference

Clone the repository:

```bash
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
cd jetson-inference
```

Before building, two patches are required to make the CMake build compatible
with **numpy 1.19.5** on JetPack 4.6.1. The `npymath` static library is not
present in this numpy version and must be removed from both binding targets.

**Patch 1** — `python/bindings/CMakeLists.txt`:

```bash
python3 - << 'PYEOF'
import re
path = "python/bindings/CMakeLists.txt"
with open(path) as f:
    content = f.read()
patched = re.sub(r'(\s*)(.*npymath.*)', r'\1# \2', content)
with open(path, "w") as f:
    f.write(patched)
print("Patch 1 applied.")
PYEOF
```

**Patch 2** — `utils/python/bindings/CMakeLists.txt`:

```bash
python3 - << 'PYEOF'
path = "utils/python/bindings/CMakeLists.txt"
with open(path) as f:
    lines = f.readlines()
new_lines = [
    "  # npymath removed — not available with numpy 1.19.5 on JetPack 4.6.1\n"
    if "npymath" in line and not line.strip().startswith("#")
    else line
    for line in lines
]
with open(path, "w") as f:
    f.writelines(new_lines)
print("Patch 2 applied.")
PYEOF
```

Verify both patches are clean before building:

```bash
grep -rn "npymath" python/bindings/CMakeLists.txt \
                   utils/python/bindings/CMakeLists.txt \
  | grep -v "^.*:.*#"
# Should print nothing
```

Build and install:

```bash
mkdir build && cd build
cmake -DPYTHON3_PACKAGES=ON ..
make -j4
sudo make install
sudo ldconfig
```

Verify the installation:

```bash
python3 -c "import jetson_inference; print('jetson_inference OK')"
python3 -c "import jetson_utils; print('jetson_utils OK')"
```

> **Note:** The older `jetson.inference` / `jetson.utils` dot-notation imports
> still work but are deprecated. Use `jetson_inference` and `jetson_utils`
> (underscore) in new code.

---

## Step 8 — Install torch2trt and trt_pose (Pose Estimation)

Required for `nano-explorer vision pose` and `nano-explorer vision gesture`.

### Dependencies

```bash
pip3 install tqdm cython pycocotools
```

### Install torch2trt

```bash
cd ~/code
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
git checkout 9a048b0
python3 setup.py install   # use your virtualenv python3, not sudo
```

### Install trt_pose

```bash
cd ~/code
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python3 setup.py install
```

### Copy the topology file into the package

trt_pose does not install `human_pose.json` into the package directory.
Copy it manually so `pose.py` can find it:

```bash
cp ~/code/trt_pose/tasks/human_pose/human_pose.json \
   ~/.virtualenvs/nano_explorer_py36/lib/python3.6/site-packages/trt_pose-0.0.1-py3.6-linux-aarch64.egg/trt_pose/
```

### Download model weights

The official Google Drive link is currently inaccessible.
Download the weights manually from a browser on your laptop:

---

## Step 9 — Install ORB-SLAM2

Required for `nano-explorer map slam`.

> **Warning — known fragile (2025).** There is no actively maintained ORB-SLAM2
> Python bindings repo. The only publicly accessible option is
> `jskinn/ORB_SLAM2-PythonBindings`, which targets Python 3.5 / Ubuntu 14–16 and
> requires manual CMakeLists patching for Python 3.6 and modern Boost.
> Expect friction. The ecosystem has largely moved on to ORB-SLAM3, which has
> better (though still unofficial) Python bindings — worth considering if you are
> willing to take on the higher build complexity.

### System dependencies

```bash
sudo apt install -y \
    libglew-dev \
    libepoxy-dev \
    libeigen3-dev \
    libboost-dev libboost-thread-dev libboost-filesystem-dev \
    libboost-python-dev \
    libopencv-dev \
    cmake
```

### Build Pangolin (viewer dependency)

> Pin to **v0.6** — newer Pangolin master requires GCC 9+ and is incompatible
> with the GCC 7 shipped in JetPack 4.6.

```bash
cd ~/code
git clone --branch v0.6 https://github.com/stevenlovegrove/Pangolin
cd Pangolin
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
sudo make install
sudo ldconfig
```

### Build ORB-SLAM2 Python bindings

Clone the (pre-patched) bindings repo and ORB-SLAM2 core:
```bash
cd ~/code
git clone git@github.com:ahestevenz/ORB_SLAM2-PythonBindings.git
git clone git@github.com:ahestevenz/ORB_SLAM2.git
```

> `ahestevenz/ORB_SLAM2` is a fork of `raulmur/ORB_SLAM2` with the OpenCV4/Eigen3/
> Pangolin compatibility fixes and the PythonBindings integration
> (`orbslam-changes.diff`) already applied. `ahestevenz/ORB_SLAM2-PythonBindings`
> is a fork of `jskinn/ORB_SLAM2-PythonBindings` with the Boost Python component
> name fixed for Python 3.6 and the OpenCV4 `pyboost_cv3_converter.cpp` fixes
> already applied — no manual patching needed for either. Diff against each
> repo's `upstream` remote if you need to see exactly what changed.

Build ORB-SLAM2 core:

> **Important:** build outside the virtualenv. The pip-installed cmake inside
> the virtualenv is self-contained and cannot find system libraries.
> `deactivate` before running `build.sh`; re-activate only for the Python
> bindings install below.

```bash
deactivate

# Install system cmake if not already present
sudo apt install -y cmake

cd ~/code/ORB_SLAM2
chmod +x build.sh

# Clear any stale build dirs from previous failed attempts
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build

./build.sh          # ~30 min on the Nano; uses -j internally — watch for OOM
```

Build and install the Python bindings:
```bash
cd ~/code/ORB_SLAM2-PythonBindings

mkdir build && cd build
# ORBSlamPython.cpp uses #include <ORB_SLAM2/KeyFrame.h>.
# Headers live at ORB_SLAM2/include/KeyFrame.h (no prefix), so create the
# expected subdirectory via a symlink and point ORB_SLAM2_INCLUDE_DIR one level up.
ln -sf ~/code/ORB_SLAM2/include ~/code/ORB_SLAM2/ORB_SLAM2
cmake \
    -DORB_SLAM2_DIR=$HOME/code/ORB_SLAM2 \
    -DORB_SLAM2_INCLUDE_DIR=$HOME/code/ORB_SLAM2 \
    -DORB_SLAM2_LIBRARIES=$HOME/code/ORB_SLAM2/lib/libORB_SLAM2.so \
    -DPYTHON_EXECUTABLE=$(which python3) \
    ..
make -j2
sudo cp ../lib/orbslam2.so $(python3 -c "import site; print(site.getsitepackages()[0])")
```

Verify:
```bash
python3 -c "import orbslam2; print('orbslam2 OK')"
```

### Download the ORB vocabulary file

Required at runtime — path is set via `vocabulary` in `config/models/slam.yaml`:
```bash
mkdir -p ~/code/nano-explorer/assets/models
wget https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz \
     -O /tmp/ORBvoc.txt.tar.gz
tar -xf /tmp/ORBvoc.txt.tar.gz -C ~/code/nano-explorer/assets/models/
```

---

## Step 10 — Verify Hardware

### Camera
```bash
ls /dev/video*
```

### I2C devices (motor driver)
```bash
sudo i2cdetect -y -r 1
```

The motor driver (typically at address `0x40` or `0x60`) should appear in the grid.

---

## Step 11 — Verify Camera with GStreamer + OpenCV

Run this quick test to confirm the CSI camera pipeline is working end-to-end:
```bash
python3 - <<'EOF'
import cv2

pipeline = (
    'nvarguscamerasrc sensor-id=0 sensor-mode=3 ! '
    'video/x-raw(memory:NVMM), width=(int)1640, height=(int)1232, '
    'format=(string)NV12, framerate=(fraction)30/1 ! '
    'nvvidconv flip-method=0 ! '
    'video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true'
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print('Opened:', cap.isOpened())
ret, frame = cap.read()
print('Read:', ret)
if ret:
    print('Frame shape:', frame.shape)  # Expected: (224, 224, 3)
cap.release()
EOF
```

**Expected output:**
```
Opened: True
Read: True
Frame shape: (224, 224, 3)
```

---

## Troubleshooting

### Camera fails after reboot (dead CSI port on I2C bus 8)

**Root cause:** A dead CSI port on I2C bus 8 can crash the `nvargus-daemon` on its first use after boot.

**Fix — restart the daemon before any camera usage:**
```bash
sudo systemctl restart nvargus-daemon
sleep 2
```

**Better fix — add a warm-up script to your startup routine:**

Create `/usr/local/bin/warmup-camera.sh`:
```bash
#!/bin/bash
echo "Warming up camera..."
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=10 ! \
  'video/x-raw(memory:NVMM), format=NV12' ! \
  nvvidconv ! fakesink 2>/dev/null
echo "Camera ready."
```
```bash
chmod +x /usr/local/bin/warmup-camera.sh
```

Run this once after every boot before starting your JetBot application. It clears the daemon state reliably.

---

## Quick Reference

| Component | Version |
|-----------|---------|
| JetPack | 4.6 |
| CUDA | 10.2 |
| Python | 3.6 |
| PyTorch | 1.8.0 |
| torchvision | 0.9.0 |
| Architecture | Maxwell (Jetson Nano 4GB) |


For basic run:
pip install 'pynput==1.7.6'

# One-time setup — enable uinput for your user
sudo modprobe uinput
sudo chmod a+rw /dev/uinput
