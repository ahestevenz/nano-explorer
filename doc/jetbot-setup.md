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

## Step 7 — Verify Hardware

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

## Step 8 — Verify Camera with GStreamer + OpenCV

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
