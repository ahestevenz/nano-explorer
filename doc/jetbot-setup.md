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

## Step 9a — Install ORB-SLAM2

Required for `nano-explorer map slam`.

> **Warning — known fragile (2025).** There is no actively maintained ORB-SLAM2
> Python bindings repo. The only publicly accessible option is
> `jskinn/ORB_SLAM2-PythonBindings`, which targets Python 3.5 / Ubuntu 14–16 and
> requires manual CMakeLists patching for Python 3.6 and modern Boost.
> Expect friction. ORB-SLAM3 (Step 9b below) fixes the exact monocular
> initialization failure this JetBot hits most often — its post-triangulation
> map-point gate was lowered from 100 to 50 — and includes debug instrumentation
> for feature/match/RANSAC diagnostics. It has a higher build cost (notably a
> newer OpenCV requirement) but is the recommended path going forward.

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

Required at runtime — path is set via the `Vocabulary` key in
`config/models/orbslam2_mono.yaml` (or `orbslam3_mono.yaml` for ORB-SLAM3):
```bash
mkdir -p ~/code/nano-explorer/assets/models
wget https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz \
     -O /tmp/ORBvoc.txt.tar.gz
tar -xf /tmp/ORBvoc.txt.tar.gz -C ~/code/nano-explorer/assets/models/
```

---

## Step 9b — Install ORB-SLAM3

Recommended path for `nano-explorer map slam` going forward (see the warning in
Step 9). This step has two separate builds, done in order:

1. **C++ core** (`ahestevenz/ORB_SLAM3`) — produces `libORB_SLAM3.so` plus the
   standalone example binaries.
2. **Python bindings** (`ahestevenz/pyorbslam`, a fork of
   `edavalosanaya/pyorbslam`) — what nano-explorer actually imports
   (`from pyorbslam import orbslam3`). This is a separate build with its *own*
   vendored copy of ORB-SLAM3 (a distinct copy from the C++ core in the first
   part, with its own `shared_ptr`-based memory-leak fixes) — not a dependency
   on the core library built in the first part.

### C++ core — compile ORB_SLAM3

Builds `ahestevenz/ORB_SLAM3` (branch `jetson-b01-maxwell`) — a fork of
`UZ-SLAMLab/ORB_SLAM3` with debug instrumentation added for `nano-explorer`:
`System::GetLastInitDetections()`, `GetLastInitRawMatches()`, and
`GetLastInitInlierMatches()` expose the raw ORB feature count, raw ratio-test
match count, and RANSAC/cheirality-surviving match count for the most recent
monocular initialization attempt (all three return `-1` if that stage wasn't
reached). Call them right after `TrackMonocular()`, same as the existing
`GetTrackingState()`.

This part covers the **C++ core library only** (`libORB_SLAM3.so` + example
binaries); the Python bindings are the second part below.

#### OpenCV version — already handled, no rebuild needed

Upstream ORB-SLAM3's `CMakeLists.txt` hard-requires OpenCV ≥ 4.4
(`find_package(OpenCV 4.4)`, `FATAL_ERROR` otherwise) — stricter than
ORB-SLAM2's ≥3.0, and higher than JetPack 4.6.1's stock **OpenCV 4.1.1**.

This turned out to be a conservative floor, not a real API dependency: there's
no `CV_VERSION`/`CV_MAJOR_VERSION`-gated code anywhere in `src/` or `include/`
(only two log-line prints), and
[UZ-SLAMLab/ORB_SLAM3#456](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/456)
confirms other users already compile and run fine against OpenCV 4.2.0 by
just editing that one line. The `jetson-b01-maxwell` branch already
lowers it to `4.1.1` to match the Nano's stock version:

```cmake
find_package(OpenCV 4.1.1)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.1.1 not found.")
   endif()
```

So no OpenCV rebuild should be needed — confirm your stock version satisfies
it and move on:

```bash
pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv
```

> **If the main `ORB_SLAM3` build below fails on an actual missing OpenCV
> symbol** (not a `find_package` version error — a real compile error citing
> something undefined), that would mean 4.1.1 genuinely lacks something 4.2.0+
> has. In that case, fall back to building OpenCV ≥4.4 from source: remove
> apt's `libopencv-dev`/`python3-opencv` first (so its `.pc`/headers don't
> conflict), then build OpenCV 4.5.4 with `opencv_contrib`, `-D
> OPENCV_GENERATE_PKGCONFIG=ON`, `-D BUILD_opencv_python3=ON`, `-D
> WITH_CUDA=ON -D CUDA_ARCH_BIN=5.3`, using `make -j2` (not `-j4` — a
> well-known OOM source on a 4GB Nano); expect 2–3+ hours and re-verify with
> Step 11 afterward, since it replaces the `cv2` every other nano-explorer
> vision command depends on. This has not been needed in practice so far.

#### Additional system dependencies

Beyond what Step 9 already installed for ORB-SLAM2 (Pangolin v0.6, Eigen3,
Boost, OpenCV dev headers), ORB-SLAM3 additionally needs:

```bash
sudo apt install -y libboost-serialization-dev libssl-dev
```

`libboost-serialization-dev` backs the Atlas/map save-load functionality
(`boost::serialization` is used throughout `Atlas.h`/`Map.h`/`KeyFrameDatabase.h`);
`libssl-dev` backs `System.cc`'s `<openssl/md5.h>` use for map checksums. Eigen3
≥3.3.0 and CMake ≥3.4 are also required by the bundled `Thirdparty/Sophus` —
both already satisfied by JetPack 4.6.1's stock versions (Eigen 3.3.4, CMake
3.10), no action needed.

#### Recommended: increase swap before building

ORB-SLAM3 is a substantially larger codebase than ORB-SLAM2 (multi-map, IMU,
extra camera models) and the smaller ORB-SLAM2 build already carries an OOM
warning on this board. Give yourself more headroom first:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h   # confirm the new swap shows up
```

#### Clone and build

```bash
cd ~/code
git clone --branch jetson-b01-maxwell git@github.com:ahestevenz/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
```

`build.sh` builds `Thirdparty/{DBoW2,g2o,Sophus}` with an unqualified `make -j`
(all cores) before the main library — the same OOM risk flagged for ORB-SLAM2's
`build.sh`. Cap it at `-j2` first:

```bash
sed -i 's/make -j$/make -j2/; s/make -j4$/make -j2/' build.sh
grep -n "make -j" build.sh   # confirm every occurrence is now -j2

deactivate   # build outside the virtualenv, same reasoning as Step 9's ORB-SLAM2 build
./build.sh   # expect 45-90+ min on the Nano; larger codebase than ORB-SLAM2
```

#### Verify the build

```bash
ls -la lib/libORB_SLAM3.so
```

Confirm the nano-explorer debug instrumentation actually compiled in:

```bash
nm -D lib/libORB_SLAM3.so | c++filt | grep -i "GetLastInit"
```

Expect three lines, one per new getter:
```
... ORB_SLAM3::System::GetLastInitDetections()
... ORB_SLAM3::System::GetLastInitRawMatches()
... ORB_SLAM3::System::GetLastInitInlierMatches()
```
If this prints nothing, the build picked up a stale object file —
`rm -rf build` and rebuild.

The ORB vocabulary is already bundled at `Vocabulary/ORBvoc.txt.tar.gz` and
`build.sh` un-tars it automatically — no separate download needed here (unlike
ORB-SLAM2's manual `wget` above).

### Python bindings — build pyorbslam

Builds `ahestevenz/pyorbslam` (branch `jetson-b01-maxwell`) — a fork of
`edavalosanaya/pyorbslam` with the same three `GetLastInit*` getters from the
C++ core above wired through to Python (`System.get_last_init_detections()`,
`get_last_init_raw_matches()`, `get_last_init_inlier_matches()`), plus the same
OpenCV 4.1.1 fix applied to its own vendored copy of ORB-SLAM3 (this repo
bundles its own `src/ORB_SLAM3` — a separate copy from the C++ core build
above, with its own `shared_ptr`-based memory-leak fixes — not a submodule
pointing at that repo).

> **Warning — significantly less verified than the C++ core build above, read
> before starting.**
> `pyorbslam`'s own README only tests **Ubuntu 22.04** and documents just two
> system dependencies (`libopencv-dev libeigen3-dev`) — reading the actual
> `CMakeLists.txt`/`pyproject.toml` files turns up several more, none of them
> confirmed against a real JetPack 4.6.1 device. Treat every step below as a
> documented starting point, not a guarantee — budget real troubleshooting time.

#### System dependencies

Unlike the C++ core build above, everything in this part — the `pip3 install
--upgrade cmake` below, and the `pip3 install .` build itself — needs to run
**inside** the `nano_explorer_py36` virtualenv used throughout this doc (a
Python-packaging operation needs that venv's own `pip`/`python3.6`, not the
system one):
```bash
workon nano_explorer_py36   # or: source ~/.virtualenvs/nano_explorer_py36/bin/activate
```

Beyond the `libboost-dev libboost-thread-dev libboost-filesystem-dev
libboost-python-dev libopencv-dev` already installed above (OpenCV/Eigen3 are
already satisfied):

```bash
sudo apt install -y libboost-numpy-dev
```

(No PyQt5 apt package needed — see the fourth point below; this fork no
longer requires it at all for what nano-explorer uses.)

**Four things confirmed by actually running this on a real JetPack 4.6.1
device:**

- **`py-build-cmake~=0.1.8` cannot install on Python 3.6 — already fixed in
  this fork.** Confirmed against PyPI's full release history: every
  `py-build-cmake` release from `0.1.0` onward (including upstream's `~=0.1.8`
  pin) requires Python ≥3.7, and `0.0.11.post1` is the *only* release that ever
  supported 3.6. Running `pip3 install .` against upstream's `pyproject.toml`
  fails immediately in the "Installing build dependencies" step with `ERROR:
  Could not find a version that satisfies the requirement py-build-cmake~=0.1.8
  (from versions: 0.0.11.post1)`. This fork's `pyproject.toml` is already
  pinned to `py-build-cmake==0.0.11.post1` instead, with `find_python = true`
  removed (that option doesn't exist in `0.0.11.post1`, and its config parser
  rejects unrecognized keys outright) — every other key both versions'
  schemas were checked against matched exactly. No action needed here, but
  if you're seeing the error above, confirm you're on this fork's pinned
  branch and not a plain clone of upstream.

- **CMake ≥ 3.20 must be installed from inside the venv specifically.**
  `pip3 install --upgrade cmake` run *outside* any virtualenv (plain system
  `pip3`) fails with `ModuleNotFoundError: No module named 'skbuild'` — no
  prebuilt wheel resolves there, so pip falls back to a from-source build of
  `cmake` itself, which needs `scikit-build` and fails. Run `workon
  nano_explorer_py36` (or activate the venv) **first**, then
  `pip3 install --upgrade cmake` — inside the venv, a prebuilt wheel resolves
  fine (confirmed: `cmake` 3.28.4 installed cleanly, `Requirement already
  satisfied` on a second run). This is the same venv-first ordering the box
  above already covers, just spelling out the exact failure mode if skipped.

- **Boost.NumPy needs an explicit, separate apt package.** `src/python/CMakeLists.txt`
  requires `find_package(Boost REQUIRED COMPONENTS python numpy3)` — confirmed
  on-device that this fails with `Could NOT find Boost (missing: numpy3)
  (found version "1.65.1")` if only Step 9's `libboost-python-dev` is
  installed. The fix is **not** a from-source Boost rebuild — `libboost-numpy-dev`
  (and the exact-version `libboost-numpy1.65-dev`/`libboost-numpy1.65.1`) are
  real, already-available Ubuntu 18.04/bionic packages (confirmed via
  `apt-cache search boost-numpy` on-device) that simply weren't installed by
  Step 9, which only ever needed plain Boost.Python. Already folded into the
  apt command above.

- **PyQt5 used to be a hard dependency for no reason — already fixed in this
  fork.** Upstream's `pyproject.toml` declared `PyQt5`/`pyqtgraph`/`trimesh`/
  `pyzmq`/... as hard `[project]` dependencies, even though they're only ever
  imported by `trajectory_drawer/` (a visualization feature nano-explorer
  never uses — its own integration is just `from pyorbslam import orbslam3`).
  Confirmed on-device: this broke a plain `pip3 install .` **entirely** —
  building PyQt5 needs `sip`, and every `sip` 5.x/6.x release fails with
  `AttributeError: module 'setuptools.build_meta' has no attribute
  '__legacy__'` against this venv's pip/setuptools combination, while every
  PyQt5 release old enough to predate that (`<=5.14.2`) ships a
  `pyproject.toml` with an invalid PEP 518 requirement string (`'sip >=5.0.1
  <6'`, missing a comma) that this pip's parser rejects outright — every
  PyQt5 version from 5.14.0 to 5.15.1 failed one way or the other. This
  fork's `pyproject.toml` moves `PyQt5` and everything else exclusive to
  `trajectory_drawer/`/`tools.py` into `viewer`/`tools` optional extras, and
  `__init__.py` now imports both best-effort (logs a warning instead of
  crashing) — so a plain `pip3 install .` no longer touches PyQt5 at all. No
  action needed; noted here so the old apt-based workaround (a `python3-pyqt5`
  system package) isn't confused for something still required.

#### Clone

```bash
cd ~/code
git clone --branch jetson-b01-maxwell git@github.com:ahestevenz/pyorbslam.git
cd pyorbslam
```

No `--recursive` needed — `src/ORB_SLAM3`/`DBoW2`/`g2o`/`Sophus` are plain
vendored directories in this repo, not git submodules.

#### A note on the hardcoded Debug build (don't try to fix this one)

`src/CMakeLists.txt` unconditionally sets `set(CMAKE_DEBUG TRUE)` near the
top, and both branches of the `if(NOT CMAKE_DEBUG)` below it set
`CMAKE_BUILD_TYPE Debug` — the "release" branch has `set(CMAKE_BUILD_TYPE
Release)` sitting there **commented out with `# causes errors`**. That's the
original author telling us Release mode is already known to break something,
undocumented what. `pyproject.toml`'s own `build_type = "Release"` gets
silently overridden back to Debug by this — real (confirmed on-device: actual
compiler invocations show `-O0 -g`, not `-O2`/`-O3`), but flipping
`CMAKE_DEBUG` to force it is not a safe fix given that comment. Left alone.

#### Build

Still inside `nano_explorer_py36`:

```bash
pip3 install .
```

**Confirmed on-device: `-j3` (this fork's original default) causes severe
swap thrashing on a 4GB Nano.** `ps aux` during a real build showed parallel
`cc1plus` jobs each holding ~1–1.15GB RSS, sitting in `D` (uninterruptible
sleep) state, with over 60 minutes of wall-clock elapsed for under 6 minutes
of actual CPU time on a single Sophus test compile unit — the process was
blocked on swap I/O essentially the entire time, not computing. This fork's
`pyproject.toml` is already changed to `build_args = ["-j1"]`. Expect this to
run notably longer than the plain C++ core build above as a result (single job,
plus this repo compiles its own separate copy of DBoW2/g2o/Sophus/ORB_SLAM3
from scratch, plus the Python bindings, plus its own C++ GoogleTest suite) —
budget several hours, not tens of minutes, and treat "no visible pip output
for a long time" as normal rather than stuck (see the note above `pip
install` about `prepare_metadata_for_build_wheel` actually building the full
wheel).

This pulls in every dependency in `pyproject.toml`'s `[project.dependencies]`
— now pared down to just `numpy`, `tqdm`, `PyYAML`, `Pillow` (see the fixed
item below for why `opencv-python-headless`/`PyQt5`/`pyqtgraph`/etc. aren't
in this list at all anymore).

**Confirmed on-device: `opencv-python-headless` cannot install from PyPI on
this device — already removed from this fork entirely (not moved to an
extra, just deleted).** No PyPI `opencv-python-headless` release publishes an
aarch64+cp36 wheel, so `pip` falls back to building from source — and every
source release fails one of two ways, confirmed across 5+ versions from
`5.0.0.93` down to `4.10.0.82`: newer, scikit-build-backed releases fail with
`ModuleNotFoundError: No module named 'wheel.wheelfile'` (a `wheel`-package
version mismatch inside the isolated build env); older, setup.py-backed
releases fail with the same `setuptools.build_meta.__legacy__`
`AttributeError` seen with `sip` below. This matches a policy nano-explorer's
own `requirements.txt` already states outright — *"OpenCV 4.1.1 is bundled
with JetPack — do not reinstall from pip"* — since the system OpenCV confirmed
in the C++ core part above is already importable as `cv2` in this exact venv
(same as every other vision command in this project), and `pyorbslam`'s own code just does
`import cv2` with no real version floor that needs the PyPI package.

**Confirmed on-device: the repo's own C++ GoogleTest suite (`src/test/`) fails
to build on GCC 7.5.0 — already disabled in this fork.** Five test `.cpp`
files (`test_mono`/`test_map`/`test_map_complex`/`test_key_frame`/`test_atlas`)
`#include<filesystem>`, a C++17 header that doesn't exist at all in GCC
7.5.0's libstdc++ (JetPack 4.6.1's stock compiler — only
`<experimental/filesystem>` exists there, under different flags/linking, with
known gaps). The real failure mode is worth knowing even though it's fixed:
building `test_mono.cpp` errors with `fatal error: filesystem: No such file or
directory` and fails the **entire** `pip install .`, even though everything
nano-explorer actually needs — `libORB_SLAM3.so` and the `orbslam3` Python
extension module itself — had already finished building successfully by that
point (88–90% through the build, per the percentage markers in `pip`'s
output). This fork's `src/CMakeLists.txt` comments out `add_subdirectory(test)`
and its `FetchContent` googletest fetch entirely — confirmed via `install()`
that neither is referenced by anything actually packaged.

#### Verify

```bash
python3 -c "import pyorbslam; from pyorbslam import orbslam3; print('pyorbslam OK')"
python3 -c "
from pyorbslam import orbslam3
print([m for m in dir(orbslam3.System) if 'last_init' in m])
"
```

Expect the second command to print all three new methods:
`['get_last_init_detections', 'get_last_init_inlier_matches', 'get_last_init_raw_matches']`.

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
