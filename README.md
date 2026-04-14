# 🤖 nano-explorer
[![CI Tests](https://github.com/ahestevenz/nano-explorer/actions/workflows/tests.yml/badge.svg)](https://github.com/ahestevenz/nano-explorer/actions/workflows/tests.yml)
[![Lint](https://github.com/ahestevenz/nano-explorer/actions/workflows/lint.yml/badge.svg)](https://github.com/ahestevenz/nano-explorer/actions/workflows/lint.yml)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![JetPack](https://img.shields.io/badge/JetPack-4.6.1-green.svg)](https://developer.nvidia.com/embedded/jetpack)
[![Platform](https://img.shields.io/badge/platform-Jetson%20Nano-76b900.svg)](https://developer.nvidia.com/embedded/jetson-nano)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular command-line toolkit for the **Waveshare JetBot** powered by the **NVIDIA Jetson Nano 4GB (Maxwell, B01)**.

Covers basic motion, computer vision, navigation and ML experiments — all driven from a single CLI entry point.

---

## Platform

| Component   | Version                          |
|-------------|----------------------------------|
| JetPack     | 4.6.1 (L4T 32.7.1)              |
| OS          | Ubuntu 18.04 aarch64             |
| Python      | 3.6                              |
| CUDA        | 10.2                             |
| cuDNN       | 8.2                              |
| TensorRT    | 8.2                              |
| OpenCV      | 4.1.1 (JetPack bundled)          |
| PyTorch     | 1.10.0 (Jetson wheel)            |
| TorchVision | 0.11.0                           |

> ⚠️ PyTorch ≥ 1.11 requires Python ≥ 3.7 and is **not** compatible with JetPack 4.6.x on the Nano.


## Installation

See [doc/jetbot-setup.md](doc/jetbot-setup.md) for full Jetson Nano setup instructions.

```bash
git clone https://github.com/ahestevenz/nano-explorer.git
cd nano-explorer
pip3 install -e .
```

---

## Usage

```
nano-explorer <group> <command> [options]
```

### Motion Operations

```bash
# Teleoperation via keyboard (arrow keys)
nano-explorer motion teleop --speed SPEED
nano-explorer motion teleop --turn-gain GAIN
nano-explorer motion teleop --mode {auto,arrows,pynput,stdin}
nano-explorer motion teleop --stream
nano-explorer motion teleop --stream-port PORT

# Camera streaming
nano-explorer motion stream --mode {mjpeg,opencv}
nano-explorer motion stream --port PORT
nano-explorer motion stream --width W
nano-explorer motion stream --height H

# Collision avoider
nano-explorer motion collision --model PATH
nano-explorer motion collision --threshold T  # Blocked probability threshold
nano-explorer motion collision --speed SPEED
nano-explorer motion collision --stream
nano-explorer motion collision --stream-port PORT
```

### Computer Vision & Detection

### Navigation & Mapping (Experimental)

### Machine Learning (Experimental)

---

## Configuration

All model paths, camera parameters and detection thresholds live in `config/`.
Edit the relevant YAML before running a command — no source changes needed.

See [`config/README.md`](config/README.md) for field descriptions.

---

## Running Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Single group
python3 -m pytest tests/vision/ -v
```

---

## Development

These tools run on your **laptop/desktop** — not on the Nano itself.

### Setup

```bash
pip install pre-commit black ruff pylint
pre-commit install   # registers the git hook
```

### Running the checks manually

```bash
# Run all hooks against every file
pre-commit run --all-files

# Or run individual tools
ruff check lib/ commands/        # linter
ruff format lib/ commands/       # formatter check
black --check lib/ commands/     # format check
pylint lib/ commands/            # static analysis
```

### Tools

| Tool | Role |
|---|---|
| [ruff](https://docs.astral.sh/ruff/) | Linter + import sorter (replaces flake8 + isort) |
| [black](https://black.readthedocs.io/) | Opinionated code formatter |
| [pylint](https://pylint.readthedocs.io/) | Deep static analysis |
| [pre-commit](https://pre-commit.com/) | Git hook manager — runs all of the above on commit |

The same checks run automatically in CI on every push and pull request via GitHub Actions.

---

## Contributing

1. Fork → branch → PR against `main`
2. All new code must pass `flake8` and include a matching test
3. CI runs automatically on every push and pull request

---

## License

MIT — see [LICENSE](LICENSE).
