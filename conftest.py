import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

# Import cv2 before the mock loop so the real library is used when available.
# sys.modules.setdefault below is a no-op for any module already imported here.
with contextlib.suppress(ImportError):
    import cv2  # noqa: F401

# Mock hardware-only modules so they never need to be installed on CI
for mod in [
    "cv2",
    "torch",
    "torchvision",
    "torchvision.transforms",
    "torchvision.models",
    "jetbot",
    "torch2trt",
    "apriltag",
    "orbslam2",
    "rtabmap",
    "rtabmap.util",
]:
    sys.modules.setdefault(mod, MagicMock())
