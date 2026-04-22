import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

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
]:
    sys.modules.setdefault(mod, MagicMock())
