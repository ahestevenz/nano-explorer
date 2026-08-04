from pathlib import Path

from pydantic import BaseSettings, Field  # pylint: disable = no-name-in-module

# NOTE: only correct when nano-explorer is installed editable (`pip install -e .`).
# A regular `pip install .` copies lib/ into site-packages, so __file__ resolves
# inside the venv instead of the git checkout, and PROJECT_ROOT_PATH silently
# points at the wrong place. DEPLOY_ROOT below is used instead for defaults that
# must survive a non-editable install (e.g. the collision model path).
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

# Fixed, install-mode-independent location of the actual git checkout on the
# robot. Override any setting below without touching this file by dropping a
# NANO_<SETTING>=value line in ~/.nano-explorer.env (see Config.env_file).
DEPLOY_ROOT = Path.home() / "code" / "nano-explorer"
DEFAULT_COLLISION_MODEL_PATH = DEPLOY_ROOT / "assets/models/collision_avoidance.pth"

# Fallback source when collision_model_path doesn't exist locally — see
# lib/motion/collision.py::_download_from_hub.
DEFAULT_COLLISION_MODEL_HF_REPO_ID = "ahestevenz/collision_avoidance"
DEFAULT_COLLISION_MODEL_HF_FILENAME = "collision_avoidance.pth"
DEFAULT_COLLISION_MODEL_HF_REVISION = "v1.1"


class NanoSettings(BaseSettings):  # pylint: disable = no-name-in-module
    # Motion
    default_speed: float = Field(0.3, ge=0.0, le=1.0)
    default_turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    stream_port: int = Field(8080, gt=1024, lt=65535)
    collision_model_path: Path = DEFAULT_COLLISION_MODEL_PATH
    collision_model_hf_repo_id: str = DEFAULT_COLLISION_MODEL_HF_REPO_ID
    collision_model_hf_filename: str = DEFAULT_COLLISION_MODEL_HF_FILENAME
    collision_model_hf_revision: str = DEFAULT_COLLISION_MODEL_HF_REVISION
    collision_threshold: float = Field(0.6, ge=0.0, le=1.0)
    # Per-wheel power trim — corrects forward-drift from motor manufacturing
    # variance. Multiplicative, applied via jetbot.Robot's left/right_motor_alpha,
    # so it can only ever scale a commanded speed down, never above what was
    # requested. 1.0 = no correction. See tools/calibrate_motors.py.
    motor_left_trim: float = Field(1.0, gt=0.0, le=1.0)
    motor_right_trim: float = Field(1.0, gt=0.0, le=1.0)
    camera_source: str = "csi"
    camera_device_id: int = Field(0, ge=0)

    # Vision — config file paths (can be overridden via NANO_* env vars)
    detection_config_path: Path = PROJECT_ROOT_PATH / "config/models/detection.yaml"
    face_config_path: Path = PROJECT_ROOT_PATH / "config/models/face.yaml"
    segmentation_config_path: Path = PROJECT_ROOT_PATH / "config/models/segmentation.yaml"
    pose_config_path: Path = PROJECT_ROOT_PATH / "config/models/pose.yaml"

    # Navigation
    line_follow_config_path: Path = PROJECT_ROOT_PATH / "config/models/line_follow.yaml"
    road_follow_model_path: Path = PROJECT_ROOT_PATH / "assets/models/road_follower.pth"
    apriltag_config_path: Path = PROJECT_ROOT_PATH / "config/models/apriltag.yaml"

    # Mapping
    odometry_config_path: Path = PROJECT_ROOT_PATH / "config/models/odometry.yaml"
    slam_config_path: Path = PROJECT_ROOT_PATH / "config/models/slam.yaml"

    class Config:
        env_prefix = "NANO_"
        # Optional config file: NANO_COLLISION_MODEL_PATH=/custom/path.pth etc.
        # Requires `pip install python-dotenv`; ignored entirely if the file
        # doesn't exist, so it's a no-op until someone creates it.
        env_file = Path.home() / ".nano-explorer.env"
        env_file_encoding = "utf-8"
