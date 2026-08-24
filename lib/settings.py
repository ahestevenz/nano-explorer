import shutil
from pathlib import Path

from loguru import logger
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

# All YAML config (config/*.yaml — model backends, thresholds, motor trim) lives
# under here at runtime, never inside the installed package's own config/ tree:
# a `pip install --upgrade` only ever touches the package, so anything editable
# living there (tuned thresholds, a hard-won motor trim) would be silently wiped
# on upgrade. assets/models/*.pth weight files are NOT covered by this — they're
# large versioned binaries, not per-user tunables, and stay on PROJECT_ROOT_PATH
# / DEPLOY_ROOT as before.
USER_CONFIG_DIR = Path.home() / ".nano-explorer" / "config"


def user_config_path(relative: str) -> Path:
    """
    Path under USER_CONFIG_DIR for a config file (e.g. "models/slam.yaml",
    "motors/trim.yaml"), mirroring config/'s own layout. Pure path arithmetic,
    no I/O — safe to call for every NanoSettings field on every CLI invocation.
    Call ensure_user_config() from the owning module's own validator to seed it
    lazily, only when that specific config is actually needed.
    """
    return USER_CONFIG_DIR / relative


def ensure_user_config(path: Path) -> Path:
    """
    Seed `path` (expected under USER_CONFIG_DIR) from the matching bundled
    default in PROJECT_ROOT_PATH/config/ the first time it's needed, if it
    doesn't exist yet. Returns `path` unchanged either way — drop straight
    into a pydantic validator: `v = ensure_user_config(v)`.

    Only a missing file triggers a copy; once seeded, a file here is never
    touched again by this function, so user edits (a tuned threshold, a
    calibrated trim) persist across `pip install --upgrade`.
    """
    if path.exists():
        return path
    try:
        relative = path.relative_to(USER_CONFIG_DIR)
    except ValueError:
        return path  # not under USER_CONFIG_DIR — nothing we know how to seed
    bundled = PROJECT_ROOT_PATH / "config" / relative
    if not bundled.exists():
        return path  # no bundled default either; let the caller's own "not found" fire
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled, path)
    logger.info(f"Seeded {path} from bundled default {bundled}")
    return path


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
    # Per-wheel power trim lives in its own YAML file, not here — trim values
    # used to be NANO_MOTOR_LEFT_TRIM/RIGHT_TRIM env-settings fields, but a
    # real (exported) shell env var silently overrides a same-named .env file
    # entry in pydantic's BaseSettings (env vars are merged in over the .env
    # file's values, not the reverse), which made a stale export impossible
    # to distinguish from a freshly-calibrated value. A plain YAML file read
    # directly by lib/motor.py has no such hidden precedence. See
    # tools/calibrate_motors.py and config/motors/trim.yaml.
    motor_trim_config_path: Path = user_config_path("motors/trim.yaml")
    camera_source: str = "csi"
    camera_device_id: int = Field(0, ge=0)

    # Vision — config file paths (can be overridden via NANO_* env vars).
    # Live under ~/.nano-explorer/config/ (see USER_CONFIG_DIR above), seeded
    # from the bundled config/ defaults the first time each is actually used.
    detection_config_path: Path = user_config_path("models/detection.yaml")
    face_config_path: Path = user_config_path("models/face.yaml")
    segmentation_config_path: Path = user_config_path("models/segmentation.yaml")
    pose_config_path: Path = user_config_path("models/pose.yaml")

    # Navigation
    line_follow_config_path: Path = user_config_path("models/line_follow.yaml")
    road_follow_model_path: Path = PROJECT_ROOT_PATH / "assets/models/road_follower.pth"
    apriltag_config_path: Path = user_config_path("models/apriltag.yaml")

    # Mapping
    odometry_config_path: Path = user_config_path("models/odometry.yaml")
    slam_config_path: Path = user_config_path("models/slam.yaml")

    class Config:
        env_prefix = "NANO_"
        # Optional config file: NANO_COLLISION_MODEL_PATH=/custom/path.pth etc.
        # Requires `pip install python-dotenv`; ignored entirely if the file
        # doesn't exist, so it's a no-op until someone creates it.
        env_file = Path.home() / ".nano-explorer.env"
        env_file_encoding = "utf-8"
