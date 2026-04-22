from pathlib import Path

from pydantic import BaseSettings, Field  # pylint: disable = no-name-in-module

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent


class NanoSettings(BaseSettings):  # pylint: disable = no-name-in-module
    default_speed: float = Field(0.3, ge=0.0, le=1.0)
    default_turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    stream_port: int = Field(8080, gt=1024, lt=65535)
    collision_model_path: Path = PROJECT_ROOT_PATH / "assets/models/collision_avoidance.pth"
    collision_threshold: float = Field(0.5, ge=0.0, le=1.0)
    camera_source: str = "csi"
    camera_device_id: int = Field(0, ge=0)

    class Config:
        env_prefix = "NANO_"
