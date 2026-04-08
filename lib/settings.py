# lib/settings.py
from pydantic import BaseSettings, Field


class NanoSettings(BaseSettings):
    default_speed: float = Field(0.3, ge=0.0, le=1.0)
    default_turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    stream_port: int = Field(8080, gt=1024, lt=65535)
    collision_model_path: str = "assets/models/collision_avoidance.pth"
    collision_threshold: float = Field(0.5, ge=0.0, le=1.0)
    camera_source: str = "csi"
    camera_device_id: int = Field(0, ge=0)

    class Config:
        env_prefix = "NANO_"