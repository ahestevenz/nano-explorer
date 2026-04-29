"""
Human pose estimation using trt_pose (NVIDIA AI IOT).

trt_pose runs a ResNet18 + PAF head optimised with TensorRT on the Nano.
At 224×224 it achieves ~22 FPS on JetPack 4.6.1.

YAML fields (config/models/pose.yaml):
    model_weights: path to .pth weights file
    model_engine:  path to cached .engine (auto-built on first run, ~5 min)
    topology:      "human_pose"  (trt_pose built-in COCO 17-keypoint)
    width:         input width  (default 224)
    height:        input height (default 224)

Dependencies (install from source — see doc/jetbot-setup.md):
    trt_pose   https://github.com/NVIDIA-AI-IOT/trt_pose
    torch2trt  https://github.com/NVIDIA-AI-IOT/torch2trt
"""

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.settings import PROJECT_ROOT_PATH

# Skeleton connections for visualisation (COCO keypoint names)
_SKELETON_EDGES = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("neck",           "nose"),
]


class PoseConfig(BaseModel):
    """
    Pose estimation configuration.

    Args:
        config_path: Path to pose YAML config.
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/pose.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path")
    def config_must_exist(cls, v):  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"Pose config not found: {v}")
        return v


class PoseEstimator:
    """
    Human pose estimator using trt_pose.

    Construct via PoseEstimator(**config.dict()).
    """

    def __init__(self, **kwargs):
        self._config = PoseConfig(**kwargs)
        self._model = None
        self._topology = None
        self._parse_obj = None
        self._transform = None
        self._device = None
        self._width = 224
        self._height = 224
        self._server = None

    def _load(self) -> None:
        import torch
        import torchvision.transforms as T
        import yaml

        try:
            import trt_pose.coco
            import trt_pose.models
            from trt_pose.parse_objects import ParseObjects
        except ImportError as e:
            raise RuntimeError(
                "trt_pose is not installed.\n"
                "Build from source: https://github.com/NVIDIA-AI-IOT/trt_pose"
            ) from e

        with open(self._config.config_path) as f:
            cfg = yaml.safe_load(f)

        self._width  = cfg.get("width", 224)
        self._height = cfg.get("height", 224)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self._height, self._width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        topology_file = Path(trt_pose.coco.__file__).parent / "human_pose.json"
        with open(topology_file) as f:
            human_pose = json.load(f)

        self._topology  = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts       = len(human_pose["keypoints"])
        num_links       = len(human_pose["skeleton"])
        self._parse_obj = ParseObjects(self._topology)
        self._keypoint_names = human_pose["keypoints"]

        base_model = trt_pose.models.resnet18_baseline_att(
            num_parts, 2 * num_links
        ).cuda().eval()

        weights_path = Path(cfg["model_weights"])
        engine_path  = Path(cfg.get(
            "model_engine",
            str(weights_path).replace(".pth", "_trt.engine"),
        ))

        if engine_path.exists():
            from torch2trt import TRTModule  # pylint: disable=import-error
            self._model = TRTModule()
            self._model.load_state_dict(torch.load(str(engine_path)))
            logger.success(f"Loaded TRT engine: {engine_path}")
        else:
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Pose weights not found: {weights_path}\n"
                    "Download from the trt_pose GitHub releases."
                )
            base_model.load_state_dict(torch.load(str(weights_path)))
            logger.info("Building TensorRT engine — this takes ~5 min on first run …")
            from torch2trt import torch2trt  # pylint: disable=import-error
            data = torch.zeros((1, 3, self._height, self._width)).cuda()
            self._model = torch2trt(
                base_model, [data], fp16_mode=True, max_workspace_size=1 << 25
            )
            torch.save(self._model.state_dict(), str(engine_path))
            logger.success(f"TRT engine saved: {engine_path}")


    def infer(self, frame: np.ndarray):
        """Run inference on a BGR frame. Returns (counts, objects, peaks)."""
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).unsqueeze(0).cuda()
        with torch.no_grad():
            cmap, paf = self._model(inp)
        cmap = cmap.cpu()
        paf  = paf.cpu()
        counts, objects, peaks = self._parse_obj(cmap, paf)
        return counts, objects, peaks


    def annotate(self, frame: np.ndarray, counts, objects, peaks) -> np.ndarray:
        """Overlay skeleton keypoints on a copy of frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        for i in range(int(counts[0])):
            kps = {}
            for j in range(self._topology.shape[0] // 2):
                k = int(objects[0, i, j])
                if k >= 0:
                    y = int(float(peaks[0, j, k, 0]) * h)
                    x = int(float(peaks[0, j, k, 1]) * w)
                    kps[j] = (x, y)
                    cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
        return out


    def run(self) -> None:
        self._load()

        if self._config.stream:
            self._server = MjpegServer(port=self._config.stream_port)
            self._server.start()

        with Camera() as cam:
            logger.info("Pose estimation running — Ctrl+C to stop")
            try:
                while True:
                    frame = cam.read()
                    counts, objects, peaks = self.infer(frame)
                    logger.debug(f"Detected {int(counts[0])} person(s)")

                    if self._server is not None:
                        annotated = self.annotate(frame, counts, objects, peaks)
                        self._server.frame_buffer.put(annotated)

            except KeyboardInterrupt:
                pass
            finally:
                if self._server is not None:
                    self._server.stop()
                logger.info("Pose estimator stopped.")
