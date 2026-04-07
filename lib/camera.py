"""
Unified camera abstraction for Jetson Nano.

Supports:
  - CSI camera via GStreamer + nvarguscamerasrc (best performance)
  - USB camera via V4L2 / OpenCV VideoCapture

Usage:
    cam = Camera(width=640, height=480, fps=30, source="csi")
    cam.open()
    frame = cam.read()   # numpy BGR array
    cam.release()

    # Or as a context manager:
    with Camera() as cam:
        frame = cam.read()
"""

from loguru import logger

# GStreamer pipeline for CSI camera (uses NVIDIA hardware decoder)
_GST_CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw,width={w},height={h},format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1"
)


class Camera:
    """
    Thin wrapper around OpenCV VideoCapture, with a GStreamer pipeline
    for the CSI camera and plain V4L2 for USB cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        source: str = "csi",
        device_id: int = 0,
    ):
        """
        Args:
            width:     Capture width in pixels.
            height:    Capture height in pixels.
            fps:       Target frames per second.
            source:    "csi" for the CSI ribbon camera, "usb" for a USB webcam.
            device_id: V4L2 device index for USB cameras (ignored for CSI).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.device_id = device_id
        self._cap = None

    def open(self) -> None:
        """Open the camera. Raises RuntimeError if it cannot be opened."""
        import cv2

        if self.source == "csi":
            pipeline = _GST_CSI_PIPELINE.format(
                w=self.width, h=self.height, fps=self.fps
            )
            logger.info("Opening CSI camera via GStreamer pipeline")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info(f"Opening USB camera at /dev/video{self.device_id}")
            self._cap = cv2.VideoCapture(self.device_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open {self.source} camera. "
                "Check connections and run: v4l2-ctl --list-devices"
            )
        logger.success(f"Camera ready: {self.width}x{self.height} @ {self.fps} fps")

    def read(self):
        """
        Read one frame.

        Returns:
            numpy.ndarray: BGR frame, shape (H, W, 3).

        Raises:
            RuntimeError: if the camera is not open or the read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
