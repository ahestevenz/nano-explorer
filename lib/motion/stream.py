"""
Camera streaming over Wi-Fi in two modes:

  mjpeg  — aiohttp server pushing MJPEG frames; open in any browser at
            http://<nano-ip>:<port>/stream
  opencv — local OpenCV window (useful when connected via HDMI/VNC)

JetPack 4.6.1 compatible:
    aiohttp   3.7.4  (last version supporting Python 3.6)
    OpenCV    4.1.1  (bundled with JetPack)
"""

import asyncio
import threading

from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, FrameBuffer
from lib.network import get_wifi_ip



class CameraStreamerConfig(BaseModel):
    """
    Args:
        mode:   "mjpeg" | "opencv"
        port:   Server port (mjpeg  only)
        width:  Capture / stream width
        height: Capture / stream height
        fps:    Target frame rate
    """

    mode: str = "mjpeg"
    port: int = Field(8080, gt=1024, lt=65535)
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)

    @validator("mode")
    def mode_must_be_valid(cls, v):
        if v not in ("mjpeg", "opencv"):
            raise ValueError("mode must be 'mjpeg' or 'opencv'")
        return v


class CameraStreamer:
    """
    Multi-mode camera streamer.
    """

    def __init__(self, **kwargs):
        self._config = CameraStreamerConfig(**kwargs)
        self._buf = FrameBuffer()
        self._cam = Camera(
            width=self._config.width, height=self._config.height, fps=self._config.fps
        )

    def run(self) -> None:
        logger.info(f"Starting camera stream — mode={self._config.mode}")
        dispatch = {
            "mjpeg": self._run_mjpeg,
            "opencv": self._run_opencv,
        }
        fn = dispatch.get(self._config.mode)
        if fn is None:
            raise ValueError(f"Unknown stream mode: {self._config.mode}")
        fn()

    def _capture_loop(self, stop_event: threading.Event) -> None:
        with self._cam:
            while not stop_event.is_set():
                frame = self._cam.read()
                self._buf.put(frame)

    def _run_mjpeg(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            raise RuntimeError(
                "aiohttp is required for MJPEG streaming. "
                "Run: pip3 install 'aiohttp==3.7.4'"
            )

        stop_event = threading.Event()
        cap_thread = threading.Thread(
            target=self._capture_loop, args=(stop_event,), daemon=True
        )
        cap_thread.start()

        async def _mjpeg_handler(request):
            response = web.StreamResponse()
            response.content_type = "multipart/x-mixed-replace; boundary=frame"
            await response.prepare(request)

            while True:
                frame = self._buf.get()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                import cv2

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                data = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
                try:
                    await response.write(data)
                except Exception:
                    break
                await asyncio.sleep(1.0 / self._config.fps)

        app = web.Application()
        app.router.add_get("/stream", _mjpeg_handler)
        ip = get_wifi_ip()
        nano_ip = ip if ip is not None else "<nano-ip>"
        logger.success(
            f"MJPEG server running — open http://{nano_ip}:{self._config.port}/stream"
        )
        try:
            web.run_app(app, port=self._config.port)
        finally:
            stop_event.set()

    def _run_opencv(self) -> None:
        import cv2

        logger.info("OpenCV display mode — press 'q' to quit")
        with self._cam:
            while True:
                frame = self._cam.read()
                cv2.imshow("nano-explorer | camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
