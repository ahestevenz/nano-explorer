"""
CameraMotionMixIn — reusable base class for any component that:
  - Opens a single CSI/USB camera
  - Optionally serves an annotated MJPEG stream
  - Runs a dedicated capture thread to decouple frame capture
    from the inference/control loop

Usage:
    class MyProcessor(CameraMotionMixIn):
        def run(self) -> None:
            cam = self._open_camera()
            stop = threading.Event()
            if self._config.stream:
                self._start_stream()
                self._start_capture_thread(cam, stop)
            try:
                while True:
                    frame = cam.read()
                    result = self._process(frame)
                    if self._config.stream and self._server is not None:
                        self._server.frame_buffer.put(
                            self._annotate_frame(frame, result)
                        )
            except KeyboardInterrupt:
                pass
            finally:
                stop.set()
                self._close_camera(cam)
"""

import threading
from typing import Any

from loguru import logger

from lib.camera import Camera, MjpegServer
from lib.network import get_wifi_ip


class CameraMotionMixIn:
    """
    Mixin providing camera open/close, MJPEG streaming, and optional
    arrow-key teleoperation helpers.

    Expects the subclass to set self._config before calling any method here.
    self._config must have:
        stream:      bool
        stream_port: int
    """

    def __init__(self) -> None:
        self._server = None
        ip = get_wifi_ip()
        self._nano_ip = ip if ip is not None else "<nano-ip>"

    def _open_camera(self) -> Camera:
        """Open and return the shared camera instance."""
        cam = Camera()
        cam.open()
        return cam

    def _close_camera(self, cam: Camera) -> None:
        """
        Stop the MJPEG server first, then release the camera.

        Order matters — the capture thread holds a reference to cam,
        so the server (and its capture thread) must be stopped before
        cam.release() is called to avoid a GStreamer crash.
        """
        if self._server is not None:
            self._server.stop()
            self._server = None
        cam.release()

    def _start_server_stream(self, stream_port: int) -> None:
        """Start the MJPEG server pointed at the shared frame buffer."""
        self._server = MjpegServer(port=stream_port)
        self._server.start()
        logger.info(f"Camera stream -> http://{self._nano_ip}:{stream_port}/stream")

    def _start_capture_thread(self, cam: Camera, stop_event: threading.Event) -> threading.Thread:
        """
        Push frames from cam into the MJPEG frame buffer in a dedicated thread.

        Decouples the capture rate from the inference / control loop so the
        stream stays live even when the model takes longer than one frame to run.

        Args:
            cam:        Open Camera instance to read from.
            stop_event: Set this event to stop the thread cleanly.

        Returns:
            The started daemon thread.
        """

        def _loop():
            while not stop_event.is_set():
                try:
                    frame = cam.read()
                    self._server.frame_buffer.put(frame)
                except Exception:  # pylint: disable=broad-except
                    break

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t

    def _start_stream(self, cam: Camera, stop_event: threading.Event, stream_port: int) -> None:
        """
        Start the stream and capture thread in one call if stream=True.
        """
        self._start_server_stream(stream_port=stream_port)
        self._start_capture_thread(cam, stop_event)

    def _push_frame(self, frame: Any) -> None:
        """
        Push an annotated frame to the MJPEG buffer if streaming is active.
        """
        self._server.frame_buffer.put(frame)

    def _start_teleop_thread(
        self, stop_event: threading.Event, speed: float = 0.3, turn_gain: float = 0.5
    ) -> threading.Thread:
        """
        Start arrow-key teleoperation in a background daemon thread.

        Instantiates a TeleopController and runs its arrow loop in a thread,
        sharing stop_event with the caller so either side can signal a clean exit.

        Args:
            speed:      Linear speed [0.0, 1.0].
            turn_gain:  Differential turn gain [0.0, 1.0].
            stop_event: threading.Event — set on quit; poll in the caller's loop.

        Returns:
            The started daemon thread.
        """
        from lib.motion.teleoperation import TeleopController

        controller = TeleopController(speed=speed, turn_gain=turn_gain)
        t = threading.Thread(
            target=controller._run_arrows,
            kwargs={"stop_event": stop_event},
            daemon=True,
            name="arrow-teleop",
        )
        t.start()
        logger.info("[teleop] Arrow keys active — q to quit")
        return t
