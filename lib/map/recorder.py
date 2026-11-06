"""
Camera + IMU recorder for offline visual-inertial SLAM (`nano-explorer map record`).

Drive with the arrow keys while it records; q (or Ctrl+C) stops. The output is
a EuRoC-format dataset, which ORB-SLAM3's mono_inertial_euroc example reads as
it is and Kalibr's kalibr_bagcreater turns into a rosbag:

    DIR/
      mav0/cam0/data/<t_ns>.png   grayscale frames, named by timestamp
      mav0/cam0/data.csv          #timestamp [ns],filename
      mav0/cam0/times_raw.csv     #frame,timestamp [ns],read [ns],pts [ns]
      mav0/imu0/data.csv          #timestamp [ns],w_x,w_y,w_z [rad/s],a_x,a_y,a_z [m/s^2]
      times.txt                   one frame timestamp [ns] per line (ORB-SLAM3's list)
      meta.json                   settings, git revision, timing statistics, drops

Every timestamp is CLOCK_MONOTONIC in ns, the clock the IMU reader uses. Frames
are stamped with their read time (`map camera-check` on the Nano found the
buffer PTS no better, see ADDING_IMU.md); times_raw.csv keeps the read time and
the PTS of every frame, so the timestamps can be smoothed offline later.

--imu-only records just the IMU, without camera or teleop: the long stationary
recording for the Allan variance.
"""

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, FrameTimes, analyze_frame_times
from lib.camera_motion_mixin import CameraMotionMixIn
from lib.imu import ImuReader, ImuSample
from lib.settings import PROJECT_ROOT_PATH

_CAM_HEADER = "#timestamp [ns],filename\n"
_RAW_HEADER = "#frame,timestamp [ns],read [ns],pts [ns]\n"
_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
)

# Warn at the start if the card has less room than this. 640x480 grayscale PNGs
# at 30 fps are roughly 5-7 MB/s.
_MIN_FREE_BYTES = 2 * 1024**3
_STATUS_EVERY_S = 10.0
_IMU_ONLY_STATUS_EVERY_S = 60.0


class RecorderConfig(BaseModel):
    """
    Args:
        out:             Output directory; must not exist yet, or be empty.
        imu_only:        Record the IMU only (no camera, no teleop).
        seconds:         Stop after this long; None records until q / Ctrl+C.
        width / height / fps / max_exposure_ms: Camera settings; keep them as
                         the SLAM run will use them.
        png_compression: PNG compression level 0-9. Low is fast to write on
                         the Nano; PNG is lossless at every level.
        queue_frames:    Frames that may wait for the writer before new ones are
                         dropped (and counted).
        imu_rate_hz / imu_bus: As for `map imu-check`.
        stream / stream_port: Serve the camera view over MJPEG (costs CPU).
        speed / turn_gain: Teleop motor speed and turn gain, as for `map slam`.
    """

    out: Path
    imu_only: bool = False
    seconds: Optional[float] = Field(None, gt=0)
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)
    max_exposure_ms: Optional[float] = Field(None, gt=0)
    png_compression: int = Field(1, ge=0, le=9)
    queue_frames: int = Field(60, gt=0)
    imu_rate_hz: float = Field(200.0, gt=0, le=400)
    imu_bus: int = Field(0, ge=0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.2, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)

    @validator("out")
    def out_must_be_new(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if v.exists() and (not v.is_dir() or any(v.iterdir())):
            raise ValueError(f"{v} already exists and is not an empty directory")
        return v


class EurocWriter:
    """
    Writes one EuRoC dataset. write_frame() runs on the writer thread and the
    write_imu() calls on the recording loop: they touch different files.
    """

    def __init__(self, root: Path, camera: bool, png_compression: int = 1):
        self.root = root
        self._png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        mav0 = root / "mav0"
        (mav0 / "imu0").mkdir(parents=True, exist_ok=True)
        self._imu = self._open(mav0 / "imu0" / "data.csv", _IMU_HEADER)
        self._cam: Optional[TextIO] = None
        self._raw: Optional[TextIO] = None
        self._img_dir = mav0 / "cam0" / "data"
        if camera:
            self._img_dir.mkdir(parents=True, exist_ok=True)
            self._cam = self._open(mav0 / "cam0" / "data.csv", _CAM_HEADER)
            self._raw = self._open(mav0 / "cam0" / "times_raw.csv", _RAW_HEADER)
        self.frame_stamps: List[int] = []
        self.imu_first_ns: Optional[int] = None
        self.imu_last_ns: Optional[int] = None
        self.imu_samples = 0

    @staticmethod
    def _open(path: Path, header: str) -> TextIO:
        f = path.open("w")
        f.write(header)
        return f

    def write_frame(self, index: int, t_ns: int, frame: np.ndarray, times: FrameTimes) -> None:
        if self._cam is None or self._raw is None:
            raise RuntimeError("EurocWriter was created without a camera")
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        name = f"{t_ns}.png"
        if not cv2.imwrite(str(self._img_dir / name), gray, self._png_params):
            raise RuntimeError(f"Failed to write {self._img_dir / name}")
        self._cam.write(f"{t_ns},{name}\n")
        pts = "" if times.pts_ns is None else str(times.pts_ns)
        self._raw.write(f"{index},{t_ns},{times.read_ns},{pts}\n")
        self.frame_stamps.append(t_ns)

    def write_imu(self, samples: List[ImuSample]) -> None:
        for s in samples:
            g, a = s.gyro, s.acc
            self._imu.write(
                f"{s.t_ns},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f}\n"
            )
        if samples:
            if self.imu_first_ns is None:
                self.imu_first_ns = samples[0].t_ns
            self.imu_last_ns = samples[-1].t_ns
            self.imu_samples += len(samples)

    def close(self, meta: Dict[str, Any]) -> None:
        for f in (self._imu, self._cam, self._raw):
            if f is not None:
                f.close()
        if self._cam is not None:
            (self.root / "times.txt").write_text("".join(f"{t}\n" for t in self.frame_stamps))
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


class FrameWriterThread:
    """
    Encodes and writes frames off the recording loop, so a slow PNG write never
    holds up the next cam.read() (which would delay that frame's read time).
    A full queue drops the new frame and counts it: a missing frame is harmless,
    a late timestamp isn't.
    """

    def __init__(self, writer: EurocWriter, max_frames: int):
        self._writer = writer
        self._queue: Any = queue.Queue(maxsize=max_frames)
        self.dropped = 0
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-writer")
        self._thread.start()

    def submit(self, index: int, t_ns: int, frame: np.ndarray, times: FrameTimes) -> bool:
        """Queue a frame; False if it was dropped."""
        try:
            self._queue.put_nowait((index, t_ns, frame, times))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self.error is not None:
                continue  # keep emptying the queue so close() can finish
            try:
                self._writer.write_frame(*item)
            except Exception as e:  # pylint: disable=broad-except
                self.error = e
                logger.error(f"Frame writer failed: {e}")

    def close(self) -> None:
        """Write every queued frame, then stop the thread."""
        self._queue.put(None)
        self._thread.join()


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT_PATH),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,  # text= is Python 3.7+
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class Recorder(CameraMotionMixIn):
    """Records a EuRoC dataset; see the module docstring."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = RecorderConfig(**kwargs)
        self._stop = threading.Event()
        self._frame_times: List[FrameTimes] = []

    def run(self) -> Dict[str, Any]:
        """Record until q / Ctrl+C / --seconds; returns the meta.json contents."""
        cfg = self._config
        cfg.out.mkdir(parents=True, exist_ok=True)
        self._check_free_space()
        writer = EurocWriter(cfg.out, camera=not cfg.imu_only, png_compression=cfg.png_compression)
        imu = ImuReader(bus=cfg.imu_bus, rate_hz=cfg.imu_rate_hz)
        t_start = time.monotonic()
        frame_writer: Optional[FrameWriterThread] = None
        imu_started = False
        try:
            # IMU first, so its samples cover the first frame.
            imu.start()
            imu_started = True
            if cfg.imu_only:
                self._record_imu_only(imu, writer)
            else:
                frame_writer = FrameWriterThread(writer, cfg.queue_frames)
                self._record_camera(imu, writer, frame_writer)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            if imu_started:
                imu.stop()
                writer.write_imu(imu.drain())
            if frame_writer is not None:
                logger.info("Writing the queued frames...")
                frame_writer.close()
            try:
                meta = self._meta(writer, imu if imu_started else None, frame_writer, t_start)
            except Exception as e:  # pylint: disable=broad-except
                # The data is on disk by now: never lose it over the summary.
                logger.error(f"Could not build meta.json: {e}")
                meta = {"error": f"meta.json incomplete: {e!r}"}
            writer.close(meta)
        if "error" in meta:
            logger.warning("Recording saved without its statistics (see the error above).")
        else:
            self._log_summary(meta)
        if frame_writer is not None and frame_writer.error is not None:
            raise RuntimeError(f"Recording incomplete: {frame_writer.error}")
        return meta

    def _check_free_space(self) -> None:
        free = shutil.disk_usage(str(self._config.out)).free
        if free < _MIN_FREE_BYTES:
            logger.warning(
                f"Only {free / 1024**3:.1f} GB free at {self._config.out}: "
                "about 5-7 MB/s are needed while the camera records"
            )

    def _deadline_passed(self, t_start: float) -> bool:
        s = self._config.seconds
        return s is not None and time.monotonic() - t_start >= s

    def _record_imu_only(self, imu: ImuReader, writer: EurocWriter) -> None:
        cfg = self._config
        self._start_quit_listener(self._stop)
        logger.info(
            f"Recording the IMU only into {cfg.out} — keep the robot still. q or Ctrl+C stops"
            + (f", or after {cfg.seconds:g} s." if cfg.seconds else ".")
        )
        t_start = last_status = time.monotonic()
        while not self._stop.is_set() and not self._deadline_passed(t_start):
            time.sleep(0.5)
            writer.write_imu(imu.drain())
            if time.monotonic() - last_status >= _IMU_ONLY_STATUS_EVERY_S:
                last_status = time.monotonic()
                st = imu.stats()
                logger.info(
                    f"{(last_status - t_start) / 60:.0f} min: {writer.imu_samples} samples, "
                    f"{st.rate_hz:.1f} Hz, late {st.late}, errors {st.errors}"
                )

    def _record_camera(
        self, imu: ImuReader, writer: EurocWriter, frame_writer: FrameWriterThread
    ) -> None:
        cfg = self._config
        cam = self._open_camera(
            width=cfg.width, height=cfg.height, fps=cfg.fps, exposure_max_ms=cfg.max_exposure_ms
        )
        teleop: Optional[threading.Thread] = None
        try:
            if cfg.stream:
                self._start_server_stream(stream_port=cfg.stream_port)
            teleop = self._start_teleop_thread(
                stop_event=self._stop, speed=cfg.speed, turn_gain=cfg.turn_gain
            )
            logger.info(f"Recording into {cfg.out} — arrow keys to drive, q to stop")
            self._camera_loop(cam, imu, writer, frame_writer)
        finally:
            self._stop.set()
            if teleop is not None:
                teleop.join(timeout=2.0)  # its exit stops the motors
            self._close_camera(cam)

    def _camera_loop(
        self, cam: Camera, imu: ImuReader, writer: EurocWriter, frame_writer: FrameWriterThread
    ) -> None:
        t_start = last_status = time.monotonic()
        dropped_logged = 0
        while not self._stop.is_set() and not self._deadline_passed(t_start):
            frame, times = cam.read_timed()
            index = len(self._frame_times)
            self._frame_times.append(times)
            frame_writer.submit(index, times.read_ns, frame, times)
            writer.write_imu(imu.drain())
            if self._server is not None:
                self._push_frame(frame)
            if frame_writer.error is not None:
                raise RuntimeError(f"Frame writer failed: {frame_writer.error}")
            if time.monotonic() - last_status >= _STATUS_EVERY_S:
                last_status = time.monotonic()
                logger.info(
                    f"{last_status - t_start:.0f} s: {len(self._frame_times)} frames, "
                    f"{writer.imu_samples} IMU samples, dropped {frame_writer.dropped}"
                )
                if frame_writer.dropped > dropped_logged:
                    logger.warning(
                        "Frames dropped: the writer can't keep up (slow SD card?). "
                        "Try --png-compression 0 or a lower --fps."
                    )
                    dropped_logged = frame_writer.dropped

    def _meta(
        self,
        writer: EurocWriter,
        imu: Optional[ImuReader],
        frame_writer: Optional[FrameWriterThread],
        t_start: float,
    ) -> Dict[str, Any]:
        cfg = self._config
        meta: Dict[str, Any] = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_revision": _git_revision(),
            "duration_s": round(time.monotonic() - t_start, 3),
            "clock": "CLOCK_MONOTONIC, ns",
            "config": json.loads(cfg.json()),
            "imu": {
                "samples": writer.imu_samples,
                "first_ns": writer.imu_first_ns,
                "last_ns": writer.imu_last_ns,
                "stats": imu.stats()._asdict() if imu is not None else None,
            },
        }
        if not cfg.imu_only:
            frames = writer.frame_stamps
            timing = None
            if len(self._frame_times) >= 3:
                timing = analyze_frame_times(self._frame_times, cfg.fps)._asdict()
                timing["read"] = timing["read"]._asdict()
                if timing["pts"] is not None:
                    timing["pts"] = timing["pts"]._asdict()
            meta["camera"] = {
                "timestamps": "read",
                "frames_read": len(self._frame_times),
                "frames_written": len(frames),
                "frames_dropped": frame_writer.dropped if frame_writer is not None else 0,
                "first_ns": frames[0] if frames else None,
                "last_ns": frames[-1] if frames else None,
                "timing": timing,
            }
            meta["imu_covers_camera"] = not frames or _covers(
                writer.imu_first_ns, writer.imu_last_ns, frames[0], frames[-1]
            )
        return meta

    @staticmethod
    def _log_summary(meta: Dict[str, Any]) -> None:
        imu = meta["imu"]
        logger.info(f"duration:  {meta['duration_s']:.1f} s")
        st = imu["stats"]
        if st is not None and st["samples"]:
            logger.info(
                f"IMU:       {imu['samples']} samples, {st['rate_hz']:.1f} Hz, "
                f"late {st['late']}, errors {st['errors']}"
            )
        cam = meta.get("camera")
        if cam is not None:
            logger.info(
                f"camera:    {cam['frames_written']} frames written, "
                f"{cam['frames_dropped']} dropped"
            )
            t = cam["timing"]
            if t is not None:
                r = t["read"]
                logger.info(
                    f"frame interval: std {r['std_ms']:.2f} ms, p99 {r['p99_ms']:.2f}, "
                    f"max {r['max_ms']:.2f}, late {r['late']}"
                )
            if not meta["imu_covers_camera"]:
                logger.warning("The IMU samples don't cover every frame")
        logger.success("Recording saved.")


def _covers(
    imu_first: Optional[int], imu_last: Optional[int], frame_first: int, frame_last: int
) -> bool:
    """True if the IMU's time range spans the first to the last frame."""
    if imu_first is None or imu_last is None:
        return False
    return imu_first <= frame_first and imu_last >= frame_last
