#!/usr/bin/env python3
"""
Check `nano-explorer map record` datasets on the laptop, before calibration and
offline ORB-SLAM3 runs.

    python3 helpers/check_recording.py ~/recordings/seq01 [~/recordings/seq02 ...]
    python3 helpers/check_recording.py ~/recordings/seq* --write-times

Each recording gets OK / WARN / FAIL lines for:
  - layout:    the EuRoC files exist, every listed frame is on disk, times.txt
               matches cam0/data.csv, meta.json was saved without an error.
  - frames:    timestamps increase; lost frames (PTS gaps), and whether the
               writer dropped them (frame numbers skip: queue full, slow SD
               card), the recording loop stalled (then appsink dropped them),
               or the camera didn't deliver them.
  - read time: how late each frame was read compared to the usual
               (read time - PTS), the error in its timestamp. The camera's first
               second is ignored: its first frames come late (seq01: 20 and 13 ms).
  - IMU:       rate, gaps, increasing timestamps, and that it covers the frames.

--write-times writes DIR/times_trimmed.txt: the frame list without the first
--skip-start seconds, without frames read more than --max-read-error-ms late,
and without frames the camera delivered late (PTS jitter). ORB-SLAM3's mono_inertial_euroc takes it in place of times.txt, so the
recording itself is never changed.

Exit code: 1 if any recording has a FAIL, else 0. Needs only numpy.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_IMU_RATE_HZ = 200.0
_FPS = 30.0


class Check(NamedTuple):
    level: str
    name: str
    message: str


class Recording(NamedTuple):
    """Frame and IMU timestamps of one dataset, in ns (CLOCK_MONOTONIC)."""

    frame_ns: np.ndarray  # cam0/data.csv timestamps
    names: List[str]  # cam0/data.csv filenames
    read_ns: Optional[np.ndarray]  # times_raw.csv, None if the file is missing
    pts_ns: Optional[np.ndarray]  # times_raw.csv, None if the PTS is missing
    imu_ns: np.ndarray  # imu0/data.csv timestamps
    index: Optional[np.ndarray] = None  # times_raw.csv frame numbers (gaps: writer drops)


class Report(NamedTuple):
    path: Path
    checks: List[Check]
    keep: List[int]  # frame indices for times_trimmed.txt

    @property
    def failed(self) -> bool:
        return any(c.level == FAIL for c in self.checks)


def _csv_rows(path: Path) -> List[List[str]]:
    with path.open() as f:
        return [line.rstrip("\n").split(",") for line in f if line.strip() and line[0] != "#"]


def load_recording(root: Path) -> Recording:
    cam = _csv_rows(root / "mav0/cam0/data.csv")
    frame_ns = np.array([int(r[0]) for r in cam], dtype=np.int64)
    names = [r[1] for r in cam]
    read_ns = pts_ns = index = None
    raw_path = root / "mav0/cam0/times_raw.csv"
    if raw_path.exists():
        raw = _csv_rows(raw_path)
        index = np.array([int(r[0]) for r in raw], dtype=np.int64)
        read_ns = np.array([int(r[2]) for r in raw], dtype=np.int64)
        if raw and all(len(r) > 3 and r[3] for r in raw):
            pts_ns = np.array([int(r[3]) for r in raw], dtype=np.int64)
    imu_ns = np.array([int(r[0]) for r in _csv_rows(root / "mav0/imu0/data.csv")], dtype=np.int64)
    return Recording(frame_ns, names, read_ns, pts_ns, imu_ns, index)


def _check_layout(root: Path, rec: Recording) -> List[Check]:
    checks = []
    img_dir = root / "mav0/cam0/data"
    missing = [n for n in rec.names if not (img_dir / n).exists()]
    on_disk = len(list(img_dir.glob("*.png")))
    if missing:
        checks.append(
            Check(FAIL, "layout", f"{len(missing)} listed frames missing, e.g. {missing[0]}")
        )
    elif on_disk != len(rec.names):
        checks.append(Check(WARN, "layout", f"{on_disk} PNGs on disk, {len(rec.names)} listed"))
    else:
        checks.append(Check(OK, "layout", f"{len(rec.names)} frames on disk"))

    times_path = root / "times.txt"
    if not times_path.exists():
        checks.append(Check(FAIL, "layout", "times.txt missing"))
    else:
        times = [int(t) for t in times_path.read_text().split()]
        if times != rec.frame_ns.tolist():
            checks.append(Check(FAIL, "layout", "times.txt doesn't match cam0/data.csv"))

    meta_path = root / "meta.json"
    if not meta_path.exists():
        checks.append(Check(WARN, "layout", "meta.json missing"))
    elif "error" in json.loads(meta_path.read_text()):
        checks.append(Check(WARN, "layout", "meta.json saved with an error (no statistics)"))
    if rec.read_ns is None:
        checks.append(Check(WARN, "layout", "times_raw.csv missing: no read-time checks"))
    return checks


def _check_frames(
    rec: Recording, fps: float, skip_start_s: float, max_error_ms: float
) -> List[Check]:
    if len(rec.frame_ns) < 3:
        return [Check(FAIL, "frames", f"only {len(rec.frame_ns)} frames")]
    dt = np.diff(rec.frame_ns) / 1e6
    if (dt <= 0).any():
        return [
            Check(FAIL, "frames", f"timestamps not increasing at {int((dt <= 0).sum())} frames")
        ]
    duration = (rec.frame_ns[-1] - rec.frame_ns[0]) / 1e9
    checks = [Check(OK, "frames", f"{len(dt) + 1} frames over {duration:.1f} s")]
    if rec.pts_ns is not None:
        pdt = np.diff(rec.pts_ns) / 1e6
        period = 1000.0 / fps
        gaps = np.where(pdt > 1.5 * period)[0]
        in_start = (rec.frame_ns[gaps + 1] - rec.frame_ns[0]) / 1e9 < skip_start_s
        if in_start.any():
            checks[0] = checks[0]._replace(
                message=checks[0].message + f"; {int(in_start.sum())} gap(s) in the first "
                f"{skip_start_s:g} s (skipped)"
            )
        gaps = gaps[~in_start]
        late_frames = pts_late_frames(rec, fps)
        after = (rec.frame_ns[late_frames] - rec.frame_ns[0]) / 1e9 >= skip_start_s
        late_frames = late_frames[after]
        if len(late_frames):
            worst = (pdt[late_frames - 1] - period).max()
            checks.append(
                Check(
                    WARN,
                    "frames",
                    f"{len(late_frames)} frames delivered late by the camera (nothing lost), "
                    f"up to +{worst:.0f} ms: trimmed",
                )
            )
        gaps = gaps[~np.isin(gaps + 1, late_frames)]  # a late frame isn't a lost one
        # Frame numbers that jump: the frames were read but the writer's queue
        # was full (slow SD card). Else a frame read late right after a gap
        # waited in appsink: the recording loop stalled and appsink dropped the
        # frames meanwhile. Otherwise the camera itself didn't deliver them.
        not_saved = np.zeros(len(gaps), bool)
        if rec.index is not None:
            not_saved = np.diff(rec.index)[gaps] > 1
        err = read_errors_ms(rec)
        late = np.array([err is not None and err[i + 1] > max_error_ms for i in gaps], bool)
        stalled = late & ~not_saved
        for label, sel in (
            ("not saved, writer queue full", not_saved),
            ("recording loop stalled", stalled),
            ("camera missed frames", ~stalled & ~not_saved),
        ):
            g = gaps[sel]
            if len(g):
                missed = int(np.round(pdt[g] / period).sum() - len(g))
                where = ", ".join(f"frame {i + 1} ({pdt[i]:.0f} ms)" for i in g[:5])
                checks.append(
                    Check(
                        WARN, "frames", f"{label}: ~{missed} frames lost in {len(g)} gaps: {where}"
                    )
                )
    return checks


def pts_late_frames(rec: Recording, fps: float) -> np.ndarray:
    """
    Frames the camera delivered late: a PTS interval over 1.25 periods followed
    by one under 0.75 (seq03: 51 ms then 16-19 ms). Nothing was lost, but the
    frame's PTS, and so its read time, is ~half a period late: the read-time
    check can't see that, since read - PTS stays normal.
    """
    if rec.pts_ns is None or len(rec.pts_ns) < 3:
        return np.zeros(0, dtype=np.int64)
    period = 1000.0 / fps
    pdt = np.diff(rec.pts_ns) / 1e6
    late = (pdt[:-1] > 1.25 * period) & (pdt[1:] < 0.75 * period)
    if rec.index is not None:
        late &= np.diff(rec.index)[:-1] == 1  # not right after a writer drop
    return np.where(late)[0] + 1


def read_errors_ms(rec: Recording) -> Optional[np.ndarray]:
    """Per frame: how much later than usual it was read (read - PTS - median), in ms."""
    if rec.read_ns is None or rec.pts_ns is None:
        return None
    lag = (rec.read_ns - rec.pts_ns).astype(np.float64) / 1e6
    return lag - np.median(lag)


def _check_read_time(rec: Recording, skip_start_s: float, max_error_ms: float) -> List[Check]:
    err = read_errors_ms(rec)
    if err is None:
        return [Check(WARN, "read time", "no PTS recorded: read-time error unknown")]
    after = (rec.frame_ns - rec.frame_ns[0]) / 1e9 >= skip_start_s
    start_late = int((err[~after] > max_error_ms).sum())
    e = err[after]
    late = int((e > max_error_ms).sum())
    msg = f"error p99 {np.percentile(e, 99):.2f} ms, max {e.max():.2f} ms"
    if start_late:
        msg += f"; {start_late} late frames in the first {skip_start_s:g} s (skipped)"
    if late:
        return [Check(WARN, "read time", f"{msg}; {late} frames > {max_error_ms:g} ms late")]
    return [Check(OK, "read time", msg)]


def _check_imu(rec: Recording, rate_hz: float) -> List[Check]:
    t = rec.imu_ns
    if len(t) < 2:
        return [Check(FAIL, "IMU", f"only {len(t)} samples")]
    dt = np.diff(t) / 1e6
    if (dt <= 0).any():
        return [Check(FAIL, "IMU", f"timestamps not increasing at {int((dt <= 0).sum())} samples")]
    rate = (len(t) - 1) / ((t[-1] - t[0]) / 1e9)
    period = 1000.0 / rate_hz
    late = int((dt > 1.5 * period).sum())
    checks = []
    level = OK if abs(rate - rate_hz) < 0.02 * rate_hz else WARN
    checks.append(
        Check(
            level, "IMU", f"{len(t)} samples, {rate:.1f} Hz, max gap {dt.max():.1f} ms, late {late}"
        )
    )
    if dt.max() > 50.0:
        checks.append(Check(WARN, "IMU", f"gap of {dt.max():.0f} ms (> 50 ms)"))
    f0, f1 = rec.frame_ns[0], rec.frame_ns[-1]
    if t[0] > f0 or t[-1] < f1:
        checks.append(
            Check(
                FAIL,
                "IMU",
                f"doesn't cover the frames: IMU starts {(t[0] - f0) / 1e6:+.0f} ms, "
                f"ends {(t[-1] - f1) / 1e6:+.0f} ms from the first/last frame",
            )
        )
    return checks


def frames_to_keep(
    rec: Recording, skip_start_s: float, max_error_ms: float, fps: float = _FPS
) -> List[int]:
    after = (rec.frame_ns - rec.frame_ns[0]) / 1e9 >= skip_start_s
    err = read_errors_ms(rec)
    ok = after if err is None else after & (err <= max_error_ms)
    ok[pts_late_frames(rec, fps)] = False
    # The IMU must start before the first frame ORB-SLAM3 sees.
    ok &= rec.frame_ns > rec.imu_ns[0]
    return [int(i) for i in np.where(ok)[0]]


def check_recording(
    root: Path,
    skip_start_s: float = 1.0,
    max_error_ms: float = 5.0,
    fps: float = _FPS,
    imu_rate_hz: float = _IMU_RATE_HZ,
) -> Report:
    try:
        rec = load_recording(root)
    except (OSError, ValueError, IndexError) as e:
        return Report(root, [Check(FAIL, "layout", f"can't read the dataset: {e}")], [])
    checks = _check_layout(root, rec)
    checks += _check_frames(rec, fps, skip_start_s, max_error_ms)
    if not any(c.level == FAIL and c.name == "frames" for c in checks):
        checks += _check_read_time(rec, skip_start_s, max_error_ms)
    checks += _check_imu(rec, imu_rate_hz)
    keep = frames_to_keep(rec, skip_start_s, max_error_ms, fps) if len(rec.imu_ns) else []
    return Report(root, checks, keep)


def write_trimmed_times(root: Path, report: Report) -> Path:
    rec = load_recording(root)
    out = root / "times_trimmed.txt"
    out.write_text("".join(f"{rec.frame_ns[i]}\n" for i in report.keep))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("dirs", nargs="+", type=Path, metavar="DIR", help="Recording directories")
    parser.add_argument(
        "--skip-start",
        type=float,
        default=1.0,
        metavar="S",
        help="Seconds of frames to ignore at the start (default: 1)",
    )
    parser.add_argument(
        "--max-read-error-ms",
        type=float,
        default=5.0,
        metavar="MS",
        help="Frames read later than this are flagged and trimmed (default: 5)",
    )
    parser.add_argument(
        "--write-times",
        action="store_true",
        help="Write DIR/times_trimmed.txt for ORB-SLAM3",
    )
    args = parser.parse_args(argv)

    failed = False
    for root in args.dirs:
        report = check_recording(root, args.skip_start, args.max_read_error_ms)
        failed |= report.failed
        print(f"\n{root}")
        for c in report.checks:
            print(f"  {c.level:<4}  {c.name:<9}  {c.message}")
        if args.write_times and not report.failed:
            out = write_trimmed_times(root, report)
            print(f"  wrote {out.name}: {len(report.keep)} frames")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
