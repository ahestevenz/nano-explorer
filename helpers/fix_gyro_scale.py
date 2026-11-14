#!/usr/bin/env python3
"""
Rescale the gyroscope in `nano-explorer map record` datasets made before
lib/imu.py applied GYR_SCALE (the BNO055 doesn't report the true rate; see
ADDING_IMU.md, hand-rotation tests).

    python3 helpers/fix_gyro_scale.py --from-range 500 ~/recordings/seq0[1-5]
    python3 helpers/fix_gyro_scale.py --from-range 2000 ~/recordings/seq06

--from-range is the gyro range the recording was made with:
  500   lib/imu.py before ed1f9a5: read 3.669x the true rate  -> x 1/3.669
  2000  ed1f9a5, before GYR_SCALE: read 0.920x the true rate  -> x 1.087
(--scale S sets the factor directly instead.)

The original is kept as mav0/imu0/data_raw.csv and the fix is noted in
meta.json ("gyro_fix"). A recording that is already fixed, or was recorded
with the correction (meta.json "imu.gyro_scale"), is skipped. Only the three
gyro columns change; timestamps and accelerometer are copied as they are.
Needs only the standard library.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Measured true / reported, per recording range (ADDING_IMU.md).
SCALE_BY_RANGE = {500: 1 / 3.669, 2000: 1.087}


def fix_recording(root: Path, scale: float, source: str) -> str:
    """Rescale root's gyro columns; returns what was done (or why not)."""
    imu = root / "mav0/imu0/data.csv"
    raw = root / "mav0/imu0/data_raw.csv"
    meta_path = root / "meta.json"
    if not imu.exists():
        return f"skipped: {imu} not found"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if raw.exists() or "gyro_fix" in meta:
        return "skipped: already fixed"
    if "gyro_scale" in meta.get("imu", {}):
        return "skipped: recorded with the gyro correction already applied"

    shutil.copy2(str(imu), str(raw))
    lines = raw.read_text().splitlines()
    out = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            out.append(line)
            continue
        cols = line.split(",")
        gyro = [f"{float(v) * scale:.6f}" for v in cols[1:4]]
        out.append(",".join([cols[0]] + gyro + cols[4:]))
    imu.write_text("\n".join(out) + "\n")

    meta["gyro_fix"] = {
        "scale": scale,
        "source": source,
        "original": "mav0/imu0/data_raw.csv",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return f"gyro x {scale:.4f} ({source}); original kept as data_raw.csv"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("dirs", nargs="+", type=Path, metavar="DIR", help="Recording directories")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-range",
        type=int,
        choices=sorted(SCALE_BY_RANGE),
        help="Gyro range (dps) the recordings were made with",
    )
    group.add_argument("--scale", type=float, help="Multiply the gyro columns by this")
    args = parser.parse_args(argv)

    if args.scale is not None:
        scale, source = args.scale, "--scale"
    else:
        scale = SCALE_BY_RANGE[args.from_range]
        source = f"recorded at ±{args.from_range} dps"
    for root in args.dirs:
        print(f"{root}: {fix_recording(root, scale, source)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
