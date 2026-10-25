"""
BNO055 IMU on the Jetson Nano's I2C bus 0 (header pins 27/28).

Raw accelerometer + gyroscope for visual-inertial SLAM (see ADDING_IMU.md):

    with ImuReader(rate_hz=200) as imu:
        ...
        for s in imu.drain():   # every sample read since the last call
            s.t_ns, s.gyro, s.acc

The chip runs in ACCGYRO mode: no on-chip fusion (its output would be filtered
and late) and no magnetometer (the motors and indoor steel throw it off).
Registers are read directly with smbus2 — Adafruit's BNO055 library and Blinka
no longer support Python 3.6.

Reading happens in its own process, so SLAM and the camera loop can't delay the
sampling. Every sample is timestamped with CLOCK_MONOTONIC, the same clock the
camera frames must use, at the midpoint of its I2C read.
"""

import collections
import multiprocessing as mp
import queue
import struct
import time
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

# Register map (BNO055 datasheet, section 4.2). Page 0 unless noted.
_REG_CHIP_ID = 0x00
_REG_PAGE_ID = 0x07
_REG_ACC_DATA = 0x08  # ACC x/y/z, MAG x/y/z, GYR x/y/z: 18 bytes, int16 little-endian
_REG_UNIT_SEL = 0x3B
_REG_OPR_MODE = 0x3D
_REG_PWR_MODE = 0x3E
_REG_ACC_CONFIG = 0x08  # page 1
_REG_GYR_CONFIG_0 = 0x0A  # page 1
_REG_GYR_CONFIG_1 = 0x0B  # page 1

_CHIP_ID = 0xA0
_MODE_CONFIG = 0x00
_MODE_ACCGYRO = 0x05
_PWR_NORMAL = 0x00

# UNIT_SEL: accelerometer in m/s² (bit 0 clear), gyroscope in rad/s (bit 1 set):
# the units ORB-SLAM3 and Kalibr expect.
_UNIT_SEL = 0x02
_ACC_LSB_PER_MS2 = 100.0
_GYR_LSB_PER_RADS = 900.0

# Sensor configs only stick in non-fusion modes like ACCGYRO (fusion modes
# overwrite them). They are fixed here because the noise values calibrated from
# an Allan-variance recording are only valid for these exact ranges and filters.
# Accelerometer: ±4 g, 125 Hz bandwidth (data updates at 2x that, above 200 Hz
# polling), normal power.
_ACC_CONFIG = 0b000_100_01
# Gyroscope: ±500 dps (a hand-waved Kalibr sequence can pass 250 dps), 47 Hz
# bandwidth (400 Hz data rate), normal power.
_GYR_CONFIG_0 = 0b00_011_010
_GYR_CONFIG_1 = 0x00

# The datasheet's mode switch times: 19 ms from CONFIG to any other mode, 7 ms
# back. Padded, and the first data after switching is still all zeros.
_MODE_SWITCH_S = 0.03
_FIRST_DATA_S = 0.05

# Give up after this many I2C failures in a row: the chip is gone, not glitching.
_MAX_CONSECUTIVE_ERRORS = 50


class ImuSample(NamedTuple):
    """One reading: CLOCK_MONOTONIC time [ns], gyro (x, y, z) [rad/s], acc (x, y, z) [m/s²]."""

    t_ns: int
    gyro: Tuple[float, float, float]
    acc: Tuple[float, float, float]


class ImuStats(NamedTuple):
    """Sampling quality since the reader started. Intervals in ms."""

    samples: int
    rate_hz: float
    dt_mean_ms: float
    dt_std_ms: float
    dt_p99_ms: float
    dt_max_ms: float
    late: int  # intervals over 1.5x the target period
    errors: int  # failed I2C reads


def monotonic_ns() -> int:
    """CLOCK_MONOTONIC in ns (time.monotonic_ns() is Python 3.7+)."""
    return int(time.clock_gettime(time.CLOCK_MONOTONIC) * 1e9)


def decode(raw: List[int]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """(gyro [rad/s], acc [m/s²]) from the 18 data bytes starting at ACC_DATA."""
    ax, ay, az, _mx, _my, _mz, gx, gy, gz = struct.unpack("<9h", bytes(raw))
    acc = (ax / _ACC_LSB_PER_MS2, ay / _ACC_LSB_PER_MS2, az / _ACC_LSB_PER_MS2)
    gyro = (gx / _GYR_LSB_PER_RADS, gy / _GYR_LSB_PER_RADS, gz / _GYR_LSB_PER_RADS)
    return gyro, acc


class BNO055:
    """
    Register-level driver. `bus` is an open smbus2.SMBus (or anything with the
    same read_byte_data / write_byte_data / read_i2c_block_data methods).
    """

    def __init__(self, bus: Any, address: int = 0x28):
        self._bus = bus
        self._address = address

    def setup(self) -> None:
        """Check the chip, then configure it for raw ACCGYRO output."""
        chip_id = self._bus.read_byte_data(self._address, _REG_CHIP_ID)
        if chip_id != _CHIP_ID:
            raise RuntimeError(
                f"No BNO055 at 0x{self._address:02x}: chip ID 0x{chip_id:02x}, "
                f"expected 0x{_CHIP_ID:02x}. Check the wiring (bus 0, pins 27/28)."
            )
        self._write(_REG_OPR_MODE, _MODE_CONFIG)
        time.sleep(_MODE_SWITCH_S)
        self._write(_REG_PWR_MODE, _PWR_NORMAL)
        self._write(_REG_PAGE_ID, 1)
        self._write(_REG_ACC_CONFIG, _ACC_CONFIG)
        self._write(_REG_GYR_CONFIG_0, _GYR_CONFIG_0)
        self._write(_REG_GYR_CONFIG_1, _GYR_CONFIG_1)
        self._write(_REG_PAGE_ID, 0)
        self._write(_REG_UNIT_SEL, _UNIT_SEL)
        self._write(_REG_OPR_MODE, _MODE_ACCGYRO)
        time.sleep(_MODE_SWITCH_S + _FIRST_DATA_S)

    def read(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """(gyro [rad/s], acc [m/s²]) in one I2C transfer, so both come from the same moment."""
        return decode(self._bus.read_i2c_block_data(self._address, _REG_ACC_DATA, 18))

    def _write(self, reg: int, value: int) -> None:
        self._bus.write_byte_data(self._address, reg, value)


def sample_loop(
    read: Callable[[], Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    period_s: float,
    emit: Callable[[ImuSample], None],
    should_stop: Callable[[], bool],
    on_error: Callable[[Exception], None],
    clock_ns: Callable[[], int] = monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """
    Call `read` every `period_s` and `emit` each timestamped sample until
    `should_stop()`. Deadlines are absolute, so a late read doesn't push every
    later one back; after a stall longer than a period the schedule restarts
    from now instead of firing a burst of catch-up reads. Raises RuntimeError
    after _MAX_CONSECUTIVE_ERRORS failed reads in a row.
    """
    period_ns = int(period_s * 1e9)
    next_ns = clock_ns()
    consecutive_errors = 0
    while not should_stop():
        wait_ns = next_ns - clock_ns()
        if wait_ns > 0:
            sleep(wait_ns / 1e9)
        t0 = clock_ns()
        try:
            gyro, acc = read()
        except OSError as e:  # I2C NACK / timeout (BNO055 clock stretching)
            consecutive_errors += 1
            on_error(e)
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(f"BNO055: {consecutive_errors} I2C reads failed in a row") from e
        else:
            consecutive_errors = 0
            emit(ImuSample((t0 + clock_ns()) // 2, gyro, acc))
        next_ns += period_ns
        if clock_ns() - next_ns > period_ns:
            next_ns = clock_ns()


def _reader_process(cfg: "ImuConfig", out: Any, stop: Any, errors: Any, failed: Any) -> None:
    """Child process: open the bus, set up the chip, sample until `stop` is set."""
    from smbus2 import SMBus  # deferred: not installed on CI

    def _count_error(_e: Exception) -> None:
        with errors.get_lock():
            errors.value += 1

    try:
        with SMBus(cfg.bus) as bus:
            imu = BNO055(bus, cfg.address)
            imu.setup()
            sample_loop(imu.read, 1.0 / cfg.rate_hz, out.put, stop.is_set, _count_error)
    except Exception as e:  # pylint: disable=broad-except
        failed.value = 1
        logger.error(f"IMU reader stopped: {e}")


class ImuConfig(BaseModel):
    """
    Args:
        bus:      I2C bus number (/dev/i2c-N). The BNO055 is on bus 0, apart from
                  the motor driver on bus 1.
        address:  0x28, or 0x29 with the board's I2C_ADDR pin pulled high.
        rate_hz:  Polling rate.
    """

    bus: int = Field(0, ge=0)
    address: int = Field(0x28, ge=0x28, le=0x29)
    rate_hz: float = Field(200.0, gt=0, le=400)


class _IntervalStats:
    """
    Running sample-interval statistics: totals since the start, plus a bounded
    window of recent intervals for the percentile, so a multi-hour stationary
    recording for the Allan variance doesn't grow memory.
    """

    _WINDOW = 12000  # one minute at 200 Hz

    def __init__(self, period_ms: float):
        self._late_ms = 1.5 * period_ms
        self._last_ns: Optional[int] = None
        self.count = 0
        self._n_dt = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self._max = 0.0
        self._late = 0
        self._recent: Any = collections.deque(maxlen=self._WINDOW)

    def add(self, t_ns: int) -> None:
        self.count += 1
        if self._last_ns is not None:
            dt = (t_ns - self._last_ns) / 1e6
            self._n_dt += 1
            self._sum += dt
            self._sumsq += dt * dt
            self._max = max(self._max, dt)
            self._late += dt > self._late_ms
            self._recent.append(dt)
        self._last_ns = t_ns

    def snapshot(self, errors: int) -> ImuStats:
        if not self._n_dt:
            return ImuStats(self.count, 0.0, 0.0, 0.0, 0.0, 0.0, 0, errors)
        mean = self._sum / self._n_dt
        std = max(0.0, self._sumsq / self._n_dt - mean * mean) ** 0.5
        return ImuStats(
            samples=self.count,
            rate_hz=1000.0 / mean,
            dt_mean_ms=mean,
            dt_std_ms=std,
            dt_p99_ms=float(np.percentile(np.fromiter(self._recent, float), 99)),
            dt_max_ms=self._max,
            late=self._late,
            errors=errors,
        )


class ImuReader:
    """
    Runs the BNO055 sampling loop in a child process. Call drain() regularly
    (every frame is fine): samples queue up in memory until drained.
    """

    def __init__(self, **kwargs: Any):
        self._config = ImuConfig(**kwargs)
        self._queue: Any = None
        self._stop: Any = None
        self._errors: Any = None
        self._failed: Any = None
        self._proc: Optional[mp.Process] = None
        self._pending: List[ImuSample] = []  # collected while stopping
        self._intervals = _IntervalStats(1000.0 / self._config.rate_hz)

    @property
    def config(self) -> ImuConfig:
        return self._config

    def start(self) -> None:
        self._queue = mp.Queue()
        self._stop = mp.Event()
        self._errors = mp.Value("i", 0)
        self._failed = mp.Value("b", 0)
        self._proc = mp.Process(
            target=_reader_process,
            args=(self._config, self._queue, self._stop, self._errors, self._failed),
            name="imu-reader",
            daemon=True,
        )
        self._proc.start()
        logger.info(
            f"IMU reader started: BNO055 at 0x{self._config.address:02x} on "
            f"/dev/i2c-{self._config.bus}, {self._config.rate_hz:g} Hz"
        )

    def drain(self) -> List[ImuSample]:
        """Every sample read since the last call, oldest first."""
        samples, self._pending = self._pending + self._take(), []
        for s in samples:
            self._intervals.add(s.t_ns)
        if not samples and self._failed is not None and self._failed.value:
            raise RuntimeError("IMU reader process failed; see the log above")
        return samples

    def stats(self) -> ImuStats:
        """Timing of every sample drained so far."""
        return self._intervals.snapshot(self._errors.value if self._errors is not None else 0)

    def stop(self) -> None:
        """Stop the child; samples it read before stopping stay available to drain()."""
        if self._proc is None:
            return
        self._stop.set()
        # Keep emptying the queue while waiting: a child with queued data that
        # nobody reads never exits (multiprocessing's "joining processes that
        # use queues" pitfall).
        deadline = time.monotonic() + 2.0
        while self._proc.is_alive() and time.monotonic() < deadline:
            self._pending.extend(self._take())
            self._proc.join(timeout=0.05)
        self._pending.extend(self._take())
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None
        logger.info("IMU reader stopped.")

    def _take(self) -> List[ImuSample]:
        samples: List[ImuSample] = []
        if self._queue is None:
            return samples
        while True:
            try:
                samples.append(self._queue.get_nowait())
            except queue.Empty:
                return samples

    def __enter__(self) -> "ImuReader":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def _burn(stop: Any) -> None:
    """Busy loop until `stop` is set: stands in for SLAM's CPU load."""
    while not stop.is_set():
        sum(i * i for i in range(10000))


class ImuCheckConfig(BaseModel):
    """
    Args:
        seconds:  How long to run the reader.
        rate_hz:  Polling rate.
        bus:      I2C bus number.
        load:     Keep every other core busy, to see the jitter the reader gets
                  while ORB-SLAM3 is running.
    """

    seconds: float = Field(10.0, gt=0)
    rate_hz: float = Field(200.0, gt=0, le=400)
    bus: int = Field(0, ge=0)
    load: bool = False


class ImuCheck:
    """
    Health and timing check of the BNO055 and the reader process
    (`nano-explorer map imu-check`). Logs the latest reading about once a
    second; at the end, the sampling statistics and the mean of every reading.

    What to look for:
      - rate close to the target, few late intervals and no errors. The p99 and
        max interval show how bad the jitter gets: that decides whether the
        reader can stay in Python (ADDING_IMU.md, stage 1).
      - Robot still: |acc| about 9.8 m/s² (a few % off is an uncalibrated
        bias), gyro mean near 0 rad/s (what's left is its bias).
    """

    def __init__(self, **kwargs: Any):
        self._config = ImuCheckConfig(**kwargs)

    def run(self) -> ImuStats:
        cfg = self._config
        stop_load = mp.Event()
        burners = []
        if cfg.load:
            for _ in range(max(1, mp.cpu_count() - 1)):
                p = mp.Process(target=_burn, args=(stop_load,), daemon=True)
                p.start()
                burners.append(p)
            logger.info(f"CPU load: {len(burners)} busy processes")

        gyro: List[Tuple[float, float, float]] = []
        acc: List[Tuple[float, float, float]] = []
        try:
            with ImuReader(bus=cfg.bus, rate_hz=cfg.rate_hz) as imu:
                t_end = time.monotonic() + cfg.seconds
                while time.monotonic() < t_end:
                    time.sleep(1.0)
                    samples = imu.drain()
                    if not samples:
                        continue
                    gyro.extend(s.gyro for s in samples)
                    acc.extend(s.acc for s in samples)
                    g, a = samples[-1].gyro, samples[-1].acc
                    logger.info(
                        f"acc m/s2: {a[0]:6.2f} {a[1]:6.2f} {a[2]:6.2f}   "
                        f"gyro rad/s: {g[0]:7.3f} {g[1]:7.3f} {g[2]:7.3f}   "
                        f"({len(samples)} samples)"
                    )
                imu.stop()
                samples = imu.drain()
                gyro.extend(s.gyro for s in samples)
                acc.extend(s.acc for s in samples)
                st = imu.stats()
        finally:
            stop_load.set()
            for p in burners:
                p.join(timeout=1.0)

        logger.info(f"samples:   {st.samples}   errors: {st.errors}")
        logger.info(f"rate:      {st.rate_hz:.1f} Hz (target {cfg.rate_hz:g})")
        logger.info(
            f"interval:  mean {st.dt_mean_ms:.2f} ms, std {st.dt_std_ms:.2f}, "
            f"p99 {st.dt_p99_ms:.2f}, max {st.dt_max_ms:.2f}, late {st.late}"
        )
        if acc:
            a_mean, g_mean = np.mean(acc, axis=0), np.mean(gyro, axis=0)
            logger.info(
                f"acc mean:  {a_mean[0]:.3f} {a_mean[1]:.3f} {a_mean[2]:.3f} m/s2   "
                f"|acc| {np.linalg.norm(a_mean):.3f}"
            )
            logger.info(f"gyro mean: {g_mean[0]:.4f} {g_mean[1]:.4f} {g_mean[2]:.4f} rad/s")
        return st
