"""
Tests for lib/imu.py — BNO055 register handling, the sampling loop and the
reader's statistics, with a fake I2C bus and clock (no hardware needed) — and
for the `map imu-check` command in commands/mapping.py.

Run with:
    pytest tests/imu/ -v
"""

import argparse
import queue
import struct
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from lib import imu
from lib.imu import BNO055, ImuConfig, ImuReader, ImuSample, decode, sample_loop
from lib.settings import NanoSettings


class FakeBus:
    """Records writes; returns a chip ID and a fixed 18-byte data block."""

    def __init__(self, chip_id=0xA0, data=None):
        self.chip_id = chip_id
        self.data = data or [0] * 18
        self.writes = []
        self.block_reads = []

    def read_byte_data(self, _addr, reg):
        assert reg == 0x00
        return self.chip_id

    def write_byte_data(self, addr, reg, value):
        self.writes.append((addr, reg, value))

    def read_i2c_block_data(self, addr, reg, length):
        self.block_reads.append((addr, reg, length))
        return list(self.data)


def _raw(acc=(0, 0, 0), mag=(0, 0, 0), gyro=(0, 0, 0)):
    return list(struct.pack("<9h", *acc, *mag, *gyro))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(imu.time, "sleep", lambda _s: None)


class FakeClock:
    """Nanosecond clock that only moves when sleep() is called or a read takes time."""

    def __init__(self):
        self.ns = 1_000_000_000

    def __call__(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += int(seconds * 1e9)


# decode()
class TestDecode:
    def test_units(self):
        gyro, acc = decode(_raw(acc=(981, -100, 0), gyro=(900, -450, 9)))
        assert acc == pytest.approx((9.81, -1.0, 0.0))
        assert gyro == pytest.approx((1.0, -0.5, 0.01))

    def test_magnetometer_ignored(self):
        gyro, acc = decode(_raw(mag=(1234, -5678, 42)))
        assert acc == (0.0, 0.0, 0.0)
        assert gyro == (0.0, 0.0, 0.0)


# BNO055
class TestBNO055:
    def test_setup_writes(self):
        bus = FakeBus()
        BNO055(bus).setup()
        regs = [(reg, val) for _addr, reg, val in bus.writes]
        assert regs[0] == (0x3D, 0x00)  # CONFIG mode first
        assert regs[-1] == (0x3D, 0x05)  # ACCGYRO mode last
        # Page 1 sensor configs are written between the page switches.
        p1, p0 = regs.index((0x07, 1)), regs.index((0x07, 0))
        assert set(regs[p1 + 1 : p0]) == {(0x08, 0x11), (0x0A, 0x1A), (0x0B, 0x00)}
        assert (0x3B, 0x02) in regs[p0:]  # m/s², rad/s
        assert all(addr == 0x28 for addr, _r, _v in bus.writes)

    def test_wrong_chip_id(self):
        with pytest.raises(RuntimeError, match="chip ID 0x00"):
            BNO055(FakeBus(chip_id=0x00)).setup()

    def test_read_is_one_transfer(self):
        bus = FakeBus(data=_raw(acc=(100, 0, 0), gyro=(0, 0, 900)))
        gyro, acc = BNO055(bus, address=0x29).read()
        assert bus.block_reads == [(0x29, 0x08, 18)]
        assert acc == pytest.approx((1.0, 0.0, 0.0))
        assert gyro == pytest.approx((0.0, 0.0, 1.0))


# sample_loop()
def _run_loop(clock, reads, read_ns=0, period_s=0.005, max_samples=None):
    """Run sample_loop over `reads` (values or exceptions); stop when they run out."""
    reads = list(reads)
    out, errors = [], []

    def read():
        clock.ns += read_ns
        r = reads.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def should_stop():
        return not reads or (max_samples is not None and len(out) >= max_samples)

    sample_loop(read, period_s, out.append, should_stop, errors.append, clock, clock.sleep)
    return out, errors


_READING = ((0.0, 0.0, 0.0), (0.0, 0.0, 9.8))


class TestSampleLoop:
    def test_fixed_period(self):
        clock = FakeClock()
        out, _ = _run_loop(clock, [_READING] * 5)
        dts = [b.t_ns - a.t_ns for a, b in zip(out, out[1:])]
        assert dts == [5_000_000] * 4

    def test_timestamp_is_read_midpoint(self):
        clock = FakeClock()
        start = clock.ns
        out, _ = _run_loop(clock, [_READING], read_ns=2_000_000)
        assert out[0].t_ns == start + 1_000_000

    def test_slow_reads_keep_schedule(self):
        # A 2 ms read inside a 5 ms period: deadlines are absolute, so the
        # interval stays 5 ms instead of becoming 7 ms.
        clock = FakeClock()
        out, _ = _run_loop(clock, [_READING] * 4, read_ns=2_000_000)
        dts = [b.t_ns - a.t_ns for a, b in zip(out, out[1:])]
        assert dts == [5_000_000] * 3

    def test_error_skipped(self):
        clock = FakeClock()
        out, errors = _run_loop(clock, [_READING, OSError("nack"), _READING])
        assert len(out) == 2 and len(errors) == 1
        assert out[1].t_ns - out[0].t_ns == 10_000_000  # the failed slot is a gap

    def test_gives_up_after_consecutive_errors(self):
        clock = FakeClock()
        n = imu._MAX_CONSECUTIVE_ERRORS
        with pytest.raises(RuntimeError, match="failed in a row"):
            _run_loop(clock, [OSError("gone")] * (n + 5))

    def test_errors_reset_by_success(self):
        clock = FakeClock()
        n = imu._MAX_CONSECUTIVE_ERRORS
        reads = ([OSError("x")] * (n - 1) + [_READING]) * 2 + [_READING]
        out, errors = _run_loop(clock, reads)
        assert len(errors) == 2 * (n - 1) and len(out) == 3

    def test_no_burst_after_stall(self):
        # One read stalls 50 ms (10 periods): the next read comes one period
        # later, not as 10 back-to-back catch-up reads.
        clock = FakeClock()
        stalled = {"done": False}
        out = []

        def read():
            if len(out) == 1 and not stalled["done"]:
                clock.ns += 50_000_000
                stalled["done"] = True
            return _READING

        sample_loop(
            read, 0.005, out.append, lambda: len(out) >= 4, lambda _e: None, clock, clock.sleep
        )
        dts = [b.t_ns - a.t_ns for a, b in zip(out, out[1:])]
        assert dts[0] == 30_000_000  # the stall: one period + half the 50 ms read (midpoint)
        assert all(dt >= 5_000_000 for dt in dts[1:])


# ImuReader (without a child process: the queue is filled directly)
def _reader_with(samples, rate_hz=200.0):
    reader = ImuReader(rate_hz=rate_hz)
    reader._queue = queue.Queue()
    for s in samples:
        reader._queue.put(s)
    return reader


def _sample(t_ms):
    return ImuSample(int(t_ms * 1e6), (0.0, 0.0, 0.0), (0.0, 0.0, 9.8))


class TestImuReader:
    def test_drain_returns_in_order_then_empty(self):
        samples = [_sample(5 * i) for i in range(4)]
        reader = _reader_with(samples)
        assert reader.drain() == samples
        assert not reader.drain()

    def test_stats(self):
        # 5 ms intervals with one 20 ms gap (late: over 1.5 x 5 ms).
        times = [0, 5, 10, 15, 35, 40]
        reader = _reader_with([_sample(t) for t in times])
        reader.drain()
        st = reader.stats()
        assert st.samples == 6
        assert st.dt_mean_ms == pytest.approx(8.0)
        assert st.dt_max_ms == pytest.approx(20.0)
        assert st.late == 1
        assert st.rate_hz == pytest.approx(125.0)
        assert st.errors == 0

    def test_stats_across_drains(self):
        reader = _reader_with([_sample(0), _sample(5)])
        reader.drain()
        reader._queue.put(_sample(10))
        reader.drain()
        st = reader.stats()
        assert st.samples == 3 and st.dt_max_ms == pytest.approx(5.0)

    def test_stats_empty(self):
        st = ImuReader().stats()
        assert st.samples == 0 and st.rate_hz == 0.0

    def test_drain_raises_when_child_failed(self):
        reader = _reader_with([])

        class _Flag:
            value = 1

        reader._failed = _Flag()
        with pytest.raises(RuntimeError, match="failed"):
            reader.drain()

    def test_stop_without_start(self):
        ImuReader().stop()


# ImuConfig
class TestImuConfig:
    def test_defaults(self):
        cfg = ImuConfig()
        assert (cfg.bus, cfg.address, cfg.rate_hz) == (0, 0x28, 200.0)

    @pytest.mark.parametrize("kwargs", [{"address": 0x30}, {"rate_hz": 0}, {"rate_hz": 1000}])
    def test_invalid(self, kwargs):
        with pytest.raises(ValidationError):
            ImuConfig(**kwargs)


# commands/mapping.py — `map imu-check`
def _parse(args):
    from commands.mapping import register

    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    register(sub.add_parser("map"), NanoSettings())
    return parser.parse_args(["map"] + args)


class TestImuCheckCommand:
    def test_defaults(self):
        ns = _parse(["imu-check"])
        assert ns.seconds == pytest.approx(10.0)
        assert ns.rate_hz == pytest.approx(200.0)
        assert ns.bus == 0
        assert ns.load is False

    def test_custom(self):
        ns = _parse(["imu-check", "-t", "30", "--rate", "100", "--bus", "1", "--load"])
        assert ns.seconds == pytest.approx(30.0)
        assert ns.rate_hz == pytest.approx(100.0)
        assert ns.bus == 1
        assert ns.load is True

    def test_func_is_set(self):
        from commands.mapping import _run_imu_check

        ns = _parse(["imu-check"])
        assert ns.func is _run_imu_check

    @patch("lib.imu.ImuCheck")
    def test_run_imu_check_builds_config_and_runs(self, mock_impl_cls):
        from commands.mapping import _run_imu_check

        _run_imu_check(_parse(["imu-check", "-t", "5", "--rate", "100"]))

        mock_impl_cls.assert_called_once_with(seconds=5.0, rate_hz=100.0, bus=0, load=False)
        mock_impl_cls.return_value.run.assert_called_once()
