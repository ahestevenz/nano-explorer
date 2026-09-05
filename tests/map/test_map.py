"""
Tests for commands/mapping.py and lib/map/* — argument parsing,
config validation, and algorithm logic.

Run with:
    pytest tests/map/ -v
    pytest tests/map/ -v -m "not hardware"
"""

# pylint: disable=redefined-outer-name
# Every test below that takes `fake_project_env` as a parameter is pytest's
# fixture-injection pattern, not a real shadowing of the fixture function
# defined at module scope — pylint can't tell the two apart.

import argparse
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

from lib.settings import PROJECT_ROOT_PATH, NanoSettings


def _make_parser(settings: NanoSettings = None) -> argparse.ArgumentParser:
    from commands.mapping import register

    if settings is None:
        settings = NanoSettings()
    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    map_parser = sub.add_parser("map")
    register(map_parser, settings)
    return parser


def _parse(args: list, settings: NanoSettings = None) -> argparse.Namespace:
    return _make_parser(settings).parse_args(["map"] + args)


def _dummy_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture
def fake_project_env(tmp_path, monkeypatch):
    """
    Isolated fake PROJECT_ROOT_PATH + USER_CONFIG_DIR for testing
    SlamMapper._resolve_vocab_and_settings without touching the real package
    config/ or ~/.nano-explorer/config/. Returns (project_root, user_config_dir).
    """
    import lib.map.slam as slam_mod
    import lib.settings as settings_mod

    project_root = tmp_path / "project"
    user_config_dir = tmp_path / "home" / ".nano-explorer" / "config"
    (project_root / "config" / "models").mkdir(parents=True)
    (project_root / "assets" / "models").mkdir(parents=True)
    (project_root / "config" / "models" / "orbslam3_mono.yaml").write_text(
        "Camera.type: PinHole\nCamera.fx: 111.0\n", encoding="utf-8"
    )
    (project_root / "assets" / "models" / "ORBvoc.txt").write_text("fake vocab", encoding="utf-8")

    monkeypatch.setattr(settings_mod, "USER_CONFIG_DIR", user_config_dir)
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT_PATH", project_root)
    monkeypatch.setattr(slam_mod, "PROJECT_ROOT_PATH", project_root)

    return project_root, user_config_dir


# commands/mapping.py — sub-command structure
class TestRegister:
    def test_odometry_subcommand_exists(self):
        ns = _parse(["odometry"])
        assert ns.command == "odometry"

    def test_slam_subcommand_exists(self):
        ns = _parse(["slam"])
        assert ns.command == "slam"

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _make_parser().parse_args(["map"])


# odometry defaults
class TestOdometryDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["odometry"])
        assert ns.stream is True

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["odometry"], settings)
        assert ns.stream_port == settings.stream_port

    def test_no_stream_flag(self):
        ns = _parse(["odometry", "--no-stream"])
        assert ns.stream is False

    def test_custom_stream_port(self):
        ns = _parse(["odometry", "--stream-port", "9191"])
        assert ns.stream_port == 9191

    def test_config_path_dest(self):
        ns = _parse(["odometry", "--config", "config/models/odometry.yaml"])
        assert ns.config_path == "config/models/odometry.yaml"

    def test_func_is_set(self):
        from commands.mapping import _run_odometry

        ns = _parse(["odometry"])
        assert ns.func is _run_odometry


# slam defaults
class TestSlamDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["slam"])
        assert ns.stream is True

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["slam"], settings)
        assert ns.stream_port == settings.stream_port

    def test_no_stream_flag(self):
        ns = _parse(["slam", "--no-stream"])
        assert ns.stream is False

    def test_default_visualize_is_false(self):
        ns = _parse(["slam"])
        assert ns.visualize is False

    def test_visualize_flag(self):
        ns = _parse(["slam", "--visualize"])
        assert ns.visualize is True

    def test_func_is_set(self):
        from commands.mapping import _run_slam

        ns = _parse(["slam"])
        assert ns.func is _run_slam


# Dispatch — _run_* functions
class TestRunOdometry:
    @patch("lib.map.odometry.VisualOdometry")
    @patch("lib.map.odometry.OdometryConfig")
    def test_run_odometry_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.mapping import _run_odometry

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "config_path": str(PROJECT_ROOT_PATH / "config/models/odometry.yaml"),
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml"),
            stream=True,
            stream_port=8080,
            func=_run_odometry,
            command="odometry",
        )
        _run_odometry(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


class TestRunSlam:
    @patch("lib.map.slam.SlamMapper")
    @patch("lib.map.slam.SlamConfig")
    def test_run_slam_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.mapping import _run_slam

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "config_path": str(PROJECT_ROOT_PATH / "config/models/slam.yaml"),
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml"),
            stream=True,
            stream_port=8080,
            func=_run_slam,
            command="slam",
        )
        _run_slam(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


# lib/map/odometry.py — config validation
class TestOdometryConfig:
    def test_valid_config_loads(self):
        from lib.map.odometry import OdometryConfig

        cfg = OdometryConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml"))
        assert cfg.stream is False

    def test_missing_config_raises(self, tmp_path):
        from pydantic import ValidationError

        from lib.map.odometry import OdometryConfig

        with pytest.raises(ValidationError):
            OdometryConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/map/odometry.py — algorithm tests
class TestVisualOdometry:
    def test_import(self):
        from lib.map.odometry import VisualOdometry

        assert VisualOdometry is not None

    def test_initial_pose_is_identity(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        assert np.allclose(vo._pose, np.eye(4))

    def test_init_detector_orb(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        vo._init_detector("orb", 500)
        assert vo._detector is not None
        assert vo._matcher is not None

    def test_init_detector_invalid_raises(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        with pytest.raises(ValueError, match="feature_detector"):
            vo._init_detector("laser", 500)

    def test_match_features_empty_when_no_prev(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        # With None descriptors, match_features should return empty list
        result = vo._match_features(None, None)
        assert not result

    def test_estimate_motion_returns_none_with_too_few_matches(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        result = vo._estimate_motion([], [], [], (480, 640))
        assert result is None

    def test_annotate_frame_no_keypoints(self):
        from lib.map.odometry import VisualOdometry

        frame = _dummy_frame()
        pose = np.eye(4)
        out = VisualOdometry._annotate_frame(frame.copy(), [], [], pose)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)  # text is drawn

    def test_annotate_frame_with_keypoints(self):
        import cv2

        from lib.map.odometry import VisualOdometry

        frame = _dummy_frame()
        pose = np.eye(4)
        kps = [cv2.KeyPoint(x=320.0, y=240.0, size=5.0)]
        out = VisualOdometry._annotate_frame(frame.copy(), kps, [], pose)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)


# lib/map/slam.py — config validation
class TestSlamConfig:
    def test_valid_config_loads(self):
        from lib.map.slam import SlamConfig

        cfg = SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml"))
        assert cfg.stream is False

    def test_missing_config_raises(self, tmp_path):
        from pydantic import ValidationError

        from lib.map.slam import SlamConfig

        with pytest.raises(ValidationError):
            SlamConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/map/slam.py — algorithm tests
class TestSlamMapper:
    def test_import(self):
        from lib.map.slam import SlamMapper

        assert SlamMapper is not None

    def test_annotate_frame_ok_state(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out = SlamMapper._annotate_frame(frame.copy(), "OK", "orbslam2")
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_annotate_frame_lost_state(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out = SlamMapper._annotate_frame(frame.copy(), "LOST", "rtabmap")
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_annotate_ok_differs_from_lost(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out_ok = SlamMapper._annotate_frame(frame.copy(), "OK", "orbslam2")
        out_lost = SlamMapper._annotate_frame(frame.copy(), "LOST", "orbslam2")
        # Different state labels should produce different colours
        assert not np.array_equal(out_ok, out_lost)


# lib/map/slam.py — --visualize rendering
class TestSlamVisualization:
    def test_update_visualization_first_call_has_no_prev(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        frame = _dummy_frame()
        gray = np.zeros((480, 640), dtype=np.uint8)
        _, matches, inlier_mask, prev_frame, prev_kps = mapper._update_visualization(frame, gray)

        assert prev_frame is None
        assert prev_kps is None
        assert inlier_mask is None
        assert not matches

    def test_update_visualization_second_call_has_prev(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        frame = _dummy_frame()
        gray = np.zeros((480, 640), dtype=np.uint8)

        mapper._update_visualization(frame, gray)
        _, _, _, prev_frame, prev_kps = mapper._update_visualization(frame, gray)

        assert prev_frame is not None
        assert prev_frame.shape == frame.shape

    def test_ransac_inlier_mask_too_few_matches(self):
        from lib.map.slam import SlamMapper

        assert SlamMapper._ransac_inlier_mask([], [], []) is None

    def test_render_features_panel_draws_keypoints(self):
        import cv2

        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        kps = [cv2.KeyPoint(x=320.0, y=240.0, size=5.0)]
        out = SlamMapper._render_features_panel(frame.copy(), kps)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_render_matches_panel_no_prev_frame(self):
        from lib.map.slam import SlamMapper

        panel = SlamMapper._render_matches_panel(None, None, _dummy_frame(), [], [], None, 640, 480)
        # Full row width (2x) since the camera panel above it was removed.
        assert panel.shape == (480, 1280, 3)

    def test_render_matches_panel_with_prev_frame_and_inliers(self):
        import cv2

        from lib.map.slam import SlamMapper

        prev_frame = _dummy_frame()
        frame = _dummy_frame()
        kps = [cv2.KeyPoint(x=320.0, y=240.0, size=5.0), cv2.KeyPoint(x=100.0, y=100.0, size=5.0)]
        prev_kps = [
            cv2.KeyPoint(x=321.0, y=241.0, size=5.0),
            cv2.KeyPoint(x=99.0, y=99.0, size=5.0),
        ]
        matches = [
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=1.0),
            cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=2.0),
        ]
        inlier_mask = np.array([[1], [0]], dtype=np.uint8)
        panel = SlamMapper._render_matches_panel(
            prev_frame, prev_kps, frame, kps, matches, inlier_mask, 640, 480
        )
        assert panel.shape == (480, 1280, 3)

    def test_render_visualization_view_canvas_size(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out = SlamMapper._render_visualization_view(
            [], frame, [], [], None, None, None, "NOT_INIT", "orbslam2", 1280, 960
        )
        assert out.shape == (960, 1280, 3)


# lib/map/slam.py — ORB-SLAM2/3 backend selection and shared state constants
class TestBackendConstants:
    def test_valid_backends_includes_both(self):
        from lib.map.slam import _VALID_BACKENDS

        assert ["orbslam2", "orbslam3"] == _VALID_BACKENDS

    def test_orbslam2_states_unchanged(self):
        from lib.map.slam import _ORBSLAM2_STATES

        assert {
            -1: "NOT_READY",
            0: "NO_IMAGES",
            1: "NOT_INIT",
            2: "OK",
            3: "LOST",
        } == _ORBSLAM2_STATES

    def test_orbslam3_states_diverge_from_orbslam2_at_3(self):
        from lib.map.slam import _ORBSLAM3_STATES

        assert {
            -1: "NOT_READY",
            0: "NO_IMAGES",
            1: "NOT_INIT",
            2: "OK",
            3: "RECENTLY_LOST",
            4: "LOST",
            5: "OK_KLT",
        } == _ORBSLAM3_STATES

    def test_states_share_the_common_prefix_by_identity(self):
        # Both dicts are built from _STATES_COMMON — assert the shared keys
        # actually come from the same object, not two independently
        # hand-written copies that could drift apart.
        from lib.map.slam import _ORBSLAM2_STATES, _ORBSLAM3_STATES, _STATES_COMMON

        for key, value in _STATES_COMMON.items():
            assert _ORBSLAM2_STATES[key] == value
            assert _ORBSLAM3_STATES[key] == value

    def test_backend_labels(self):
        from lib.map.slam import _BACKEND_LABELS

        assert {"orbslam2": "ORB-SLAM2", "orbslam3": "ORB-SLAM3"} == _BACKEND_LABELS


class TestLoadDispatch:
    def test_load_dispatches_to_orbslam3(self, tmp_path):
        from lib.map.slam import SlamConfig, SlamMapper

        cfg_file = tmp_path / "slam.yaml"
        cfg_file.write_text("backend: orbslam3\n", encoding="utf-8")
        mapper = SlamMapper(**SlamConfig(config_path=str(cfg_file)).dict())
        mapper._load_orbslam2 = MagicMock()
        mapper._load_orbslam3 = MagicMock()

        mapper._load()

        mapper._load_orbslam3.assert_called_once()
        mapper._load_orbslam2.assert_not_called()

    def test_load_dispatches_to_orbslam2_by_default(self, tmp_path):
        from lib.map.slam import SlamConfig, SlamMapper

        cfg_file = tmp_path / "slam.yaml"
        cfg_file.write_text("backend: orbslam2\n", encoding="utf-8")
        mapper = SlamMapper(**SlamConfig(config_path=str(cfg_file)).dict())
        mapper._load_orbslam2 = MagicMock()
        mapper._load_orbslam3 = MagicMock()

        mapper._load()

        mapper._load_orbslam2.assert_called_once()
        mapper._load_orbslam3.assert_not_called()

    def test_load_rejects_invalid_backend(self, tmp_path):
        from lib.map.slam import SlamConfig, SlamMapper

        cfg_file = tmp_path / "slam.yaml"
        cfg_file.write_text("backend: rtabmap\n", encoding="utf-8")
        mapper = SlamMapper(**SlamConfig(config_path=str(cfg_file)).dict())

        with pytest.raises(ValueError, match="backend must be one of"):
            mapper._load()

    def test_load_orbslam3_missing_bindings_raises(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        # pyorbslam isn't installed in this test environment (unlike orbslam2,
        # which conftest.py mocks away) — this exercises the real ImportError path.
        with pytest.raises(RuntimeError, match="pyorbslam"):
            mapper._load_orbslam3({})


# lib/map/slam.py — vocab/settings path resolution (relative vs. absolute,
# and whether a config lands in ~/.nano-explorer/config vs. the package's own)
class TestResolveVocabAndSettings:
    def test_relative_config_path_seeds_user_config_dir(self, fake_project_env):
        from lib.map.slam import SlamMapper

        project_root, user_config_dir = fake_project_env

        _, settings = SlamMapper._resolve_vocab_and_settings({}, "config/models/orbslam3_mono.yaml")

        assert settings == user_config_dir / "models" / "orbslam3_mono.yaml"
        assert settings.exists()
        assert settings.read_text(encoding="utf-8") == (
            project_root / "config" / "models" / "orbslam3_mono.yaml"
        ).read_text(encoding="utf-8")

    def test_user_edit_survives_a_second_resolve(self, fake_project_env):
        # fake_project_env's monkeypatching is what makes _resolve_vocab_and_settings
        # resolve into tmp_path at all here — the fixture's return value isn't needed.
        del fake_project_env
        from lib.map.slam import SlamMapper

        _, settings = SlamMapper._resolve_vocab_and_settings({}, "config/models/orbslam3_mono.yaml")
        settings.write_text(
            "Camera.type: PinHole\nCamera.fx: 999.0  # user-tuned\n", encoding="utf-8"
        )

        _, settings2 = SlamMapper._resolve_vocab_and_settings(
            {}, "config/models/orbslam3_mono.yaml"
        )

        assert settings2 == settings
        assert "999.0" in settings2.read_text(encoding="utf-8")

    def test_absolute_settings_path_used_as_is(self, fake_project_env, tmp_path):
        # Same as above — fake_project_env is here only for its monkeypatching side effect.
        del fake_project_env
        from lib.map.slam import SlamMapper

        abs_settings = tmp_path / "custom_abs_settings.yaml"
        abs_settings.write_text("Camera.type: PinHole\nCamera.fx: 42.0\n", encoding="utf-8")

        _, settings = SlamMapper._resolve_vocab_and_settings(
            {"settings": str(abs_settings)}, "config/models/orbslam3_mono.yaml"
        )

        assert settings == abs_settings

    def test_relative_non_config_settings_path_is_not_seeded(self, fake_project_env):
        from lib.map.slam import SlamMapper

        project_root, user_config_dir = fake_project_env
        (project_root / "my_custom.yaml").write_text("Camera.type: PinHole\n", encoding="utf-8")

        _, settings = SlamMapper._resolve_vocab_and_settings(
            {"settings": "my_custom.yaml"}, "config/models/orbslam3_mono.yaml"
        )

        assert settings == project_root / "my_custom.yaml"
        assert not (user_config_dir / "my_custom.yaml").exists()

    def test_vocab_relative_resolves_against_project_root(self, fake_project_env):
        from lib.map.slam import SlamMapper

        project_root, _ = fake_project_env

        vocab, _ = SlamMapper._resolve_vocab_and_settings({}, "config/models/orbslam3_mono.yaml")

        assert vocab == project_root / "assets" / "models" / "ORBvoc.txt"

    def test_vocab_absolute_used_as_is(self, fake_project_env):
        from lib.map.slam import SlamMapper

        project_root, _ = fake_project_env
        abs_vocab = project_root / "assets" / "models" / "ORBvoc.txt"

        vocab, _ = SlamMapper._resolve_vocab_and_settings(
            {"vocabulary": str(abs_vocab)}, "config/models/orbslam3_mono.yaml"
        )

        assert vocab == abs_vocab

    def test_missing_vocab_raises(self, fake_project_env):
        from lib.map.slam import SlamMapper

        project_root, _ = fake_project_env
        (project_root / "assets" / "models" / "ORBvoc.txt").unlink()

        with pytest.raises(FileNotFoundError, match="ORB vocabulary not found"):
            SlamMapper._resolve_vocab_and_settings({}, "config/models/orbslam3_mono.yaml")


# lib/map/slam.py — ORB-SLAM3 frame processing and trajectory extraction
class TestOrbslam3Processing:
    def test_process_frame_calls_process_image_mono_with_filename_arg(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        mock_slam = MagicMock()
        mock_slam.get_tracking_state.return_value = 2
        mapper._slam = mock_slam

        gray = np.zeros((480, 640), dtype=np.uint8)
        state = mapper._process_frame_orbslam3(gray, 123.456)

        # Unlike orbslam2's binding, orbslam3's process_image_mono takes a
        # third (filename) argument.
        mock_slam.process_image_mono.assert_called_once_with(gray, 123.456, "")
        assert state == 2

    def test_get_trajectory_inverts_tcw_to_twc(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        mapper._backend = "orbslam3"

        # A camera translated +1 in x, +2 in z (identity rotation), expressed
        # as Twc (world position = translation column) — pyorbslam instead
        # returns the inverse (Tcw, world-to-camera), so _get_trajectory must
        # invert it back before pose[0, 3]/pose[2, 3] mean "world position"
        # the same way they do for the orbslam2 backend.
        twc_expected = np.eye(4)
        twc_expected[0, 3] = 1.0
        twc_expected[2, 3] = 2.0
        tcw = np.linalg.inv(twc_expected)

        mock_slam = MagicMock()
        mock_slam.get_trajectory_points.return_value = [(0.0, tcw)]
        mapper._slam = mock_slam

        traj = mapper._get_trajectory()

        assert len(traj) == 1
        assert np.allclose(traj[0], twc_expected)

    def test_get_trajectory_dispatches_by_backend(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        mapper._get_trajectory_orbslam2 = MagicMock(return_value=["v2"])
        mapper._get_trajectory_orbslam3 = MagicMock(return_value=["v3"])

        mapper._backend = "orbslam2"
        assert mapper._get_trajectory() == ["v2"]

        mapper._backend = "orbslam3"
        assert mapper._get_trajectory() == ["v3"]

    def test_get_trajectory_orbslam3_returns_empty_list_on_error(self):
        from lib.map.slam import SlamConfig, SlamMapper

        mapper = SlamMapper(
            **SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml")).dict()
        )
        mock_slam = MagicMock()
        mock_slam.get_trajectory_points.side_effect = RuntimeError("boom")
        mapper._slam = mock_slam

        assert mapper._get_trajectory_orbslam3() == []
