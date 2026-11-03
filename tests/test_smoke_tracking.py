"""Lightweight smoke CLI/configuration checks; no models, Torch or GPU needed."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = ("PENGUIN_TRACKING_FPS", "PENGUIN_TRACKING_HEIGHT", "PENGUIN_TRACKING_CHUNK_SECONDS")
spec = importlib.util.spec_from_file_location("smoke_tracking", ROOT / "scripts" / "smoke_tracking.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def read_spawned_config(path):
    from tools.tracking_config import TARGET_FPS, TARGET_HEIGHT, CHUNK_DURATION_SEC
    Path(path).write_text(json.dumps([TARGET_FPS, TARGET_HEIGHT, CHUNK_DURATION_SEC]))


class SmokeTests(unittest.TestCase):
    def args(self, *extra):
        return smoke.build_parser().parse_args(["--video", "video.mp4", "--output-dir", "smoke-results", *extra])

    def test_defaults_cover_multiple_short_chunks(self):
        args = self.args()
        self.assertEqual((args.seconds, args.chunk_seconds, args.fps, args.height, args.stage), (12, 4, 1, 480, "pipeline"))
        self.assertGreater(args.seconds, args.chunk_seconds)

    def test_fractional_settings_and_sam3_only_are_supported(self):
        args = self.args("--fps", "2.5", "--chunk-seconds", "4.2", "--start", "1.5", "--stage", "sam3")
        self.assertEqual((args.fps, args.chunk_seconds, args.start, args.stage), (2.5, 4.2, 1.5, "sam3"))

    def test_invalid_arguments_fail_before_inference(self):
        for key, value in (("--seconds", "0"), ("--fps", "nan"), ("--fps", "inf"),
                           ("--start", "-1"), ("--height", "4.5"), ("--height", "0"), ("--chunk-seconds", "-2")):
            with self.subTest(key=key, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(key, value)

    def test_summary_accepts_simultaneous_animals_but_unique_times_per_track(self):
        tracks = {"a": [dict(second=0, x=0, y=0, vx=0, vy=0), dict(second=.4, x=2, y=0, vx=5, vy=0)],
                  "b": [dict(second=0, x=5, y=0, vx=0, vy=0)]}
        summary = smoke.summarize_tracks(tracks)
        self.assertEqual((summary["track_count"], summary["observation_count"], summary["maximum_speed"]), (2, 3, 5))
        self.assertEqual(summary["last_timestamp_seconds"], .4)

    def test_duplicate_times_nonfinite_and_missing_velocities_are_rejected(self):
        from tools.tracking_results import TrackingError
        for observations in ([dict(second=0, x=0, y=0, vx=0, vy=0)] * 2,
                             [dict(second=0, x=0, y=0, vx=float("nan"), vy=0)],
                             [dict(second=0, x=0, y=0)]):
            with self.subTest(observations=observations), self.assertRaises((ValueError, TrackingError)):
                smoke.summarize_tracks({"a": observations})

    def test_empty_result_is_inconclusive_not_success(self):
        status, code, reasons = smoke.outcome(smoke.summarize_tracks({}), [{"status": "ok", "gpu": {"cuda_available": True, "peak_allocated_bytes": 10}}])
        self.assertEqual((status, code), ("inconclusive", 2))
        self.assertTrue(reasons)

    def test_tracks_without_measured_cuda_are_inconclusive(self):
        stats = {"track_count": 1}
        self.assertEqual(smoke.outcome(stats, [])[0:2], ("inconclusive", 2))
        worker = {"status": "ok", "gpu": {"cuda_available": True, "peak_allocated_bytes": 10}}
        self.assertEqual(smoke.outcome(stats, [worker])[0:2], ("ok", 0))

    def test_malformed_cuda_metrics_never_count_as_gpu_proof(self):
        for worker in ({"status": "ok", "gpu": None},
                       {"status": "ok", "gpu": {"cuda_available": True, "peak_allocated_bytes": "100"}},
                       {"status": "ok", "gpu": {"cuda_available": True, "peak_allocated_bytes": True}}):
            with self.subTest(worker=worker):
                self.assertEqual(smoke.outcome({"track_count": 1}, [worker])[0:2], ("inconclusive", 2))

    def test_corrupt_worker_report_is_retained_as_error_instead_of_masking_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "gpu_metrics"
            metrics.mkdir()
            (metrics / "bad.json").write_text('{"status": "ok", "elapsed": NaN}')
            (metrics / "truncated.json").write_text("{")
            reports = smoke.read_worker_reports(directory)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report["status"] == "error" for report in reports))
        self.assertEqual(smoke.outcome({"track_count": 1}, reports)[0:2], ("inconclusive", 2))

    def test_environment_settings_survive_a_real_spawn(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}):
            smoke.configure_environment(self.args("--fps", "2.5", "--height", "320", "--chunk-seconds", "4.2"))
            destination = Path(directory) / "config.json"
            process = multiprocessing.get_context("spawn").Process(target=read_spawned_config, args=(str(destination),))
            process.start()
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("Spawned configuration probe did not finish.")
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(json.loads(destination.read_text()), [2.5, 320, 4.2])

    def test_configuration_rejects_bad_environment_values(self):
        for key, value in ((ENV_KEYS[0], "nan"), (ENV_KEYS[1], "2.5"), (ENV_KEYS[2], "0")):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: value}), mock.patch.dict(sys.modules, {"transformers": None}):
                config_spec = importlib.util.spec_from_file_location("smoke_bad_config", ROOT / "tools" / "tracking_config.py")
                config = importlib.util.module_from_spec(config_spec)
                with self.assertRaisesRegex(ValueError, key):
                    config_spec.loader.exec_module(config)

    def test_proxy_gop_is_an_integer_and_never_zero_with_fractional_settings(self):
        config = types.ModuleType("tools.tracking_config")
        config.TARGET_FPS, config.TARGET_HEIGHT, config.CHUNK_DURATION_SEC = 2.5, 480, 4.2
        with mock.patch.dict(sys.modules, {"tools.tracking_config": config}):
            utils_spec = importlib.util.spec_from_file_location("smoke_proxy_utils", ROOT / "tools" / "tracking_utils.py")
            utils = importlib.util.module_from_spec(utils_spec)
            utils_spec.loader.exec_module(utils)
        for fps in (2.5, .01):
            with self.subTest(fps=fps), mock.patch.object(utils, "run_ffmpeg") as run, mock.patch.object(utils, "validate_video"), redirect_stdout(io.StringIO()):
                utils.create_lightweight_proxy_video("source", "destination", target_fps=fps)
                command = run.call_args.args[0]
                gop = command[command.index("-g") + 1]
                self.assertTrue(gop.isdigit())
                self.assertGreaterEqual(int(gop), 1)


if __name__ == "__main__":
    unittest.main()
