"""Exercise real tracking orchestration with explicit ML/video dependency stubs."""

from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.tracking_results import TrackingError


ROOT = Path(__file__).resolve().parents[1]
REAL_SUBPROCESS_RUN = subprocess.run


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class Frame:
    shape = (100, 200, 3)

    def copy(self):
        return Frame()

    def __getitem__(self, key):
        return Frame()

    def __setitem__(self, key, value):
        pass


class Mask:
    def __init__(self, x=10, y=20):
        self.x, self.y = x, y

    def astype(self, dtype):
        return self


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stdout = io.StringIO()
        redirect = redirect_stdout(self.stdout)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.mp4"
        self.source.write_bytes(b"source video")
        self.output_json = self.directory / "tracks.json"
        self.output_video = self.directory / "annotated.mp4"
        self.videos = {str(self.source): [Frame()]}
        self.ffmpeg_failure = None
        self.num_chunks = 1
        self.worker_mode = "ok"
        self.classifier_mode = "ok"
        self.write_crop = True
        self.writer_open = True
        self.empty_tracks = False
        self.fps = 1
        self.worker_detections = None
        self.sam_crop_dirs = []
        self.crop_writes = []
        self.process_targets = []
        state = self

        class Capture:
            def __init__(self, path):
                self.path = str(path)
                self.frames = state.videos.get(self.path, [])
                self.index = 0

            def isOpened(self):
                return bool(self.frames)

            def get(self, key):
                return {1: state.fps, 2: 200, 3: 100, 4: len(self.frames)}[key]

            def read(self):
                if self.index >= len(self.frames):
                    return False, None
                frame = self.frames[self.index]
                self.index += 1
                return True, frame

            def release(self):
                pass

        class Writer:
            def __init__(self, path, *args):
                self.path = str(path)
                self.frames = []

            def isOpened(self):
                return state.writer_open

            def write(self, frame):
                self.frames.append(frame)

            def release(self):
                Path(self.path).write_bytes(b"encoded video" if self.frames else b"")
                state.videos[self.path] = list(self.frames)

        def imwrite(path, image):
            state.crop_writes.append(str(path))
            if state.write_crop:
                Path(path).write_bytes(b"crop image")
            return state.write_crop

        class Predictor:
            def handle_request(self, request):
                return {"session_id": "test-session"}

            def handle_stream_request(self, request):
                detections = [{0: (10, 20)}] if state.worker_detections is None else state.worker_detections
                for index, objects in enumerate(detections):
                    yield {"frame_index": index, "outputs": {
                        "out_obj_ids": list(objects), "out_binary_masks": [Mask(*point) for point in objects.values()],
                    }}

            def shutdown(self):
                pass

        class Process:
            def __init__(self, target, args):
                self.target = target
                self.args = args
                self.exitcode = 0

            def start(self):
                state.process_targets.append(self.target.__name__)
                if self.target.__name__ == "_run_sam3_subprocess":
                    _, video, data, _, _, _, crops = self.args
                    state.sam_crop_dirs.append(crops)
                    if state.worker_mode == "crash":
                        self.exitcode = 1
                        Path(data).write_text(json.dumps({"status": "error", "error": "GPU model unavailable"}))
                        return
                    if state.worker_mode == "missing_json":
                        return
                    if state.worker_mode == "bad_json":
                        Path(data).write_text("{not JSON")
                        return
                    if state.worker_mode != "missing_video":
                        Path(video).write_bytes(b"worker video")
                        if state.worker_mode != "unreadable_video":
                            state.videos[video] = [Frame()]
                    tracks = {} if state.empty_tracks else {"obj_0": [dict(second=0, x=10, y=20, area=10., vx=0., vy=0.)]}
                    annotations = [[]] if state.empty_tracks else [[dict(obj_id=0, cx=10, cy=20)]]
                    if state.worker_mode == "annotation_mismatch":
                        annotations.append([])
                    payload = {"status": "ok", "tracks": tracks, "text_annotations": annotations,
                               "crops_saved": [] if state.empty_tracks else [0]}
                    if state.worker_mode == "missing_crop_index":
                        payload["crops_saved"] = []
                    elif state.worker_mode == "missing_status":
                        del payload["status"]
                    Path(data).write_text(json.dumps(payload))
                    if not state.empty_tracks and state.worker_mode != "missing_crop":
                        Path(crops, "penguin_crop_0.jpg").write_bytes(b"crop")
                else:
                    crop_batch, destination, _ = self.args
                    if state.classifier_mode == "crash":
                        self.exitcode = 1
                        payload = {"status": "error", "error": "SigLIP checkpoint unavailable"}
                    elif state.classifier_mode == "missing_result":
                        return
                    elif state.classifier_mode == "incomplete":
                        payload = {"status": "ok", "results": {}}
                    else:
                        label = "NOT_PENGUIN" if state.classifier_mode == "reject" else "Adelie"
                        payload = {"status": "ok", "results": {str(obj_id): label for _, obj_id in crop_batch}}
                    Path(destination).write_text(json.dumps(payload))

            def join(self, *args, **kwargs):
                pass

            def is_alive(self):
                return False

        cv2 = module("cv2", VideoCapture=Capture, VideoWriter=Writer, CAP_PROP_FPS=1,
                     CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3, CAP_PROP_FRAME_COUNT=4,
                     VideoWriter_fourcc=lambda *args: 0, putText=lambda *args: None,
                     FONT_HERSHEY_SIMPLEX=0, imwrite=imwrite, line=lambda *args: None,
                     addWeighted=lambda overlay, *args: overlay)
        config = module("tools.tracking_config", TARGET_FPS=1, TARGET_HEIGHT=720,
                        CHUNK_DURATION_SEC=10, MAX_DIST_REL_WIDTH=.1, SIGLIP_CONFIDENCE_THRESH=.1)
        runtime = module("tools.sam3_runtime", resolve_sam3_runtime=lambda: types.SimpleNamespace(
            source_root=self.directory, builder_kwargs=lambda: {}))
        numpy = module("numpy", hypot=math.hypot, where=lambda mask: ([mask.y], [mask.x]), mean=lambda values: sum(values) / len(values))
        stubs = {
            "cv2": cv2, "numpy": numpy, "scipy": module("scipy"),
            "scipy.optimize": module("scipy.optimize", linear_sum_assignment=lambda matrix: ([], [])),
            "tools.tracking_config": config, "tools.sam3_runtime": runtime,
            "torch": module("torch", inference_mode=lambda: nullcontext(), autocast=lambda *args, **kwargs: nullcontext(),
                            bfloat16="bf16", is_tensor=lambda _: False, cuda=types.SimpleNamespace(is_available=lambda: False)),
            "sam3": module("sam3"), "sam3.model_builder": module("sam3.model_builder", build_sam3_video_predictor=lambda **kwargs: Predictor()),
        }
        self.modules_patch = mock.patch.dict(sys.modules, stubs)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.utils = self.load("tools.tracking_utils")
        self.sam = self.load("tools.tracking_sam3")
        self.siglip = self.load("tools.tracking_siglip")
        self.render = self.load("tools.tracking_render")
        self.pipeline = self.load("tools.feature_tracking")
        self.run_patch = mock.patch.object(self.utils.subprocess, "run", side_effect=self.ffmpeg)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        self.process_patch = mock.patch.object(self.pipeline.multiprocessing, "get_context",
                                              return_value=types.SimpleNamespace(Process=Process))
        self.process_patch.start()
        self.addCleanup(self.process_patch.stop)

    def load(self, name):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace(".", "/") + ".py"))
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    def ffmpeg(self, command, **kwargs):
        if "-f" in command:
            stage = "segment"
        elif "-vf" in command:
            stage = "proxy"
        else:
            stage = "encode"
        if stage == self.ffmpeg_failure:
            raise subprocess.CalledProcessError(1, command, stderr="test encoder failed")
        if stage == "segment":
            for i in range(self.num_chunks):
                path = command[-1] % i
                Path(path).write_bytes(b"chunk video")
                self.videos[path] = [Frame()]
        else:
            path = str(command[-1])
            Path(path).write_bytes(b"video")
            self.videos[path] = list(self.videos.get(str(command[command.index("-i") + 1]), [Frame()]))
        return types.SimpleNamespace(returncode=0)

    def extract(self, **kwargs):
        return self.pipeline.extract_tracks(str(self.source), str(self.output_json), str(self.output_video), **kwargs)

    def keep_old_outputs(self):
        self.output_json.write_text('{"old": []}', encoding="utf-8")
        self.output_video.write_bytes(b"old video")

    def assert_old_outputs(self):
        self.assertEqual(self.output_json.read_text(), '{"old": []}')
        self.assertEqual(self.output_video.read_bytes(), b"old video")
        self.assertEqual(list(self.directory.glob(".tracking-*")), [])

    def test_success_returns_only_requested_complete_outputs_and_isolates_crops(self):
        crop_parent = self.directory / "crops"
        first = self.extract(crop_dir=crop_parent)
        self.assertEqual(first, (str(self.output_json), str(self.output_video)))
        self.assertIn("Adelie_1", json.loads(self.output_json.read_text()))
        self.assertTrue(self.output_video.stat().st_size)
        self.extract(crop_dir=crop_parent)
        self.assertEqual(len(set(self.sam_crop_dirs)), 2)
        self.assertTrue(all(Path(path).parents[1] == crop_parent for path in self.sam_crop_dirs))
        self.assertTrue(all(not Path(path).exists() for path in self.sam_crop_dirs))

    def test_zero_detections_are_a_valid_complete_result(self):
        self.empty_tracks = True
        self.extract()
        self.assertEqual(json.loads(self.output_json.read_text()), {})
        self.assertNotIn("_run_siglip_subprocess", self.process_targets)
        self.assertTrue(self.output_video.exists())

    def test_ffmpeg_errors_include_stderr_and_preserve_previous_outputs(self):
        for stage in ("proxy", "segment", "encode"):
            with self.subTest(stage=stage):
                self.keep_old_outputs()
                self.ffmpeg_failure = stage
                with self.assertRaisesRegex(TrackingError, "test encoder failed"):
                    self.extract()
                self.assert_old_outputs()

    def test_successful_segmentation_with_no_chunks_is_failure(self):
        self.keep_old_outputs()
        self.num_chunks = 0
        with self.assertRaisesRegex(TrackingError, "no chunks"):
            self.extract()
        self.assert_old_outputs()

    def test_failed_or_corrupt_workers_never_publish(self):
        for mode in ("crash", "missing_json", "bad_json", "missing_video", "unreadable_video", "annotation_mismatch", "missing_crop_index", "missing_status"):
            with self.subTest(mode=mode):
                self.keep_old_outputs()
                self.worker_mode = mode
                with self.assertRaises(TrackingError):
                    self.extract()
                self.assert_old_outputs()

    def test_worker_error_detail_reaches_the_caller(self):
        self.worker_mode = "crash"
        with self.assertRaisesRegex(TrackingError, "GPU model unavailable"):
            self.extract()

    def test_chunks_past_three_digit_padding_are_processed_in_numeric_order(self):
        original_ffmpeg = self.ffmpeg
        factory = self.pipeline.multiprocessing.get_context("spawn").Process
        processed = []

        def segment(command, **kwargs):
            if "-f" not in command:
                return original_ffmpeg(command, **kwargs)
            for index in (1000, 999):
                path = command[-1] % index
                Path(path).write_bytes(b"chunk video")
                self.videos[path] = [Frame()]
            return types.SimpleNamespace(returncode=0)

        def process(target, args):
            if target.__name__ == "_run_sam3_subprocess":
                processed.append(int(Path(args[0]).stem.rsplit("_", 1)[1]))
            return factory(target, args)

        with mock.patch.object(self.utils.subprocess, "run", side_effect=segment), \
                mock.patch.object(self.pipeline.multiprocessing, "get_context", return_value=types.SimpleNamespace(Process=process)):
            self.extract()
        self.assertEqual(processed, [999, 1000])

    def test_missing_crop_is_an_error_before_classification(self):
        self.keep_old_outputs()
        self.worker_mode = "missing_crop"
        with self.assertRaisesRegex(TrackingError, "Classification crop"):
            self.extract()
        self.assertNotIn("_run_siglip_subprocess", self.process_targets)
        self.assert_old_outputs()

    def test_classification_infrastructure_failure_is_not_a_biological_rejection(self):
        for mode in ("crash", "missing_result", "incomplete"):
            with self.subTest(mode=mode):
                self.keep_old_outputs()
                self.classifier_mode = mode
                with self.assertRaises(TrackingError):
                    self.extract()
                self.assert_old_outputs()

    def test_video_writer_failure_does_not_publish(self):
        self.keep_old_outputs()
        self.writer_open = False
        with self.assertRaisesRegex(TrackingError, "writer"):
            self.extract()
        self.assert_old_outputs()

    def test_classifier_rejection_is_still_a_valid_empty_result(self):
        self.classifier_mode = "reject"
        self.extract()
        self.assertEqual(json.loads(self.output_json.read_text()), {})
        self.assertTrue(self.output_video.exists())

    def test_compact_mapping_is_published_on_success_and_removed_on_noncompact_success(self):
        mapping = self.directory / "tracks_species_mapping.txt"
        self.extract(compact_annotation=True)
        self.assertIn("1: Adelie", mapping.read_text())
        self.assertIn("1_1", json.loads(self.output_json.read_text()))
        self.extract(compact_annotation=False)
        self.assertFalse(mapping.exists())

    def test_mapping_preparation_error_cannot_overwrite_previous_json(self):
        self.keep_old_outputs()
        original_copy = self.pipeline.shutil.copyfile

        def fail_mapping(source, destination):
            if str(source).endswith("_species_mapping.txt"):
                raise OSError("mapping destination unavailable")
            return original_copy(source, destination)

        with mock.patch.object(self.pipeline.shutil, "copyfile", side_effect=fail_mapping):
            with self.assertRaisesRegex(TrackingError, "mapping destination unavailable"):
                self.extract(compact_annotation=True)
        self.assert_old_outputs()

    def test_final_encoder_cannot_silently_drop_frames(self):
        self.videos[str(self.source)] = [Frame(), Frame()]
        original_ffmpeg = self.ffmpeg

        def truncate_output(command, **kwargs):
            result = original_ffmpeg(command, **kwargs)
            self.videos[str(command[-1])] = [Frame()]
            return result

        with mock.patch.object(self.utils.subprocess, "run", side_effect=truncate_output):
            with self.assertRaisesRegex(TrackingError, "frame count differs"):
                self.render.render_final_video([str(self.source)], [[], []], {},
                                               str(self.directory / "raw.mp4"), str(self.output_video), 1)

    def test_sam_crop_write_failure_has_explicit_error_status_and_local_path(self):
        self.write_crop = False
        crop_dir = self.directory / "worker-crops"
        data = self.directory / "worker.json"
        with self.assertRaisesRegex(TrackingError, "save classification crop"):
            self.sam._run_sam3_subprocess(str(self.source), str(self.directory / "worker.mp4"), str(data), crop_dir=crop_dir)
        self.assertEqual(json.loads(data.read_text())["status"], "error")
        self.assertEqual(self.crop_writes, [str(crop_dir / "penguin_crop_0.jpg")])

    def test_sam_success_reports_status_and_writes_crop_in_requested_directory(self):
        crop_dir = self.directory / "worker-crops"
        data = self.directory / "worker.json"
        self.sam._run_sam3_subprocess(str(self.source), str(self.directory / "worker.mp4"), str(data), crop_dir=crop_dir)
        payload = json.loads(data.read_text())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["crops_saved"], [0])
        self.assertTrue((crop_dir / "penguin_crop_0.jpg").is_file())

    def test_sam_partial_input_decode_cannot_be_reported_as_success(self):
        cv2 = sys.modules["cv2"]
        original_capture = cv2.VideoCapture

        def truncated_capture(path):
            capture = original_capture(path)
            original_get = capture.get
            capture.get = lambda key: 2 if key == cv2.CAP_PROP_FRAME_COUNT else original_get(key)
            return capture

        data = self.directory / "worker.json"
        with mock.patch.object(cv2, "VideoCapture", side_effect=truncated_capture):
            with self.assertRaisesRegex(TrackingError, "decoded 1 input frames; expected 2"):
                self.sam._run_sam3_subprocess(str(self.source), str(self.directory / "worker.mp4"), str(data),
                                             crop_dir=self.directory / "worker-crops")
        self.assertEqual(json.loads(data.read_text())["status"], "error")

    def test_sam_keeps_raw_ids_and_every_subsecond_initial_observation(self):
        self.fps = 2.5
        self.worker_detections = [
            {7: (10, 20)},
            {7: (11, 20), 0: (50, 20)},
            {7: (12, 20), 0: (51, 20)},
            {7: (13, 20), 0: (52, 20)},
        ]
        self.videos[str(self.source)] = [Frame() for _ in self.worker_detections]
        data = self.directory / "worker.json"
        crop_dir = self.directory / "worker-crops"
        self.sam._run_sam3_subprocess(
            str(self.source), str(self.directory / "worker.mp4"), str(data),
            next_obj_id=900, prev_points={"obj_99": (10, 20)}, prev_trajectories={"obj_99": [(9, 20)]},
            crop_dir=crop_dir,
        )
        payload = json.loads(data.read_text())
        self.assertEqual(set(payload["tracks"]), {"obj_7", "obj_0"})
        self.assertEqual([point["second"] for point in payload["tracks"]["obj_7"]], [0, .4, .8, 1.2])
        self.assertEqual([point["second"] for point in payload["tracks"]["obj_0"]], [.4, .8, 1.2])
        self.assertEqual([ann["obj_id"] for ann in payload["text_annotations"][1]], [7, 0])
        self.assertEqual(set(payload["crops_saved"]), {0, 7})
        self.assertTrue((crop_dir / "penguin_crop_7.jpg").is_file())
        self.assertFalse((crop_dir / "penguin_crop_99.jpg").exists())

    def test_sam_reused_raw_id_in_another_chunk_does_not_receive_legacy_global_id(self):
        self.worker_detections = [{7: (10, 20)}]
        for index, position in enumerate((10, 150)):
            self.worker_detections = [{7: (position, 20)}]
            data = self.directory / f"worker-{index}.json"
            self.sam._run_sam3_subprocess(
                str(self.source), str(self.directory / f"worker-{index}.mp4"), str(data),
                next_obj_id=100 + index, prev_points={"obj_99": (position, 20)},
                crop_dir=self.directory / f"chunk-{index:03d}",
            )
            self.assertEqual(set(json.loads(data.read_text())["tracks"]), {"obj_7"})
        self.assertEqual(len(set(self.crop_writes)), 2)

    def test_worker_passes_explicit_config_without_changing_environment(self):
        runtime_module = sys.modules["tools.sam3_runtime"]
        runtime = runtime_module.resolve_sam3_runtime()
        config_path = str(self.directory / "custom.env")
        with mock.patch.object(runtime_module, "resolve_sam3_runtime", return_value=runtime) as resolver:
            self.sam._run_sam3_subprocess(str(self.source), str(self.directory / "worker.mp4"),
                                         str(self.directory / "worker.json"), crop_dir=self.directory / "crops",
                                         sam3_config_path=config_path)
        resolver.assert_called_once_with(config_path)

    def test_parent_propagates_config_only_as_optional_eighth_worker_argument(self):
        factory = self.pipeline.multiprocessing.get_context("spawn").Process
        captured = []

        def process(target, args):
            captured.append(args)
            return factory(target, args[:7])

        config_path = str(self.directory / "custom.env")
        with mock.patch.object(self.pipeline.multiprocessing, "get_context", return_value=types.SimpleNamespace(Process=process)):
            self.extract(sam3_config_path=config_path)
        sam_args = next(args for args in captured if len(args) == 8)
        self.assertEqual(sam_args[-1], config_path)

    def test_visual_trails_have_linear_bounded_cost_and_do_not_flush_cuda_each_frame(self):
        frames = 100
        self.worker_detections = [{7: (index, 20)} for index in range(frames)]
        self.videos[str(self.source)] = [Frame() for _ in range(frames)]
        cuda = sys.modules["torch"].cuda
        cuda.empty_cache = mock.Mock()
        cuda.is_available = lambda: True
        with mock.patch.object(sys.modules["cv2"], "line") as line:
            self.sam._run_sam3_subprocess(str(self.source), str(self.directory / "worker.mp4"),
                                         str(self.directory / "worker.json"), crop_dir=self.directory / "crops")
        cuda.empty_cache.assert_not_called()
        self.assertLessEqual(line.call_count, frames * 29)
        payload = json.loads((self.directory / "worker.json").read_text())
        self.assertEqual(len(payload["tracks"]["obj_7"]), frames)

    def test_siglip_missing_crop_fails_before_model_loading(self):
        data = self.directory / "classification.json"
        with self.assertRaises(TrackingError):
            self.siglip._run_siglip_subprocess([(str(self.directory / "missing.jpg"), 0)], str(data))
        self.assertEqual(json.loads(data.read_text())["status"], "error")

    def test_siglip_model_error_is_serialized_and_raised(self):
        crop = self.directory / "crop.jpg"
        crop.write_bytes(b"crop")
        data = self.directory / "classification.json"
        with mock.patch.object(self.siglip, "_classify_siglip_crops", side_effect=RuntimeError("model download unavailable")):
            with self.assertRaisesRegex(TrackingError, "model download unavailable"):
                self.siglip._run_siglip_subprocess([(str(crop), 0)], str(data))
        payload = json.loads(data.read_text())
        self.assertEqual(payload["status"], "error")
        self.assertNotIn("results", payload)

    def test_missing_ffmpeg_executable_is_an_actionable_failure(self):
        with mock.patch.object(self.utils.subprocess, "run", side_effect=FileNotFoundError("ffmpeg missing")):
            with self.assertRaisesRegex(TrackingError, "could not start FFmpeg"):
                self.utils.run_ffmpeg(["ffmpeg"], "Test conversion")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_real_ffmpeg_failure_returns_diagnostics_without_ml_dependencies(self):
        with mock.patch.object(self.utils.subprocess, "run", REAL_SUBPROCESS_RUN):
            with self.assertRaisesRegex(TrackingError, "Real FFmpeg probe failed.*FFmpeg exit"):
                self.utils.run_ffmpeg(["ffmpeg", "-hide_banner", "-i", str(self.directory / "missing.mp4")], "Real FFmpeg probe")


if __name__ == "__main__":
    unittest.main()
