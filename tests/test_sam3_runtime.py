"""SAM3 path/configuration tests; no ML packages, weights, GPU or network required."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.sam3_runtime import Sam3ConfigurationError, resolve_sam3_runtime


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class Sam3RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()
        self.config = self.project / "config.env"
        self.config.write_text("SAM3_ROOT=./sam3\nSAM3_CHECKPOINT=\n", encoding="utf-8")

    def make_source(self, name="sam3"):
        source = self.project / name
        package = source / "sam3"
        (package / "assets").mkdir(parents=True)
        (package / "model_builder.py").write_text("# fake source\n", encoding="utf-8")
        (package / "assets" / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"fake vocabulary")
        return source.resolve()

    def resolve(self, **kwargs):
        kwargs.setdefault("environ", {})
        return resolve_sam3_runtime(project_root=self.project, **kwargs)

    def test_default_root_works_from_other_working_directory(self):
        source = self.make_source()
        elsewhere = Path(self.temp.name) / "elsewhere"
        elsewhere.mkdir()
        with working_directory(elsewhere):
            runtime = self.resolve()
        self.assertEqual(runtime.source_root, source)
        self.assertEqual(runtime.bpe_path, source / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        self.assertIsNone(runtime.checkpoint_path)
        self.assertNotIn("checkpoint_path", runtime.builder_kwargs())

    def test_environment_overrides_config_and_checkpoint_is_forwarded(self):
        source = self.make_source("external sam3")
        checkpoint = self.project / "weights" / "sam3.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"fake checkpoint")
        runtime = self.resolve(environ={"SAM3_ROOT": "external sam3", "SAM3_CHECKPOINT": "weights/sam3.pt"})
        self.assertEqual(runtime.source_root, source)
        self.assertEqual(runtime.checkpoint_path, checkpoint.resolve())
        self.assertEqual(runtime.builder_kwargs(), {
            "bpe_path": str(runtime.bpe_path),
            "async_loading_frames": False,
            "checkpoint_path": str(checkpoint.resolve()),
        })

    def test_absolute_external_root_and_quoted_config_paths(self):
        source = self.make_source("external source")
        checkpoint = self.project / "sam3 weights.pt"
        checkpoint.write_bytes(b"fake checkpoint")
        self.config.write_text(
            f'export SAM3_ROOT="{source.as_posix()}" # checkout\nSAM3_CHECKPOINT=\'sam3 weights.pt\'\nOLLAMA_MODEL=unrelated\n',
            encoding="utf-8",
        )
        runtime = self.resolve()
        self.assertEqual(runtime.source_root, source)
        self.assertEqual(runtime.checkpoint_path, checkpoint.resolve())

    def test_empty_environment_checkpoint_retains_automatic_loading(self):
        self.make_source()
        self.config.write_text("SAM3_ROOT=./sam3\nSAM3_CHECKPOINT=absent.pt\n", encoding="utf-8")
        runtime = self.resolve(environ={"SAM3_CHECKPOINT": ""})
        self.assertIsNone(runtime.checkpoint_path)

    def test_missing_source_error_explains_submodule_recovery(self):
        with self.assertRaisesRegex(Sam3ConfigurationError, "git submodule update --init -- sam3"):
            self.resolve()

    def test_missing_tokenizer_is_reported(self):
        source = self.make_source()
        (source / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz").unlink()
        with self.assertRaisesRegex(Sam3ConfigurationError, "bpe_simple_vocab"):
            self.resolve()

    def test_root_must_contain_the_sam3_package(self):
        (self.project / "sam3").write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(Sam3ConfigurationError, "complete SAM3 checkout"):
            self.resolve()

    def test_missing_config_is_actionable(self):
        with self.assertRaisesRegex(Sam3ConfigurationError, "Cannot read SAM3 configuration"):
            self.resolve(config_path="missing.env")

    def test_malformed_config_is_rejected(self):
        for line in ["SAM3_ROOT", 'SAM3_ROOT="unterminated', 'SAM3_ROOT="./sam3" unexpected']:
            with self.subTest(line=line):
                self.config.write_text(line, encoding="utf-8")
                with self.assertRaisesRegex(Sam3ConfigurationError, "Invalid SAM3_ROOT.*line 1"):
                    self.resolve()

    def test_empty_source_root_is_rejected(self):
        with self.assertRaisesRegex(Sam3ConfigurationError, "SAM3_ROOT is empty"):
            self.resolve(environ={"SAM3_ROOT": " "})

    def test_configured_checkpoint_must_exist(self):
        self.make_source()
        with self.assertRaisesRegex(Sam3ConfigurationError, "SAM3_CHECKPOINT file does not exist"):
            self.resolve(environ={"SAM3_CHECKPOINT": "weights/missing.pt"})

    def test_checkpoint_directory_is_not_a_file(self):
        self.make_source()
        (self.project / "weights").mkdir()
        with self.assertRaisesRegex(Sam3ConfigurationError, "SAM3_CHECKPOINT file does not exist"):
            self.resolve(environ={"SAM3_CHECKPOINT": "weights"})

    def test_config_path_is_project_relative_not_cwd_relative(self):
        source = self.make_source()
        custom = self.project / "config" / "tracking.env"
        custom.parent.mkdir()
        custom.write_text("SAM3_ROOT=./sam3\n", encoding="utf-8")
        with working_directory(self.temp.name):
            runtime = self.resolve(config_path="config/tracking.env")
        self.assertEqual(runtime.source_root, source)

    def test_worker_validates_configuration_before_importing_ml_dependencies(self):
        from tools.tracking_sam3 import _run_sam3_subprocess
        from tools.tracking_results import TrackingError

        error = Sam3ConfigurationError("missing checkout before ML imports")
        with mock.patch("tools.sam3_runtime.resolve_sam3_runtime", side_effect=error):
            with self.assertRaisesRegex(TrackingError, "before ML imports"):
                _run_sam3_subprocess("unused.mp4", str(self.project / "unused-output.mp4"),
                                     str(self.project / "unused-data.json"), crop_dir=self.project / "crops")

    def test_single_quoted_backslashes_are_preserved_in_config(self):
        source = self.make_source(r"source\test")
        self.config.write_text("SAM3_ROOT='source\\test'\n", encoding="utf-8")
        self.assertEqual(self.resolve().source_root, source)

    def test_double_quoted_backslash_escapes_get_actionable_path_error(self):
        self.config.write_text('SAM3_ROOT="source\\test"\n', encoding="utf-8")
        with self.assertRaisesRegex(Sam3ConfigurationError, "control character.*forward slashes"):
            self.resolve()

    def test_control_characters_from_dotenv_environment_are_also_rejected(self):
        with self.assertRaisesRegex(Sam3ConfigurationError, "control character"):
            self.resolve(environ={"SAM3_ROOT": "source\test"})

    def test_sam3_paths_require_literal_values(self):
        self.config.write_text("SAM3_ROOT=${SAM3_HOME}/sam3\n", encoding="utf-8")
        with self.assertRaisesRegex(Sam3ConfigurationError, "variable interpolation"):
            self.resolve()


if __name__ == "__main__":
    unittest.main()
