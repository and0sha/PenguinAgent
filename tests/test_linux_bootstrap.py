"""Bootstrap validation only: these tests never install dependencies or use CUDA."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_sam3_wsl.sh"
spec = importlib.util.spec_from_file_location("check_linux_environment", ROOT / "scripts" / "check_sam3_wsl_environment.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
BASH = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")
if BASH and not Path(BASH).is_file():
    BASH = None


class BootstrapTests(unittest.TestCase):
    @unittest.skipUnless(BASH, "Bash is unavailable")
    def test_shell_syntax_is_valid(self):
        subprocess.run([BASH, "-n", str(SCRIPT)], check=True, capture_output=True)

    @unittest.skipUnless(BASH, "Bash is unavailable")
    def test_dry_run_prints_cuda_pins_without_installing_or_running_gpu_checks(self):
        result = subprocess.run([BASH, str(SCRIPT), "--venv", "/home/test user/.venvs/penguin", "--dry-run"],
                                check=True, capture_output=True, text=True)
        self.assertIn("torch==2.10.0+cu128", result.stdout)
        self.assertIn("torchvision==0.25.0+cu128", result.stdout)
        self.assertIn("--no-build-isolation", result.stdout)
        self.assertIn("pip check", result.stdout)
        self.assertNotIn("--gpu-check", result.stdout)
        self.assertNotIn("sudo ", result.stdout)

    @unittest.skipUnless(BASH, "Bash is unavailable")
    def test_gpu_check_is_an_explicit_plan_option(self):
        result = subprocess.run([BASH, str(SCRIPT), "--venv", "/home/test/.venvs/penguin", "--dry-run", "--gpu-check"],
                                check=True, capture_output=True, text=True)
        self.assertIn("--gpu-check", result.stdout)

    @unittest.skipUnless(BASH, "Bash is unavailable")
    def test_relative_or_missing_venv_and_unknown_options_fail_without_any_setup(self):
        for arguments in ([], ["--venv", "relative"], ["--venv", "/"], ["--venv"], ["--unknown"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run([BASH, str(SCRIPT), *arguments], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Bootstrap error", result.stderr)
                self.assertNotIn("pip install", result.stdout)

    def test_seed_pins_match_the_pinned_sam3_numpy_and_classifier_contract(self):
        pins = preflight.seed_pins(ROOT / "requirements-sam3-smoke.txt")
        self.assertEqual(pins["numpy"], "1.26.4")
        self.assertEqual(pins["setuptools"], "80.9.0")
        self.assertEqual(pins["scikit-learn"], "1.8.0")
        self.assertEqual(pins["transformers"], "4.57.6")

    def test_platform_failure_is_written_as_error_not_gpu_success(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(preflight.platform, "system", return_value="Windows"), redirect_stdout(io.StringIO()):
            destination = Path(directory) / "environment.json"
            self.assertEqual(preflight.main(["--output", str(destination)]), 1)
            report = json.loads(destination.read_text())
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["gpu_matmul_executed"])

    def test_import_only_preflight_does_not_attempt_cuda_matmul(self):
        from tools import sam3_runtime

        pins = preflight.seed_pins(ROOT / "requirements-sam3-smoke.txt")
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                      version=types.SimpleNamespace(cuda="12.8"), randn=mock.Mock(side_effect=AssertionError("GPU action not requested")))
        runtime = types.SimpleNamespace(source_root=ROOT / "sam3")
        builder = types.SimpleNamespace(__file__=str(ROOT / "sam3" / "sam3" / "model_builder.py"), build_sam3_video_predictor=lambda: None)

        def imported(name):
            return torch if name == "torch" else builder if name == "sam3.model_builder" else object()

        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "version_info", (3, 12, 7)), \
                mock.patch.object(sam3_runtime, "resolve_sam3_runtime", return_value=runtime), \
                mock.patch.object(preflight.subprocess, "run", return_value=types.SimpleNamespace(stdout=sam3_runtime.SAM3_REVISION)), \
                mock.patch.object(preflight.importlib, "import_module", side_effect=imported), \
                mock.patch.object(preflight.importlib.metadata, "version", side_effect=lambda name: pins.get(name, "test")):
            report = preflight.check_environment()
        self.assertEqual(report["status"], "imports_ok")
        self.assertFalse(report["gpu_matmul_executed"])
        torch.randn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
