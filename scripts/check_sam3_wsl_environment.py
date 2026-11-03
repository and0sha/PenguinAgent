"""Check installed Linux smoke dependencies; optionally execute a real CUDA matmul.

No models are constructed and no weights or Hugging Face credentials are fetched.
"""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IMPORTS = ("numpy", "cv2", "torch", "torchvision", "timm", "einops", "psutil", "pycocotools.mask",
           "transformers", "sklearn", "scipy", "joblib", "PIL", "triton", "pkg_resources")
DISTRIBUTIONS = ("setuptools", "torch", "torchvision", "numpy", "opencv-python", "timm", "einops",
                 "pycocotools", "psutil", "transformers", "scikit-learn", "scipy", "joblib", "Pillow",
                 "triton", "huggingface-hub", "sentencepiece", "sam3")


def seed_pins(path):
    """Read the explicitly pinned direct dependencies, not a transitive lock."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)", line.strip())
        if match:
            pins[match[1]] = match[2]
    return pins


def check_environment(gpu_check=False, report=None):
    report = {} if report is None else report
    report["gpu_matmul_executed"] = False
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("This environment check requires Linux and Python 3.12.")
    from tools.sam3_runtime import SAM3_REVISION, resolve_sam3_runtime

    runtime = resolve_sam3_runtime()
    revision = subprocess.run(["git", "-C", str(runtime.source_root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != SAM3_REVISION:
        raise RuntimeError(f"SAM3 revision differs from the pinned {SAM3_REVISION}: {revision}")
    for package, expected in seed_pins(PROJECT_ROOT / "requirements-sam3-smoke.txt").items():
        installed = importlib.metadata.version(package)
        if installed.split("+")[0] != expected:
            raise RuntimeError(f"{package} must match the smoke seed {expected}; installed {installed}.")
    imported = {name: importlib.import_module(name) for name in IMPORTS}
    builder = importlib.import_module("sam3.model_builder")
    if not callable(getattr(builder, "build_sam3_video_predictor", None)):
        raise RuntimeError("SAM3 video builder is unavailable.")
    expected_builder = runtime.source_root / "sam3" / "model_builder.py"
    if Path(builder.__file__).resolve() != expected_builder.resolve():
        raise RuntimeError("Imported SAM3 package does not belong to the configured checkout.")
    torch = imported["torch"]
    result = {
        "status": "imports_ok", "python": sys.version, "platform": platform.platform(),
        "sam3_revision": revision, "sam3_source": str(runtime.source_root),
        "versions": {name: importlib.metadata.version(name) for name in DISTRIBUTIONS},
        "cuda_available": bool(torch.cuda.is_available()), "torch_cuda_version": torch.version.cuda,
        "gpu_matmul_executed": False,
    }
    report.update(result)
    if gpu_check:
        if not result["cuda_available"]:
            raise RuntimeError("CUDA is not available to this PyTorch installation.")
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        if capability[0] < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The SAM3 worker requires a CUDA GPU with native BF16 support.")
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        product = left @ right
        torch.cuda.synchronize(device)
        report["gpu_matmul_executed"] = True
        if not bool(torch.isfinite(product).all().item()):
            raise RuntimeError("CUDA BF16 matmul returned nonfinite values.")
        result.update({
            "status": "gpu_matmul_ok", "gpu_matmul_executed": True,
            "device_name": torch.cuda.get_device_name(device), "compute_capability": list(capability),
            "compiled_cuda_architectures": torch.cuda.get_arch_list(),
            "native_bf16_supported": True,
            "matmul_sum": float(product.float().sum().item()),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
    report.update(result)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path in the selected Linux environment.")
    parser.add_argument("--gpu-check", action="store_true", help="Run a real synchronized BF16 CUDA matrix multiplication.")
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    report = {"gpu_matmul_executed": False}
    try:
        report = check_environment(args.gpu_check, report)
        code = 0
    except Exception as exc:
        report.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        code = 1
    report["checked_at_utc"] = started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output.resolve())}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
