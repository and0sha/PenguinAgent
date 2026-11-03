"""Run real SAM3 or tracking inference on a short private clip, without an LLM/RAG.

Examples (from the project directory, in the prepared GPU environment):
    python -B scripts/smoke_tracking.py --video videos/1.mp4 --output-dir outputs/smoke --stage sam3
    python -B scripts/smoke_tracking.py --video videos/1.mp4 --output-dir outputs/smoke --stage pipeline

Exit status: 0 = observed tracks and measured CUDA inference; 2 = inconclusive
(e.g. no detections); 1 = failure. Each attempt keeps its artifacts and summary.json
in a new directory. This script does not install packages or obtain model access.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def positive_number(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def nonnegative_number(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def positive_integer(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="Local source video containing visible penguins.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent directory for a new isolated smoke run.")
    parser.add_argument("--seconds", type=positive_number, default=12.0, help="Maximum clip length (default: 12 seconds).")
    parser.add_argument("--start", type=nonnegative_number, default=0.0, help="Source start time in seconds (default: 0).")
    parser.add_argument("--chunk-seconds", type=positive_number, default=4.0, help="Pipeline chunk duration (default: 4 seconds).")
    parser.add_argument("--fps", type=positive_number, default=1.0, help="SAM proxy frames per second (default: 1).")
    parser.add_argument("--height", type=positive_integer, default=480, help="Maximum proxy height (default: 480).")
    parser.add_argument("--stage", choices=("pipeline", "sam3"), default="pipeline",
                        help="pipeline includes real SigLIP classification; sam3 tests only real SAM3 (default: pipeline).")
    return parser


def configure_environment(args):
    """Set inherited overrides BEFORE importing tracking modules in this process."""
    values = {
        "PENGUIN_TRACKING_FPS": str(args.fps),
        "PENGUIN_TRACKING_HEIGHT": str(args.height),
        "PENGUIN_TRACKING_CHUNK_SECONDS": str(args.chunk_seconds),
    }
    os.environ.update(values)
    return values


def summarize_tracks(tracks):
    """Validate observations and summarize image-space measurements without ML."""
    from tools.tracking_results import validate_tracks

    validate_tracks(tracks)
    timestamps, speeds = [], []
    for track_id, observations in tracks.items():
        for point in observations:
            if not {"vx", "vy"}.issubset(point):
                raise ValueError(f"Track {track_id!r} has no computed velocity fields.")
            speed = math.hypot(point["vx"], point["vy"])
            if not math.isfinite(speed):
                raise ValueError(f"Track {track_id!r} has a nonfinite speed.")
            timestamps.append(point["second"])
            speeds.append(speed)
    return {
        "track_count": len(tracks), "observation_count": len(timestamps),
        "first_timestamp_seconds": min(timestamps) if timestamps else None,
        "last_timestamp_seconds": max(timestamps) if timestamps else None,
        "strictly_increasing_timestamps_per_track": True,
        "finite_velocities": True,
        "velocity_units": "proxy_pixels_per_second",
        "maximum_speed": max(speeds) if speeds else None,
        "nonzero_velocity_observations": sum(speed > 0 for speed in speeds),
    }


def outcome(stats, workers):
    reasons = []
    if stats["track_count"] == 0:
        reasons.append("No penguin tracks were detected; choose a clip with clearly visible animals.")
    def cuda_proved(worker):
        if not isinstance(worker, dict) or worker.get("status") != "ok":
            return False
        gpu = worker.get("gpu")
        return (isinstance(gpu, dict) and gpu.get("cuda_available") is True
                and type(gpu.get("peak_allocated_bytes")) is int and gpu["peak_allocated_bytes"] > 0)

    if not workers or not all(cuda_proved(worker) for worker in workers):
        reasons.append("Successful SAM3 CUDA allocation was not measured for every worker.")
    return ("inconclusive", 2, reasons) if reasons else ("ok", 0, [])


def read_worker_reports(run_dir):
    """Keep corrupt/missing instrumentation from hiding the original smoke error."""
    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    reports = []
    for path in sorted((Path(run_dir) / "gpu_metrics").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
            if not isinstance(record, dict) or record.get("status") not in ("ok", "error"):
                raise ValueError("Invalid worker measurement object.")
            reports.append(record)
        except (OSError, ValueError) as exc:
            reports.append({"status": "error", "metrics_file": str(path), "error": str(exc)})
    return reports


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def gpu_info(reset_peak=False):
    import torch

    result = {"torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
              "cuda_available": bool(torch.cuda.is_available())}
    if result["cuda_available"]:
        device = torch.cuda.current_device()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device)
        result.update({
            "device_index": int(device), "device_name": torch.cuda.get_device_name(device),
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        })
    return result


def measured_sam3_worker(*args):
    """Instrument the real worker; this top-level target is safe for spawn."""
    from tools.tracking_sam3 import _run_sam3_subprocess

    started = time.perf_counter()
    record = {"status": "error", "input_name": Path(args[0]).name}
    try:
        record["gpu"] = gpu_info(reset_peak=True)
        if not record["gpu"]["cuda_available"]:
            raise RuntimeError("PyTorch does not have access to a CUDA device.")
        _run_sam3_subprocess(*args)
        record["status"] = "ok"
        payload = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        record["raw_track_count"] = len(payload["tracks"])
        record["raw_observation_count"] = sum(len(points) for points in payload["tracks"].values())
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            record["gpu"] = gpu_info()
        except Exception as exc:
            record["gpu_measurement_error"] = str(exc)
        record["elapsed_seconds"] = time.perf_counter() - started
        metrics_dir = Path(os.environ["PENGUIN_SMOKE_METRICS_DIR"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics_dir / f"sam3-{os.getpid()}-{uuid.uuid4().hex}.json", record)


def cut_clip(source, destination, start, seconds):
    command = ["ffmpeg", "-nostdin", "-y", "-ss", str(start), "-i", str(source), "-t", str(seconds),
               "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg clip extraction failed (exit {exc.returncode}): {(exc.stderr or '')[-4000:]}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot start FFmpeg for clip extraction: {exc}") from exc
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced no nonempty input clip.")


def run_inference(args, run_dir, summary):
    # These settings are inherited naturally by multiprocessing's spawned children.
    summary["tracking_environment"] = configure_environment(args)
    os.environ["PENGUIN_SMOKE_METRICS_DIR"] = str(run_dir / "gpu_metrics")
    from tools.sam3_runtime import resolve_sam3_runtime

    runtime = resolve_sam3_runtime()
    summary["sam3_source"] = str(runtime.source_root)
    summary["sam3_checkpoint"] = str(runtime.checkpoint_path) if runtime.checkpoint_path else "automatic Hugging Face cache/download"
    summary["gpu"] = gpu_info()
    if not summary["gpu"]["cuda_available"]:
        raise RuntimeError("CUDA is unavailable to PyTorch. Check the installed CUDA build and GPU access.")
    source = args.video.expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Source video is missing or empty: {source}")
    clip = run_dir / "input_clip.mp4"
    cut_clip(source, clip, args.start, args.seconds)
    tracks_path, video_path = run_dir / "tracks.json", run_dir / "annotated.mp4"
    artifacts = {"input_clip": str(clip), "tracks": str(tracks_path), "annotated_video": str(video_path)}
    summary["artifacts"] = artifacts

    if args.stage == "pipeline":
        from tools import feature_tracking

        # Only add measurement around the real SAM3 call; no model is replaced.
        feature_tracking._run_sam3_subprocess = measured_sam3_worker
        feature_tracking.extract_tracks(str(clip), str(tracks_path), str(video_path), crop_dir=run_dir / "crops")
    else:
        from tools.tracking_utils import create_lightweight_proxy_video
        from tools.tracking_kinematics import recompute_velocities

        proxy, worker_data = run_dir / "sam3_input_proxy.mp4", run_dir / "sam3_worker.json"
        create_lightweight_proxy_video(str(clip), str(proxy))
        measured_sam3_worker(str(proxy), str(video_path), str(worker_data), 0, None, None, str(run_dir / "crops"))
        payload = json.loads(worker_data.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError("SAM3 worker did not report successful inference.")
        write_json(tracks_path, recompute_velocities(payload["tracks"]))
        artifacts.update({"sam3_proxy": str(proxy), "sam3_worker_result": str(worker_data)})
        summary["identity_scope"] = "one clip, chunk-local SAM3 IDs; no SigLIP classification or boundary stitching"

    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    summary["measurements"] = summarize_tracks(tracks)


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_parent = args.output_dir.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="smoke-", dir=output_parent))
    summary = {"stage": args.stage, "status": "error", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_video": str(args.video.expanduser().resolve()), "run_directory": str(run_dir),
               "parameters": {key: getattr(args, key) for key in ("start", "seconds", "chunk_seconds", "fps", "height")}}
    started = time.perf_counter()
    exit_code = 1
    try:
        run_inference(args, run_dir, summary)
        workers = read_worker_reports(run_dir)
        summary["workers"] = workers
        summary["status"], exit_code, summary["inconclusive_reasons"] = outcome(summary["measurements"], workers)
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        summary["workers"] = read_worker_reports(run_dir)
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        print(f"Smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"Smoke status: {summary['status']}; summary: {summary_path}")
    return exit_code


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
