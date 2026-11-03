import math
from pathlib import Path
import subprocess

from tools.tracking_config import TARGET_FPS, TARGET_HEIGHT, CHUNK_DURATION_SEC
from tools.tracking_results import TrackingError


def run_ffmpeg(command, stage):
    """Run FFmpeg and preserve the useful part of its diagnostics on failure."""
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "No FFmpeg diagnostics were returned.")[-4000:].strip()
        raise TrackingError(f"{stage} failed (FFmpeg exit {exc.returncode}): {detail}") from exc
    except OSError as exc:
        raise TrackingError(f"{stage} could not start FFmpeg: {exc}") from exc


def require_nonempty_file(path, stage):
    try:
        valid = Path(path).is_file() and Path(path).stat().st_size > 0
    except OSError as exc:
        raise TrackingError(f"{stage}: cannot access {path}: {exc}") from exc
    if not valid:
        raise TrackingError(f"{stage}: expected a nonempty file at {path}.")


def validate_video(path, stage, expected_frames=None):
    """Check the file, video metadata and a decoded frame; return fps/width/height."""
    import cv2

    require_nonempty_file(path, stage)
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise TrackingError(f"{stage}: cannot open video {path}.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or not ret or frame is None:
            raise TrackingError(f"{stage}: invalid metadata or no decodable frames in {path}.")
        if expected_frames is not None and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames:
            raise TrackingError(f"{stage}: video frame count differs from the {expected_frames} expected frames.")
        return fps, width, height
    except TrackingError:
        raise
    except Exception as exc:
        raise TrackingError(f"{stage}: cannot validate video {path}: {exc}") from exc
    finally:
        if cap is not None:
            cap.release()


def create_lightweight_proxy_video(input_path, output_path, target_fps=TARGET_FPS, target_height=TARGET_HEIGHT):
    """Create and validate a lightweight SAM video, or raise TrackingError."""
    print(f"Optimizing video to save VRAM (max height: {target_height}p, {target_fps} FPS)...")
    gop_size = max(1, round(target_fps * CHUNK_DURATION_SEC))
    command = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-map', '0:v:0', '-an', '-sn', '-dn',
        '-vf', f"fps={target_fps},scale=-2:'min({target_height},ih)'",
        '-fps_mode', 'cfr', '-g', str(gop_size), '-keyint_min', str(gop_size),
        '-sc_threshold', '0', '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', str(output_path),
    ]
    run_ffmpeg(command, "Proxy video conversion")
    validate_video(output_path, "Proxy video conversion")
    print("Proxy video ready.")


def pad_to_square(image):
    from PIL import Image

    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (128, 128, 128))
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image
