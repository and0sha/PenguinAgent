import math
import os

def _positive_number_from_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a finite positive number.") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _positive_int_from_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer.") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


# Spawned SAM3/SigLIP processes inherit these overrides from the caller.
TARGET_FPS = _positive_number_from_env("PENGUIN_TRACKING_FPS", 1)
TARGET_HEIGHT = _positive_int_from_env("PENGUIN_TRACKING_HEIGHT", 720)
CHUNK_DURATION_SEC = _positive_number_from_env("PENGUIN_TRACKING_CHUNK_SECONDS", 10)
MAX_DIST_REL_WIDTH = 0.10        
SIGLIP_CONFIDENCE_THRESH = 0.1  
SIGLIP_BATCH_SIZE = _positive_int_from_env("PENGUIN_SIGLIP_BATCH_SIZE", 16)
