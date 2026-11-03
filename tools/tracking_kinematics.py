"""Image-space velocities from finalized global track identities and timestamps."""

import math

from tools.tracking_results import TrackingError, validate_tracks


def recompute_velocities(tracks):
    """Return copied observations with centroid velocities in pixels per second.

    Identity assignment and global timestamp offsets must be complete first. The
    first observation of each track uses the existing zero-vector convention;
    every later observation, including a chunk boundary or observation gap, uses
    displacement / actual elapsed seconds. No measurements are interpolated.
    """
    validate_tracks(tracks)
    recomputed = {}
    for track_id, observations in tracks.items():
        points = []
        previous = None
        for observation in observations:
            point = dict(observation)
            vx, vy = 0.0, 0.0
            if previous is not None:
                try:
                    elapsed = point["second"] - previous["second"]
                    vx = float((point["x"] - previous["x"]) / elapsed)
                    vy = float((point["y"] - previous["y"]) / elapsed)
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise TrackingError(f"Cannot calculate a finite velocity for track {track_id!r}.") from exc
                if not math.isfinite(vx) or not math.isfinite(vy):
                    raise TrackingError(f"Cannot calculate a finite velocity for track {track_id!r}.")
            point["vx"], point["vy"] = vx, vy
            points.append(point)
            previous = point
        recomputed[track_id] = points
    return recomputed
