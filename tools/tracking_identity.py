"""Conservative geometric association at a video chunk boundary.

Raw segmentation IDs are local labels, not animal identities. This helper only
associates detections visible at frame zero of the new chunk with tracks visible
at the final frame of the previous chunk (the caller must apply both filters).
It does not perform biological re-identification or guarantee individual identity.

A previous track's last two timestamped centroids predict its next position using
constant velocity; a single observation uses the last centroid. An association
must be the mutual nearest candidate inside the distance gate, and its advantage
over the runner-up must exceed min(2 pixels, 10% of the gate) in both directions.
Ties and ambiguous competition remain unmatched. There is no greedy fallback,
raw-ID shortcut, or forced assignment: unmatched detections need fresh IDs from the
parent process. This deliberately favors identity fragmentation over unsupported
identity merges. It cannot resolve indistinguishable animals at a crossing.
"""

import math
from numbers import Real


AMBIGUITY_MARGIN_REL_GATE = 0.1
AMBIGUITY_MARGIN_MAX_PX = 2.0


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        converted = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def _validate_id(identifier):
    if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
        raise ValueError("Track IDs must be nonnegative integers")


def _predicted_positions(previous, current_second):
    predictions = {}
    for global_id in sorted(previous):
        _validate_id(global_id)
        history = previous[global_id]
        if not isinstance(history, (list, tuple)) or not history:
            raise ValueError("Each previous track needs at least one timestamped observation")
        observations = []
        for observation in history:
            if not isinstance(observation, dict) or not {"second", "x", "y"}.issubset(observation):
                raise ValueError("Previous observations must contain second, x and y")
            second = _finite_number(observation["second"], "second")
            x = _finite_number(observation["x"], "x")
            y = _finite_number(observation["y"], "y")
            if second < 0 or (observations and second <= observations[-1][0]):
                raise ValueError("Previous timestamps must be nonnegative and strictly increasing")
            observations.append((second, x, y))
        last_second, last_x, last_y = observations[-1]
        elapsed = current_second - last_second
        if elapsed <= 0:
            raise ValueError("current_second must be later than every previous track's last observation")
        if len(observations) > 1:
            prior_second, prior_x, prior_y = observations[-2]
            interval = last_second - prior_second
            prediction_x = last_x + ((last_x - prior_x) / interval) * elapsed
            prediction_y = last_y + ((last_y - prior_y) / interval) * elapsed
        else:
            prediction_x, prediction_y = last_x, last_y
        if not math.isfinite(prediction_x) or not math.isfinite(prediction_y):
            raise ValueError("Motion prediction exceeds the finite numeric range")
        predictions[global_id] = (prediction_x, prediction_y)
    return predictions


def _clear_nearest(candidates, margin):
    """Return an ID only when its nearest distance has a sufficient advantage."""
    if not candidates:
        return None
    candidates.sort()  # (distance, ID); ID order never resolves a distance tie.
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= margin:
        return None
    return candidates[0][1]


def match_boundary(previous, current, current_second, max_distance):
    """Return a deterministic injective mapping of current local ID to global ID.

    previous: dict[int, list[dict]] with second/x/y observations in chronological
        order; include ONLY tracks visible at the previous chunk's final frame.
    current: dict[int, tuple[float, float]] of ONLY new-chunk frame-zero centroids.
    current_second: absolute video timestamp of that new boundary frame.
    max_distance: positive image-coordinate gate (e.g. relative_gate * frame_width).

    All inputs are validated before matching and remain unmodified. Invalid IDs,
    nonfinite numbers, nonincreasing histories and nonpositive elapsed time raise
    ValueError. Empty inputs produce no matches. This is a geometric heuristic;
    a returned mapping is not evidence of biological identity.
    """
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError("previous and current must be dictionaries")
    current_second = _finite_number(current_second, "current_second")
    max_distance = _finite_number(max_distance, "max_distance")
    if current_second < 0 or max_distance <= 0:
        raise ValueError("current_second must be nonnegative and max_distance positive")
    # Validate keys before sorting so malformed mixed types produce ValueError.
    for identifier in previous:
        _validate_id(identifier)
    for identifier in current:
        _validate_id(identifier)
    predictions = _predicted_positions(previous, current_second)
    centroids = {}
    for local_id in sorted(current):
        centroid = current[local_id]
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            raise ValueError("Each current centroid must contain exactly x and y")
        centroids[local_id] = (_finite_number(centroid[0], "x"), _finite_number(centroid[1], "y"))

    from_current = {local_id: [] for local_id in centroids}
    from_previous = {global_id: [] for global_id in predictions}
    for local_id, (x, y) in centroids.items():
        for global_id, (predicted_x, predicted_y) in predictions.items():
            distance = math.hypot(x - predicted_x, y - predicted_y)
            # Huge finite coordinates may produce infinite subtraction/distances;
            # these pairs are outside every finite gate and cannot be candidates.
            if distance <= max_distance:
                from_current[local_id].append((distance, global_id))
                from_previous[global_id].append((distance, local_id))
    margin = min(AMBIGUITY_MARGIN_MAX_PX, max_distance * AMBIGUITY_MARGIN_REL_GATE)
    preferred_previous = {local_id: _clear_nearest(candidates, margin) for local_id, candidates in from_current.items()}
    preferred_current = {global_id: _clear_nearest(candidates, margin) for global_id, candidates in from_previous.items()}
    return {
        local_id: global_id
        for local_id, global_id in preferred_previous.items()
        if global_id is not None and preferred_current[global_id] == local_id
    }
