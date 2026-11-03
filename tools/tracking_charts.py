"""Bounded, deterministic charts for tracking data; never execute model input."""

import json
from itertools import islice
import math
from numbers import Real
from pathlib import Path
import threading
import uuid


CHART_TYPES = frozenset({"trajectory", "speed", "area", "population"})
MAX_TRACKS = 32
MAX_POINTS_PER_TRACK = 5000
MAX_TOTAL_POINTS = 20000
MAX_TITLE_LENGTH = 160
MAX_TRACK_ID_LENGTH = 256
MAX_ABSOLUTE_VALUE = 1e12
MAX_CHARTS_PER_REPORT = 8
_RENDER_LOCK = threading.RLock()


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(result) or abs(result) > MAX_ABSOLUTE_VALUE:
        raise ValueError(f"{field} must be finite and within +/-{MAX_ABSOLUTE_VALUE:g}")
    return result


def validate_chart_arguments(arguments):
    """Validate the entire tool payload, including unexpected/legacy parameters."""
    if not isinstance(arguments, dict):
        raise ValueError("Chart arguments must be an object")
    if set(arguments) - {"chart_type", "track_id", "title"}:
        raise ValueError("Only chart_type, track_id and title arguments are allowed")
    chart_type = arguments.get("chart_type")
    track_id = arguments.get("track_id", "")
    title = arguments.get("title", "")
    if not isinstance(chart_type, str) or chart_type not in CHART_TYPES:
        raise ValueError("chart_type must be trajectory, speed, area or population")
    if not isinstance(track_id, str) or len(track_id) > MAX_TRACK_ID_LENGTH:
        raise ValueError(f"track_id must be a string of at most {MAX_TRACK_ID_LENGTH} characters")
    if not isinstance(title, str) or len(title) > MAX_TITLE_LENGTH:
        raise ValueError(f"title must be a string of at most {MAX_TITLE_LENGTH} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in title):
        raise ValueError("title must be a single line of literal text")
    return {"chart_type": chart_type, "track_id": track_id, "title": title}


def dispatch_tracking_chart(arguments, callback):
    """Return a tool error rather than forwarding unvalidated keyword arguments."""
    try:
        validated = validate_chart_arguments(arguments)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
    return callback(**validated)


def _validated_tracks(tracks, chart_type, track_id):
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError("Tracking data must be a nonempty object")
    if track_id:
        if track_id not in tracks:
            raise ValueError("Unknown track_id")
        selected = [(track_id, tracks[track_id])]
    else:
        if len(tracks) > MAX_TRACKS:
            raise ValueError(f"At most {MAX_TRACKS} tracks can be plotted; select a track_id")
        selected = list(tracks.items())
    validated = []
    total_points = 0
    for pid, points in selected:
        if not isinstance(pid, str) or not pid or len(pid) > MAX_TRACK_ID_LENGTH:
            raise ValueError("Track IDs must be nonempty bounded strings")
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError("Each selected track must contain observations")
        if len(points) > MAX_POINTS_PER_TRACK:
            raise ValueError(f"At most {MAX_POINTS_PER_TRACK} observations per track are supported")
        total_points += len(points)
        if total_points > MAX_TOTAL_POINTS:
            raise ValueError(f"At most {MAX_TOTAL_POINTS} total observations are supported")
        cleaned = []
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("Each observation must be an object")
            required = {"second"}
            if chart_type in {"trajectory", "speed"}:
                required.update({"x", "y"})
            if chart_type == "area":
                required.add("area")
            if not required.issubset(point):
                raise ValueError("Observation is missing required numeric fields")
            values = {key: _number(point[key], key) for key in ("second", "x", "y", "area", "vx", "vy") if key in point}
            if values["second"] < 0 or values.get("area", 0) < 0:
                raise ValueError("second and area must be nonnegative")
            cleaned.append(values)
        cleaned.sort(key=lambda point: point["second"])
        if any(a["second"] == b["second"] for a, b in zip(cleaned, cleaned[1:])):
            raise ValueError("A track must not have duplicate timestamps")
        validated.append((pid, cleaned))
    return validated


def render_tracking_chart(tracks, output_dir, chart_type, track_id="", title=""):
    """Render an allowlisted PNG and return measured values, using no global pyplot.

    Speed is derived from centroid displacement divided by elapsed seconds;
    stored vx/vy may have different sampling semantics and are not used.
    Population counts observations at exact timestamps, without interpolation.
    """
    arguments = validate_chart_arguments({"chart_type": chart_type, "track_id": track_id, "title": title})
    selected = _validated_tracks(tracks, chart_type, arguments["track_id"])
    seconds = [point["second"] for _, points in selected for point in points]
    summary = {
        "selected_track_count": len(selected),
        "observation_count": len(seconds),
        "time_range_seconds": [min(seconds), max(seconds)],
        "per_track": {},
        "limitations": "Image-space measurements; no physical calibration or thermodynamic inference.",
    }
    series = []
    if chart_type == "population":
        counts = {}
        for _, points in selected:
            for point in points:
                second = point["second"]
                counts[second] = counts.get(second, 0) + 1
        times = sorted(counts)
        series.append(("Observed tracks", times, [counts[time] for time in times]))
        summary.update({"unit": "observed tracks", "minimum_count": min(counts.values()), "maximum_count": max(counts.values()), "population_definition": "Number of selected tracks observed at each exact timestamp; gaps are not interpolated."})
    else:
        for pid, points in selected:
            stats = {"observation_count": len(points)}
            if chart_type == "trajectory":
                x_values = [point["x"] for point in points]
                y_values = [point["y"] for point in points]
                stats.update({"x_range_px": [min(x_values), max(x_values)], "y_range_px": [min(y_values), max(y_values)], "path_length_px": sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(points, points[1:]))})
                summary["unit"] = "pixels"
            elif chart_type == "speed":
                x_values = [point["second"] for point in points[1:]]
                y_values = [math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / (b["second"] - a["second"]) for a, b in zip(points, points[1:])]
                if any(not math.isfinite(value) for value in y_values):
                    raise ValueError("Derived speed exceeds the finite numeric range")
                stats.update({"interval_count": len(y_values), "maximum_speed_px_per_second": max(y_values) if y_values else None})
                summary["unit"] = "pixels/second"
                summary["speed_definition"] = "Centroid displacement divided by elapsed seconds between consecutive observations."
            else:
                x_values = [point["second"] for point in points]
                y_values = [point["area"] for point in points]
                stats.update({"minimum_area_px_squared": min(y_values), "maximum_area_px_squared": max(y_values), "mean_area_px_squared": sum(y_values) / len(y_values)})
                summary["unit"] = "pixels squared"
            summary["per_track"][pid] = stats
            if x_values:
                series.append((pid, x_values, y_values))
        if not series:
            raise ValueError("Speed charts require at least two observations in a selected track")

    # The directory is supplied by the application, and the filename is generated here.
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"tracking_chart_{uuid.uuid4().hex}.png"
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    with _RENDER_LOCK:
        figure = Figure(figsize=(8, 5), dpi=120)
        canvas = FigureCanvasAgg(figure)
        try:
            axes = figure.subplots()
            for label, x_values, y_values in series:
                axes.plot(x_values, y_values, marker=".", markersize=3, linewidth=1, label=label)
            defaults = {"trajectory": "Tracking trajectories", "speed": "Centroid speed", "area": "Tracked mask area", "population": "Observed track count"}
            axes.set_title(title or defaults[chart_type], fontsize=12)
            if chart_type == "trajectory":
                axes.set_xlabel("X (pixels)")
                axes.set_ylabel("Y (pixels)")
                axes.invert_yaxis()
                axes.set_aspect("equal", adjustable="datalim")
            else:
                axes.set_xlabel("Time (seconds)")
                axes.set_ylabel({"speed": "Speed (pixels/second)", "area": "Area (pixels squared)", "population": "Observed tracks"}[chart_type])
            axes.grid(True, alpha=0.25)
            if len(series) <= 10:
                # Explicit handles keep arbitrary track IDs beginning with '_' visible.
                axes.legend(handles=axes.lines, labels=[item[0] for item in series], fontsize=7)
            for text in figure.findobj(Text):
                text.set_usetex(False)
                text.set_parse_math(False)
            figure.tight_layout()
            # Exclusive creation prevents replacing an existing path or symlink.
            with output_path.open("xb") as image_file:
                try:
                    canvas.print_png(image_file)
                except Exception:
                    image_file.close()
                    output_path.unlink(missing_ok=True)
                    raise
        finally:
            figure.clear()
    return {"success": True, "path": str(output_path), "chart_type": chart_type, "summary": summary}

# Small previews keep video reports inside a bounded model/browser context. The
# full validated observations remain available in the run's tracks.json.
EVIDENCE_MAX_TRACKS = 12
EVIDENCE_MAX_POINTS_PER_TRACK = 8


def build_tracking_evidence(tracks, track_id=""):
    """Return bounded numeric observations and honest endpoint/sample statistics.

    Velocity previews use the immediately preceding FULL observation, not the
    preceding preview row. Maximum speeds are explicitly sampled maxima, and
    observation counts are never described as durations. No ethological score,
    physical calibration or thermodynamic value can be inferred here.
    """
    if not isinstance(tracks, dict):
        raise ValueError("Tracking data must be an object")
    if not isinstance(track_id, str) or len(track_id) > MAX_TRACK_ID_LENGTH:
        raise ValueError("Invalid track_id")
    if track_id and track_id not in tracks:
        raise ValueError("Unknown track_id")
    selected_ids = [track_id] if track_id else list(islice(tracks, EVIDENCE_MAX_TRACKS))
    evidence = {
        "total_track_count": len(tracks),
        "included_track_count": len(selected_ids),
        "omitted_track_count": len(tracks) - len(selected_ids),
        "units": {"second": "video seconds", "x_y": "proxy image pixels", "vx_vy": "pixels/second", "area": "mask pixels squared"},
        "limitations": "Bounded observation preview; gaps are not interpolated. Image-space motion and mask area do not establish biological identity, physical speed, rarity, novelty, individual baseline, wind or heat loss.",
        "tracks": [],
    }
    for pid in selected_ids:
        if not isinstance(pid, str) or not pid or len(pid) > MAX_TRACK_ID_LENGTH:
            raise ValueError("Track IDs must be bounded nonempty strings")
        points = tracks[pid]
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError("Each included track needs observations")
        def measurement(index):
            point = points[index]
            if not isinstance(point, dict) or not {"second", "x", "y"}.issubset(point):
                raise ValueError("Observations need second, x and y")
            row = {key: _number(point[key], key) for key in ("second", "x", "y", "area") if key in point}
            if row["second"] < 0 or row.get("area", 0) < 0:
                raise ValueError("second and area must be nonnegative")
            return row
        count = len(points)
        preview_count = min(count, EVIDENCE_MAX_POINTS_PER_TRACK)
        indices = [0] if preview_count == 1 else [round(i * (count - 1) / (preview_count - 1)) for i in range(preview_count)]
        rows = []
        for index in indices:
            row = measurement(index)
            row["observation_index"] = index
            row["vx"], row["vy"], row["speed"] = None, None, None
            if index:
                prior = measurement(index - 1)
                elapsed = row["second"] - prior["second"]
                if elapsed <= 0:
                    raise ValueError("Observation timestamps must increase")
                vx = (row["x"] - prior["x"]) / elapsed
                vy = (row["y"] - prior["y"]) / elapsed
                speed = math.hypot(vx, vy)
                if not all(math.isfinite(value) for value in (vx, vy, speed)):
                    raise ValueError("Derived velocity must be finite")
                row.update({"vx": vx, "vy": vy, "speed": speed, "velocity_interval_seconds": elapsed})
            if rows and row["second"] <= rows[-1]["second"]:
                raise ValueError("Observation timestamps must increase")
            rows.append(row)
        first, last = rows[0], rows[-1]
        speeds = [row["speed"] for row in rows if row["speed"] is not None]
        evidence["tracks"].append({
            "track_id": pid, "observation_count": count,
            "preview_observation_count": len(rows),
            "omitted_observation_count": count - len(rows),
            "first_second": first["second"], "last_second": last["second"],
            "observed_span_seconds": last["second"] - first["second"],
            "endpoint_displacement_px": math.hypot(last["x"] - first["x"], last["y"] - first["y"]),
            "maximum_sampled_speed_px_per_second": max(speeds) if speeds else None,
            "observations": rows,
        })
    return evidence

class ToolResponseBudget:
    """Hard cap on cumulative tool text, including one exhaustion notice."""
    def __init__(self, max_chars=12000):
        self.remaining = max_chars
        self._stopped = False

    @property
    def exhausted(self):
        return self._stopped or self.remaining < 256

    def take(self, content):
        if self._stopped:
            return ''
        notice = json.dumps({'success': False, 'error': 'Tool context budget exhausted. Finish from existing evidence.'})
        if len(content) > max(0, self.remaining - len(notice)):
            self._stopped = True
            if len(notice) > self.remaining:
                self.remaining = 0
                return ''
            self.remaining -= len(notice)
            return notice
        self.remaining -= len(content)
        return content


def atomic_write_text(path, text):
    """Publish UTF-8 text only after a complete, flushed sibling file is ready."""
    import os
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
