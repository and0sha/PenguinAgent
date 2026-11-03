"""Provenance and validation for complete tracking runs (no ML dependencies)."""

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import uuid


MANIFEST_NAME = "tracking_manifest.json"
# Version 2 requires globally assigned identities and timestamp-based velocities.
# Older successful runs must be reprocessed rather than silently reused.
MANIFEST_VERSION = 2


class TrackingError(RuntimeError):
    """Tracking failed or the requested result is not a verified current run."""


@dataclass(frozen=True)
class TrackingResult:
    run_id: str
    output_dir: Path
    json_path: Path
    annotated_video_path: Path
    tracks: dict


def file_fingerprint(path):
    path = Path(path).resolve()
    try:
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        after = path.stat()
    except OSError as exc:
        raise TrackingError(f"Cannot read tracking input/artifact {path}: {exc}") from exc
    signature = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    if signature(before) != signature(after) or size != after.st_size:
        raise TrackingError(f"File changed while it was being read: {path}. Please retry.")
    return {"sha256": digest.hexdigest(), "size": size}


def video_identity(path):
    path = Path(path).resolve()
    fingerprint = file_fingerprint(path)
    if fingerprint["size"] == 0:
        raise TrackingError(f"Video file is empty: {path}")
    return {"path": os.path.normcase(str(path)), **fingerprint}


def atomic_write_json(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            json.dump(data, destination, ensure_ascii=False, allow_nan=False, indent=2)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_tracks(tracks):
    if not isinstance(tracks, dict):
        raise TrackingError("Tracking JSON must contain an object of tracks.")
    for track_id, observations in tracks.items():
        if not isinstance(track_id, str) or not track_id or not isinstance(observations, list) or not observations:
            raise TrackingError("Each track must have a nonempty string ID and an observation list.")
        previous = -1.0
        for point in observations:
            if not isinstance(point, dict) or not {"second", "x", "y"}.issubset(point):
                raise TrackingError(f"Track {track_id!r} is missing timestamp/position fields.")
            for field in ("second", "x", "y", "area", "vx", "vy"):
                if field not in point:
                    continue
                value = point[field]
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TrackingError(f"Track {track_id!r} has an invalid numeric {field}.")
                try:
                    finite = math.isfinite(value)
                except (OverflowError, ValueError):
                    finite = False
                if not finite or (field in {"second", "area"} and value < 0):
                    raise TrackingError(f"Track {track_id!r} has an invalid numeric {field}.")
            if point["second"] <= previous:
                raise TrackingError(f"Track {track_id!r} must have strictly increasing unique timestamps.")
            previous = point["second"]
    return tracks


def _read_snapshot(path, expected=None):
    contents = Path(path).read_bytes()
    if expected is not None:
        observed = {"name": Path(path).name, "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
        if observed != expected:
            raise TrackingError(f"Tracking artifact changed after publication: {Path(path).name}. Run tracking again.")
    return contents


def read_tracks(path, mapping_path=None, *, expected_tracks=None, expected_mapping=None):
    try:
        # Verify and parse the same immutable bytes, not separate reads of the path.
        tracks = validate_tracks(json.loads(_read_snapshot(path, expected_tracks)))
        if mapping_path is None:
            return tracks
        mapping = {}
        for line in _read_snapshot(mapping_path, expected_mapping).decode("utf-8").splitlines():
            key, separator, value = line.strip().partition(":")
            if separator and key.strip().isdigit():
                if not value.strip():
                    raise TrackingError("Species mapping contains an empty species name.")
                mapping[key.strip()] = value.strip()
        remapped = {}
        for key, observations in tracks.items():
            class_id, separator, instance_id = key.partition("_")
            mapped = f"{mapping[class_id]}_{instance_id if separator else '1'}" if class_id in mapping else key
            if mapped in remapped:
                raise TrackingError("Species mapping would merge different tracking IDs.")
            remapped[mapped] = observations
        return remapped
    except (OSError, ValueError, TypeError) as exc:
        raise TrackingError(f"Cannot load tracking data {path}: {exc}") from exc


def artifact_record(path):
    fingerprint = file_fingerprint(path)
    if fingerprint["size"] == 0:
        raise TrackingError(f"Tracking artifact is empty: {path}")
    return {"name": Path(path).name, **fingerprint}


def _verified_artifact(run_dir, record, expected_name):
    if not isinstance(record, dict) or record.get("name") != expected_name:
        raise TrackingError(f"Tracking manifest has an invalid {expected_name} artifact.")
    path = run_dir / expected_name
    if path.resolve().parent != run_dir.resolve():
        raise TrackingError("Tracking artifact points outside its run directory.")
    if artifact_record(path) != record:
        raise TrackingError(f"Tracking artifact changed after publication: {expected_name}. Run tracking again.")
    return path


def load_tracking_result(video_path, output_dir):
    """Load only the latest complete run for the exact current video contents."""
    base = Path(output_dir).resolve()
    if (base / ".tracking.lock").exists():
        raise TrackingError("A tracking run is in progress. Results are unavailable until it finishes successfully.")
    try:
        with (base / MANIFEST_NAME).open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, ValueError, RuntimeError) as exc:
        raise TrackingError("No verified tracking run is available. Run tracking for this video; legacy files cannot prove their origin.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_VERSION:
        raise TrackingError("Unsupported tracking manifest. Run tracking again.")
    if manifest.get("status") != "success":
        raise TrackingError("The latest tracking attempt is incomplete or failed. Run tracking successfully before using its results.")
    if manifest.get("source") != video_identity(video_path):
        raise TrackingError("Tracking results belong to another video or an older version of this file. Run tracking again.")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
        raise TrackingError("Invalid tracking run ID.")
    runs_dir = base / "runs"
    run_dir = runs_dir / run_id
    if runs_dir.resolve().parent != base or run_dir.resolve().parent != runs_dir.resolve():
        raise TrackingError("Tracking run directory points outside its output directory.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TrackingError("Tracking manifest is missing artifact records.")
    json_path = _verified_artifact(run_dir, artifacts.get("tracks"), "tracks.json")
    video = _verified_artifact(run_dir, artifacts.get("video"), "annotated.mp4")
    mapping = None
    if artifacts.get("species_mapping") is not None:
        mapping = _verified_artifact(run_dir, artifacts["species_mapping"], "tracks_species_mapping.txt")
    tracks = read_tracks(json_path, mapping, expected_tracks=artifacts["tracks"], expected_mapping=artifacts.get("species_mapping"))
    # Detect replacements while the data snapshot was parsed and validated.
    _verified_artifact(run_dir, artifacts["tracks"], "tracks.json")
    _verified_artifact(run_dir, artifacts["video"], "annotated.mp4")
    if mapping is not None:
        _verified_artifact(run_dir, artifacts["species_mapping"], "tracks_species_mapping.txt")
    if manifest["source"] != video_identity(video_path):
        raise TrackingError("The source video changed during result validation. Run tracking again.")
    # A different attempt may have started while artifacts were being checked.
    try:
        latest = json.loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrackingError("Tracking state changed during validation. Please retry.") from exc
    if latest != manifest or (base / ".tracking.lock").exists():
        raise TrackingError("Tracking state changed during validation. Please retry.")
    return TrackingResult(run_id, run_dir, json_path, video, tracks)
