"""Tracking orchestration with isolated runs and explicit publication status."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from tools.tracking_results import (
    MANIFEST_NAME, MANIFEST_VERSION, TrackingError, artifact_record,
    atomic_write_json, load_tracking_result, read_tracks, video_identity,
)


def extract_tracks(*args, **kwargs):
    # Configuration/provenance helpers and their tests must not import ML packages.
    from tools.feature_tracking import extract_tracks as implementation
    return implementation(*args, **kwargs)


class TrackingMixin:
    def _tracking_paths(self, video_name, output_dir=None):
        video = (Path(self.videos_dir) / video_name).resolve()
        if output_dir is None:
            output_dir = Path(self.videos_dir) / (Path(video_name).stem + "_output")
        return video, Path(output_dir).resolve()

    def invalidate_tracking_state(self):
        self.tracks = None
        self.tracking_result = None
        self.current_rag_refs = {}
        self.rag_ref_counter = 1

    def set_video_context(self, video_name, output_dir=None):
        video, base = self._tracking_paths(video_name, output_dir)
        try:
            source = video_identity(video)
        except TrackingError:
            self.invalidate_tracking_state()
            self._tracking_context = None
            raise
        context = (source["path"], source["sha256"], str(base))
        changed = context != getattr(self, "_tracking_context", None)
        if changed:
            self.invalidate_tracking_state()
            self._tracking_context = context
        return changed

    def _load_and_remap_tracks(self, json_path, output_dir):
        # Reading data alone does not confer provenance; reports require a receipt.
        mapping = Path(output_dir) / "tracks_species_mapping.txt"
        self.tracks = read_tracks(json_path, mapping if mapping.is_file() else None)
        return self.tracks

    def require_tracking_result(self, video_name, output_dir=None):
        video, base = self._tracking_paths(video_name, output_dir)
        key = (os.path.normcase(str(video)), str(base))
        try:
            result = load_tracking_result(video, base)
            if (key == getattr(self, "_tracking_failed_context", None)
                    and result.run_id == getattr(self, "_tracking_failed_run_id", None)):
                raise TrackingError("The latest tracking attempt failed. Run tracking again before using results.")
        except TrackingError:
            self.invalidate_tracking_state()
            raise
        self.tracks = result.tracks
        self.tracking_result = result
        if key == getattr(self, "_tracking_failed_context", None):
            self._tracking_failed_context = None
        return result

    def run_tracking(self, video_name, output_dir=None, compact_annotation=False):
        video, base = self._tracking_paths(video_name, output_dir)
        self.invalidate_tracking_state()
        key = (os.path.normcase(str(video)), str(base))
        self._tracking_failed_context = key
        try:
            previous = json.loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))
            self._tracking_failed_run_id = previous.get("run_id") if isinstance(previous, dict) and previous.get("status") == "success" else None
        except (OSError, ValueError):
            self._tracking_failed_run_id = None
        run_id = uuid.uuid4().hex
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "run_id": run_id,
            "status": "running",
            "source": None,
            "parameters": {"compact_annotation": bool(compact_annotation), "model_name": self.model},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        staging = None
        locked = False
        retain_lock = False
        try:
            base.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(base / ".tracking.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise TrackingError("Another tracking run is active for this output directory. A stopped process may have left a .tracking.lock file.") from exc
            locked = True
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
                lock.write(run_id)
            # Replace the active receipt before any work; no previous result is current now.
            atomic_write_json(base / MANIFEST_NAME, manifest)
            manifest["source"] = video_identity(video)
            from tools import tracking_config
            manifest["parameters"].update({
                "target_fps": tracking_config.TARGET_FPS,
                "target_height": tracking_config.TARGET_HEIGHT,
                "chunk_duration_seconds": tracking_config.CHUNK_DURATION_SEC,
                "max_distance_relative_width": tracking_config.MAX_DIST_REL_WIDTH,
                "siglip_confidence_threshold": tracking_config.SIGLIP_CONFIDENCE_THRESH,
            })
            atomic_write_json(base / MANIFEST_NAME, manifest)
            staging = Path(tempfile.mkdtemp(prefix=".tracking-stage-", dir=base)).resolve()
            if staging.parent != base:
                raise TrackingError("Invalid tracking staging directory.")
            json_path = staging / "tracks.json"
            annotated = staging / "annotated.mp4"
            returned = extract_tracks(
                str(video), str(json_path), str(annotated),
                model_name=self.model, compact_annotation=compact_annotation,
                crop_dir=str(staging / "crops"),
                sam3_config_path=getattr(self, "config_path", None),
            )
            if (not isinstance(returned, (tuple, list)) or len(returned) != 2
                    or not all(isinstance(path, (str, os.PathLike)) for path in returned)
                    or tuple(Path(path).resolve() for path in returned) != (json_path, annotated)):
                raise TrackingError("Tracking did not return the fresh artifacts requested for this run.")
            mapping = staging / "tracks_species_mapping.txt"
            read_tracks(json_path, mapping if mapping.is_file() else None)
            manifest["artifacts"] = {
                "tracks": artifact_record(json_path),
                "video": artifact_record(annotated),
                "species_mapping": artifact_record(mapping) if mapping.is_file() else None,
            }
            if video_identity(video) != manifest["source"]:
                raise TrackingError("The source video changed during tracking. Run tracking again.")
            runs = base / "runs"
            runs.mkdir(exist_ok=True)
            if runs.resolve().parent != base:
                raise TrackingError("Tracking run directory points outside the output directory.")
            published = runs / run_id
            if published.exists():
                raise TrackingError("Tracking run ID already exists; no previous run was overwritten.")
            manifest["status"] = "success"
            manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(staging / "run_manifest.json", manifest)
            os.rename(staging, published)
            # A single atomic pointer publishes the complete JSON/video/mapping bundle.
            atomic_write_json(base / MANIFEST_NAME, manifest)
        except BaseException as exc:
            self.invalidate_tracking_state()
            if locked:
                manifest["status"] = "failed"
                manifest["error"] = str(exc)
                manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    atomic_write_json(base / MANIFEST_NAME, manifest)
                except OSError as recording_error:
                    # Fail closed across restarts when the failure receipt cannot be saved.
                    retain_lock = True
                    if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise TrackingError(
                            f"Tracking failed: {exc}. Cannot record the failed state: {recording_error}. "
                            "The .tracking.lock file was retained to block stale results; check the output directory before recovery."
                        ) from exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit, TrackingError)):
                raise
            raise TrackingError(f"Tracking failed: {exc}") from exc
        finally:
            if staging is not None and staging.exists():
                # Delete only this run's known staging child, never published/history data.
                if staging.resolve().parent == base and staging.name.startswith(".tracking-stage-"):
                    shutil.rmtree(staging)
            if locked and not retain_lock:
                (base / ".tracking.lock").unlink(missing_ok=True)
        self._tracking_failed_context = None
        result = self.require_tracking_result(video_name, base)
        self._tracking_context = (manifest["source"]["path"], manifest["source"]["sha256"], str(base))
        return result.tracks, str(result.annotated_video_path)
