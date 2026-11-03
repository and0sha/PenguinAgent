"""Streamlit-independent state transitions for the current tracking run."""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
from tools.tracking_results import TrackingError


_DERIVED_SESSION_KEYS = (
    "report", "report_run_id", "report_rag_refs", "messages", "pending_tag", "last_copied_tag",
    "annotated_video", "analysis_run_id",
)


@dataclass(frozen=True)
class SavedUpload:
    filename: str
    path: Path
    changed: bool


def reset_session_analysis_state(session):
    """Remove only values derived from a video analysis, preserving user settings."""
    for key in _DERIVED_SESSION_KEYS:
        session.pop(key, None)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as video_file:
        for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def save_uploaded_video(filename, contents, videos_dir):
    """Publish changed upload bytes atomically; unchanged reruns do not rewrite it."""
    if (not isinstance(filename, str) or not filename or filename in {".", ".."}
            or any(character in filename for character in '/\\:\0')
            or any(ord(character) < 32 for character in filename)):
        raise ValueError("The uploaded video must have a plain filename")
    directory = Path(videos_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    uploaded_digest = hashlib.sha256(contents).digest()
    if (destination.is_file() and destination.stat().st_size == len(contents)
            and _file_digest(destination) == uploaded_digest):
        return SavedUpload(filename, destination, False)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".upload-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return SavedUpload(filename, destination, True)


def sync_tracking_context(session, agent, video_name, output_dir):
    """Invalidate session artifacts whenever the agent's source context changes."""
    changed = agent.set_video_context(video_name, output_dir=output_dir)
    if changed:
        reset_session_analysis_state(session)
    return changed


def begin_tracking_attempt(session, agent):
    """A failed rerun must leave no previous successful result active in the UI."""
    reset_session_analysis_state(session)
    agent.invalidate_tracking_state()


def _discard_invalid_analysis(session, agent):
    # A stable literature-only conversation has no run-bound artifacts to discard.
    if any(key in session for key in ("analysis_run_id", "report", "report_run_id", "annotated_video")):
        reset_session_analysis_state(session)
    agent.invalidate_tracking_state()


def current_tracking_result(session, agent, video_name, output_dir):
    """Return only a backend-validated current result; failed validation clears UI state."""
    if not video_name:
        _discard_invalid_analysis(session, agent)
        return None
    try:
        result = agent.require_tracking_result(video_name, output_dir=output_dir)
    except TrackingError:
        _discard_invalid_analysis(session, agent)
        return None
    if session.get("analysis_run_id") != result.run_id:
        reset_session_analysis_state(session)
        session["analysis_run_id"] = result.run_id
    return result


def store_report_for_result(session, paper, result, rag_refs=None):
    """Do not present legacy backend error strings as successful scientific reports."""
    session.pop("report", None)
    session.pop("report_run_id", None)
    session.pop("report_rag_refs", None)
    if not isinstance(paper, str) or not paper.strip():
        raise ValueError("The report generator returned no report")
    if paper.lstrip().casefold().startswith(("error:", "error ", "no tracks available", "no tracking data")):
        raise ValueError(paper.strip())
    session["report"] = paper
    session["report_run_id"] = result.run_id
    session["report_rag_refs"] = deepcopy(rag_refs or {})


def report_for_result(session, result):
    if result is None or session.get("report_run_id") != result.run_id:
        return None
    return session.get("report")
