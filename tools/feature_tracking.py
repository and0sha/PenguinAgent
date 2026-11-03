import os
import sys
import glob
import multiprocessing
import json
import math
from pathlib import Path
import shutil
import tempfile

from tools.tracking_config import CHUNK_DURATION_SEC, MAX_DIST_REL_WIDTH
from tools.tracking_utils import create_lightweight_proxy_video, require_nonempty_file, run_ffmpeg, validate_video
from tools.tracking_results import TrackingError, validate_tracks
from tools.tracking_sam3 import _run_sam3_subprocess
from tools.tracking_siglip import get_penguin_species_batch
from tools.tracking_render import render_final_video
from tools.tracking_identity import match_boundary
from tools.tracking_kinematics import recompute_velocities

def extract_tracks(video_path, output_json="tracks.json", output_video="annotated.mp4", model_name="gemma4:26b", **kwargs):
    """Produce complete results in isolation; failed stages never replace JSON."""
    video_path, output_json, output_video = map(os.fspath, (video_path, output_json, output_video))
    json_path, video_output = Path(output_json).resolve(), Path(output_video).resolve()
    if json_path == video_output or Path(video_path).resolve() in (json_path, video_output):
        raise TrackingError("Input video, output JSON and output video must have distinct paths.")
    crop_parent = kwargs.pop("crop_dir", None)
    try:
        require_nonempty_file(video_path, "Input video")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        video_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tracking-", dir=video_output.parent) as working:
            work_dir = Path(working)
            if crop_parent is None:
                crop_parent = work_dir
            Path(crop_parent).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="crops-", dir=crop_parent) as crops:
                staged_json = work_dir / "tracks.json"
                staged_video = work_dir / "annotated.mp4"
                _extract_tracks(video_path, str(staged_json), str(staged_video), model_name, crop_dir=crops, **kwargs)
                require_nonempty_file(staged_json, "Tracking JSON")
                validate_video(staged_video, "Completed tracking video")
                # Prepare the small JSON beside its destination before publishing.
                fd, pending_json = tempfile.mkstemp(prefix=".tracks-", suffix=".json", dir=json_path.parent)
                os.close(fd)
                pending_mapping = None
                try:
                    shutil.copyfile(staged_json, pending_json)
                    staged_mapping = staged_json.with_name("tracks_species_mapping.txt")
                    final_mapping = json_path.with_name(json_path.stem + "_species_mapping.txt")
                    if staged_mapping.exists():
                        fd, pending_mapping = tempfile.mkstemp(prefix=".species-", suffix=".txt", dir=json_path.parent)
                        os.close(fd)
                        shutil.copyfile(staged_mapping, pending_mapping)
                    os.replace(staged_video, video_output)
                    if pending_mapping is not None:
                        os.replace(pending_mapping, final_mapping)
                    elif final_mapping.exists():
                        final_mapping.unlink()
                    os.replace(pending_json, json_path)
                finally:
                    if os.path.exists(pending_json):
                        os.remove(pending_json)
                    if pending_mapping is not None and os.path.exists(pending_mapping):
                        os.remove(pending_mapping)
        print(f"Tracking complete. Data saved to {output_json} and {output_video}.")
        return output_json, output_video
    except TrackingError:
        raise
    except Exception as exc:
        raise TrackingError(f"Tracking failed: {exc}") from exc


def _read_chunk_result(path, exitcode, chunk_number):
    require_nonempty_file(path, f"SAM 3 chunk {chunk_number} result (process exit {exitcode})")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise TrackingError(f"SAM 3 chunk {chunk_number} returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TrackingError(f"SAM 3 chunk {chunk_number} returned an invalid result object.")
    if exitcode != 0 or data.get("status") != "ok":
        raise TrackingError(f"SAM 3 chunk {chunk_number} failed (exit {exitcode}): {data.get('error', 'missing success status')}")
    tracks, annotations, crops = (data.get(key) for key in ("tracks", "text_annotations", "crops_saved"))
    if not isinstance(tracks, dict) or not isinstance(annotations, list) or not annotations or not isinstance(crops, list):
        raise TrackingError(f"SAM 3 chunk {chunk_number} returned an incomplete result schema.")
    if any(type(obj_id) is not int or obj_id < 0 for obj_id in crops):
        raise TrackingError(f"SAM 3 chunk {chunk_number} returned invalid crop identifiers.")
    for pid, samples in tracks.items():
        if not isinstance(pid, str) or not pid.startswith("obj_") or not pid[4:].isdigit() or not isinstance(samples, list):
            raise TrackingError(f"SAM 3 chunk {chunk_number} returned an invalid track identifier or samples.")
        for sample in samples:
            if not isinstance(sample, dict) or any(
                type(sample.get(key)) not in (int, float) or not math.isfinite(sample[key])
                for key in ("second", "x", "y", "area", "vx", "vy")
            ):
                raise TrackingError(f"SAM 3 chunk {chunk_number} returned invalid tracking measurements.")
    observed_ids = {int(pid[4:]) for pid in tracks}
    for frame in annotations:
        if not isinstance(frame, list) or any(
            not isinstance(ann, dict) or any(type(ann.get(key)) is not int for key in ("obj_id", "cx", "cy"))
            for ann in frame
        ):
            raise TrackingError(f"SAM 3 chunk {chunk_number} returned invalid frame annotations.")
        frame_ids = [ann["obj_id"] for ann in frame]
        if len(frame_ids) != len(set(frame_ids)) or any(obj_id < 0 for obj_id in frame_ids):
            raise TrackingError(f"SAM 3 chunk {chunk_number} returned duplicate or invalid IDs in one frame.")
        observed_ids.update(ann["obj_id"] for ann in frame)
    if not observed_ids.issubset(set(crops)):
        raise TrackingError(f"SAM 3 chunk {chunk_number} is missing classification crops for observed objects.")
    return data


def _validate_chunk_telemetry(data, fps, chunk_number):
    """Bind each sparse telemetry sample to its actual annotated local frame.

    Samples may be a subset of frame observations, and annotation-only animals
    remain valid. A present sample must use the same local ID, centroid and
    timestamp as its frame; otherwise downstream stitching could publish data
    that contradicts the video. Tolerance is one millionth of a frame/pixel.
    """
    frames = [{ann["obj_id"]: ann for ann in frame} for frame in data["text_annotations"]]
    canonical_ids = set()
    for pid, samples in data["tracks"].items():
        local_id = int(pid[4:])
        if local_id in canonical_ids:
            raise TrackingError(f"SAM 3 chunk {chunk_number} contains duplicate canonical local track IDs.")
        canonical_ids.add(local_id)
        if not samples:
            raise TrackingError(f"SAM 3 chunk {chunk_number} has an empty sampled track.")
        previous_second = -1.0
        previous_frame_index = -1
        for sample in samples:
            second = sample["second"]
            frame_position = second * fps
            if second < 0 or second <= previous_second or not math.isfinite(frame_position):
                raise TrackingError(f"SAM 3 chunk {chunk_number} has invalid sample chronology.")
            previous_second = second
            frame_index = round(frame_position)
            if (frame_index < 0 or frame_index >= len(frames)
                    or not math.isclose(frame_position, frame_index, rel_tol=0, abs_tol=1e-6)):
                raise TrackingError(f"SAM 3 chunk {chunk_number} sample timestamp does not identify a local frame.")
            if frame_index <= previous_frame_index:
                raise TrackingError(f"SAM 3 chunk {chunk_number} repeats or reverses a sampled local frame.")
            previous_frame_index = frame_index
            annotation = frames[frame_index].get(local_id)
            if annotation is None:
                raise TrackingError(f"SAM 3 chunk {chunk_number} sample has no matching local object in its frame.")
            if (not math.isclose(sample["x"], annotation["cx"], rel_tol=0, abs_tol=1e-6)
                    or not math.isclose(sample["y"], annotation["cy"], rel_tol=0, abs_tol=1e-6)):
                raise TrackingError(f"SAM 3 chunk {chunk_number} sample centroid differs from its frame annotation.")


def _stitch_chunk(data, previous_boundary, identity_history, next_obj_id, frame_offset, fps, max_distance, chunk_crops, global_crops):
    """Give local IDs one injective global mapping, using only adjacent boundary frames."""
    annotations = data["text_annotations"]
    first_points = {ann["obj_id"]: (ann["cx"], ann["cy"]) for ann in annotations[0]}
    current_second = frame_offset / fps
    mapping = match_boundary(previous_boundary, first_points, current_second, max_distance)
    if not set(mapping).issubset(first_points) or not set(mapping.values()).issubset(previous_boundary):
        raise TrackingError("Boundary association returned IDs outside the adjacent boundary frames.")
    if len(set(mapping.values())) != len(mapping):
        raise TrackingError("Boundary association would merge different animals into one global ID.")

    # Reserve IDs from all observations, including animals absent from sampled tracks.
    local_order = []
    for frame in annotations:
        local_order.extend(ann["obj_id"] for ann in sorted(frame, key=lambda ann: ann["obj_id"]))
    local_order.extend(sorted(set(data["crops_saved"]) | {int(pid[4:]) for pid in data["tracks"]}))
    for local_id in local_order:
        if local_id not in mapping:
            mapping[local_id] = next_obj_id
            next_obj_id += 1
    if len(set(mapping.values())) != len(mapping):
        raise TrackingError("Chunk association does not assign a unique global ID to each local animal.")

    tracks = {}
    for pid, samples in data["tracks"].items():
        global_pid = f"obj_{mapping[int(pid[4:])]}"
        if global_pid in tracks:
            raise TrackingError("Chunk data contains duplicate local track identifiers.")
        tracks[global_pid] = [dict(sample, second=sample["second"] + current_second) for sample in samples]

    remapped_annotations = []
    last_ids = set()
    for frame_index, frame in enumerate(annotations):
        remapped = [dict(ann, obj_id=mapping[ann["obj_id"]]) for ann in frame]
        last_ids = {ann["obj_id"] for ann in remapped}
        if len(last_ids) != len(remapped):
            raise TrackingError("Chunk association created duplicate global IDs within a frame.")
        timestamp = (frame_offset + frame_index) / fps
        for ann in remapped:
            history = identity_history.setdefault(ann["obj_id"], [])
            history.append({"second": timestamp, "x": ann["cx"], "y": ann["cy"]})
            del history[:-2]
        remapped_annotations.append(remapped)

    global_crop_ids = set()
    for local_id in data["crops_saved"]:
        source = Path(chunk_crops) / f"penguin_crop_{local_id}.jpg"
        require_nonempty_file(source, f"Classification crop for local object {local_id}")
        global_id = mapping[local_id]
        destination = Path(global_crops) / f"penguin_crop_{global_id}.jpg"
        if not destination.exists():
            shutil.copyfile(source, destination)
        require_nonempty_file(destination, f"Classification crop for global object {global_id}")
        global_crop_ids.add(global_id)

    boundary = {global_id: list(identity_history[global_id]) for global_id in last_ids}
    return tracks, remapped_annotations, global_crop_ids, boundary, next_obj_id


def _extract_tracks(video_path, output_json, output_video, model_name, *, crop_dir, **kwargs):
    proxy_video = output_video + ".proxy.mp4"
    temp_output_video = str(Path(output_video).with_name("raw_render.mp4"))
    
    chunk_files = []
    temp_videos = []
    active_process = None
    
    if os.path.exists(proxy_video): os.remove(proxy_video)
    
    create_lightweight_proxy_video(video_path, proxy_video)
    
    print(f"Chunking video into {CHUNK_DURATION_SEC}-second segments...")
    chunk_pattern = proxy_video + "_chunk_%03d.mp4"
    run_ffmpeg([
        'ffmpeg', '-y', '-i', proxy_video,
        '-c', 'copy', '-map', '0', '-segment_time', str(CHUNK_DURATION_SEC), 
        '-reset_timestamps', '1',
        '-f', 'segment', chunk_pattern
    ], "Video segmentation")
    
    chunk_files = sorted(glob.glob(proxy_video + "_chunk_*.mp4"),
                         key=lambda path: int(Path(path).stem.rsplit("_", 1)[1]))
    if not chunk_files:
        raise TrackingError("Video segmentation produced no chunks.")
    fps, width, _ = validate_video(proxy_video, "Proxy video")
    for chunk_file in chunk_files:
        chunk_fps, _, _ = validate_video(chunk_file, "Video chunk")
        if not math.isclose(chunk_fps, fps, rel_tol=1e-6, abs_tol=1e-6):
            raise TrackingError("Video chunk frame rate differs from the constant-rate proxy.")
    
    max_dist_pixels = width * MAX_DIST_REL_WIDTH
    
    all_tracks = {}
    all_text_annotations = []
    crops_saved = set()
    
    print("Spawning isolated process for SAM 3 tracking for each chunk...")
    ctx = multiprocessing.get_context("spawn")
    
    next_obj_id = 0
    previous_boundary = {}
    identity_history = {}
    
    try:
        for i, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {i+1}/{len(chunk_files)}...")
            chunk_temp_vid = chunk_file + ".temp.mp4"
            chunk_temp_data = chunk_file + ".temp.data.json"
            temp_videos.append(chunk_temp_vid)
            
            if os.path.exists(chunk_temp_vid): os.remove(chunk_temp_vid)
            if os.path.exists(chunk_temp_data): os.remove(chunk_temp_data)
            
            chunk_crops = Path(crop_dir) / f"chunk-{i:03d}"
            chunk_crops.mkdir()
            # Workers own only chunk-local SAM IDs. Global identity belongs here.
            worker_args = (chunk_file, chunk_temp_vid, chunk_temp_data, 0, None, None, str(chunk_crops))
            if kwargs.get("sam3_config_path") is not None:
                worker_args += (os.fspath(kwargs["sam3_config_path"]),)
            p = ctx.Process(target=_run_sam3_subprocess, args=worker_args)
            active_process = p
            p.start()
            p.join()
            active_process = None
            
            data = _read_chunk_result(chunk_temp_data, p.exitcode, i + 1)
            _validate_chunk_telemetry(data, fps, i + 1)
            annotation_count = len(data["text_annotations"])
            validate_video(chunk_file, f"SAM 3 chunk {i+1} input", expected_frames=annotation_count)
            worker_fps, _, _ = validate_video(chunk_temp_vid, f"SAM 3 chunk {i+1} video", expected_frames=annotation_count)
            if not math.isclose(worker_fps, fps, rel_tol=1e-6, abs_tol=1e-6):
                raise TrackingError(f"SAM 3 chunk {i+1} video frame rate differs from the proxy.")
                
            tracks, annotations, crop_ids, previous_boundary, next_obj_id = _stitch_chunk(
                data, previous_boundary, identity_history, next_obj_id, len(all_text_annotations), fps,
                max_dist_pixels, chunk_crops, crop_dir,
            )
            for pid, samples in tracks.items():
                all_tracks.setdefault(pid, []).extend(samples)
            all_text_annotations.extend(annotations)
            crops_saved.update(crop_ids)

        all_tracks = recompute_velocities(all_tracks)
        print("Phase 3: Semantic Identification (SigLIP Batch Processing)...")
        semantic_id_map = {}
        species_counters = {}
        
        compact_annotation = kwargs.get("compact_annotation", False)
        species_to_compact_id = {}
        next_compact_id = 1
        
        crop_batch = []
        for obj_id in sorted(crops_saved):
            crop_path = os.path.join(crop_dir, f"penguin_crop_{obj_id}.jpg")
            require_nonempty_file(crop_path, f"Classification crop for object {obj_id}")
            crop_batch.append((crop_path, obj_id))
            
        if crop_batch:
            batch_results = get_penguin_species_batch(crop_batch)
            for obj_id, base_name in batch_results.items():
                if base_name == "NOT_PENGUIN":
                    semantic_id_map[int(obj_id)] = "NOT_PENGUIN"
                    semantic_id_map[str(obj_id)] = "NOT_PENGUIN"
                else:
                    if base_name not in species_counters:
                        species_counters[base_name] = 1
                    else:
                        species_counters[base_name] += 1
                    
                    if compact_annotation:
                        if base_name not in species_to_compact_id:
                            species_to_compact_id[base_name] = next_compact_id
                            next_compact_id += 1
                        final_name = f"{species_to_compact_id[base_name]}_{species_counters[base_name]}"
                    else:
                        final_name = f"{base_name}_{species_counters[base_name]}"
                        
                    semantic_id_map[int(obj_id)] = final_name
                    semantic_id_map[str(obj_id)] = final_name
                    
            if compact_annotation and species_to_compact_id:
                mapping_file = output_json.replace(".json", "_species_mapping.txt")
                with open(mapping_file, "w") as f:
                    f.write("Compact Annotation Species Mapping:\n")
                    for s_name, c_id in species_to_compact_id.items():
                        f.write(f"{c_id}: {s_name}\n")
                print(f"Saved species mapping to {mapping_file}")
            
        print("Phase 4: Final Video Rendering...")
        if temp_videos:
            render_final_video(temp_videos, all_text_annotations, semantic_id_map, temp_output_video, output_video, fps)
            
        print("Remapping tracking data...")
        final_tracks = {}
        for temp_pid, data_list in all_tracks.items():
            try:
                obj_id = int(temp_pid.split("_")[1])
                real_name = semantic_id_map.get(obj_id, temp_pid)
                if real_name != "NOT_PENGUIN":
                    final_tracks[real_name] = data_list
            except Exception:
                final_tracks[temp_pid] = data_list
        
        validate_tracks(final_tracks)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_tracks, f, indent=4, allow_nan=False)
            
        return output_json, output_video

    finally:
        if active_process is not None and active_process.is_alive():
            print("\n Interrupt detected: Terminating orphaned SAM 3 subprocess...")
            active_process.terminate()
            active_process.join(timeout=5)
            if active_process.is_alive():
                active_process.kill()
                active_process.join()
            print("Subprocess terminated. VRAM freed.")
            
        print("Cleaning up temporary disk files...")
        for f in temp_videos:
            if os.path.exists(f): os.remove(f)
        for f in chunk_files:
            if os.path.exists(f): os.remove(f)
            data_f = f + ".temp.data.json"
            if os.path.exists(data_f): os.remove(data_f)
        if os.path.exists(proxy_video):
            os.remove(proxy_video)
        if os.path.exists(temp_output_video):
            os.remove(temp_output_video)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_tracks(sys.argv[1])
