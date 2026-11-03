def _run_sam3_subprocess(video_path, temp_video, temp_data, next_obj_id=0, prev_points=None, prev_trajectories=None, crop_dir=None, sam3_config_path=None):
    """Return chunk-local SAM IDs; legacy global-ID arguments are ignored.

    Global identity and velocities are owned by the parent after chunk stitching.
    """
    from contextlib import ExitStack
    import json
    from pathlib import Path
    import tempfile
    from tools.tracking_results import TrackingError
    del next_obj_id, prev_points, prev_trajectories

    try:
        if crop_dir is None:
            # Direct legacy callers receive this directory in the result and own cleanup.
            crop_dir = tempfile.mkdtemp(prefix="penguin-crops-", dir=Path(temp_data).resolve().parent)
        Path(crop_dir).mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            _run_sam3_impl(video_path, temp_video, temp_data, str(crop_dir), resources, sam3_config_path)
    except Exception as exc:
        detail = f"SAM 3 tracking failed: {type(exc).__name__}: {exc}"
        try:
            Path(temp_data).write_text(json.dumps({"status": "error", "error": detail}), encoding="utf-8")
        except OSError:
            pass
        if isinstance(exc, TrackingError):
            raise
        raise TrackingError(detail) from exc


def _run_sam3_impl(video_path, temp_video, temp_data, crop_dir, resources, sam3_config_path=None):
    import os
    import sys
    from tools.sam3_runtime import resolve_sam3_runtime
    from tools.tracking_results import TrackingError

    runtime = resolve_sam3_runtime() if sam3_config_path is None else resolve_sam3_runtime(sam3_config_path)
    from tools.tracking_utils import require_nonempty_file, validate_video
    sam3_dir = str(runtime.source_root)
    if sam3_dir not in sys.path:
        sys.path.insert(0, sam3_dir)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from collections import deque
    import numpy as np
    import cv2
    import json
    
    print(f"Analyzing {video_path} with SAM 3 ...")
    
    from sam3.model_builder import build_sam3_video_predictor

    fps, width, height = validate_video(video_path, "SAM 3 input")
    cap = cv2.VideoCapture(video_path)
    resources.callback(cap.release)
    if not cap.isOpened():
        raise TrackingError(f"Cannot open SAM 3 input {video_path}.")
    expected_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if expected_frame_count <= 0:
        raise TrackingError("SAM 3 input has no reliable frame count.")
    
    frames = []
    print("Loading video frames into RAM for large video support...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        frame_idx += 1
    cap.release()
    if not frames:
        raise TrackingError("SAM 3 input has no decoded frames.")
    if len(frames) != expected_frame_count:
        raise TrackingError(f"SAM 3 decoded {len(frames)} input frames; expected {expected_frame_count}.")

    predictor = build_sam3_video_predictor(**runtime.builder_kwargs())
    if hasattr(predictor, "shutdown"):
        resources.callback(predictor.shutdown)
    resources.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))

    response = predictor.handle_request(request=dict(
        type="start_session", 
        resource_path=video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True
    ))
    session_id = response["session_id"]
    
    predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="penguin"))
    
    fourcc_temp = cv2.VideoWriter_fourcc(*'mp4v')
    out_temp = cv2.VideoWriter(temp_video, fourcc_temp, fps, (width, height))
    resources.callback(out_temp.release)
    if not out_temp.isOpened():
        raise TrackingError(f"Cannot open SAM 3 video writer at {temp_video}.")
    
    tracks = {}
    trajectory_history = {}
    
    frame_idx = 0
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,165,0)]
    
    text_annotations = []
    crops_saved = set()
    
    print("Phase 1: Tracking and Mask Generation (SAM 3)...")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for response in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
            if frame_idx >= len(frames) or response.get("frame_index") != frame_idx:
                raise TrackingError("SAM 3 returned unexpected or out-of-order video frames.")
            out = response["outputs"]
            obj_ids = out.get('out_obj_ids', [])
            masks = out.get('out_binary_masks', [])
            if len(obj_ids) != len(masks):
                raise TrackingError("SAM 3 returned different object-ID and mask counts.")
            frame_ids = [int(raw_id) for raw_id in obj_ids]
            if len(frame_ids) != len(set(frame_ids)) or any(raw_id < 0 for raw_id in frame_ids):
                raise TrackingError("SAM 3 returned duplicate or invalid local IDs in a frame.")
            
            frame = frames[frame_idx]
            overlay = frame.copy()
            frame_annotations = []

            for idx, raw_obj_id in enumerate(obj_ids):
                obj_id = int(raw_obj_id)
                    
                mask = masks[idx]
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                mask = mask.astype(bool)
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    continue
                
                if int(obj_id) not in crops_saved:
                    y1, y2 = max(0, min(y_coords)-15), min(frame.shape[0], max(y_coords)+15)
                    x1, x2 = max(0, min(x_coords)-15), min(frame.shape[1], max(x_coords)+15)
                    crop = frame[y1:y2, x1:x2]
                    temp_path = os.path.join(crop_dir, f"penguin_crop_{obj_id}.jpg")
                    if not os.path.exists(temp_path) and not cv2.imwrite(temp_path, crop):
                        raise TrackingError(f"Could not save classification crop {temp_path}.")
                    require_nonempty_file(temp_path, "Classification crop")
                    crops_saved.add(int(obj_id))
                
                pid = f"obj_{obj_id}"
                    
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                area = len(y_coords)
                
                # Telemetry keeps every observation; the visual trail has bounded
                # drawing cost even when the proxy FPS/chunk duration is increased.
                trajectory_history.setdefault(pid, deque(maxlen=30)).append((cx, cy))
                
                color = colors[int(obj_id) % len(colors)]
                overlay[mask] = color
                
                pts = trajectory_history[pid]
                if len(pts) > 1:
                    for j in range(1, len(pts)):
                        cv2.line(frame, pts[j-1], pts[j], color, 2)
                
                frame_annotations.append({
                    "obj_id": int(obj_id),
                    "cx": cx,
                    "cy": cy
                })
                
                # Keep each proxy-frame observation, including a new animal's first
                # subsecond appearance. The parent computes dx/dt after stitching.
                tracks.setdefault(pid, []).append({
                    "second": frame_idx / fps,
                    "x": cx,
                    "y": cy,
                    "area": float(area),
                    "vx": 0.0,
                    "vy": 0.0,
                })
            
            text_annotations.append(frame_annotations)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            out_temp.write(frame)
            frame_idx += 1
            
        
    out_temp.release()
    if frame_idx != len(frames):
        raise TrackingError(f"SAM 3 returned {frame_idx} frames for {len(frames)} input frames.")
    validate_video(temp_video, "SAM 3 worker video", expected_frames=len(frames))
    # ExitStack shuts down the predictor and the process releases its CUDA state.
    # Per-frame empty_cache() would synchronize CUDA and defeat allocator reuse.
        
    with open(temp_data, 'w', encoding='utf-8') as f:
        json.dump({
            "status": "ok",
            "tracks": tracks,
            "text_annotations": text_annotations,
            "crops_saved": list(crops_saved),
            "crop_dir": crop_dir,
        }, f)
        
    print("Phase 2: SAM 3 process completed.")
