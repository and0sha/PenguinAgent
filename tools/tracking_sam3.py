from tools.tracking_config import MAX_DIST_REL_WIDTH

def _run_sam3_subprocess(video_path, temp_video, temp_data, next_obj_id=0, prev_points=None, prev_trajectories=None):
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    import gc
    import numpy as np
    import cv2
    import json
    
    print(f"Analyzing {video_path} with SAM 3 ...")
    
    import sys
    sam3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam3"))
    if sam3_dir not in sys.path:
        sys.path.insert(0, sam3_dir)
        
    from sam3.model_builder import build_sam3_video_predictor
    bpe_path = os.path.join(sam3_dir, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_dist_pixels = width * MAX_DIST_REL_WIDTH
    
    frames = []
    print("Loading video frames into RAM for large video support...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        frame_idx += 1
    cap.release()

    predictor = build_sam3_video_predictor(bpe_path=bpe_path, async_loading_frames=False)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

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
    
    tracks = {}
    trajectory_history = {}
    
    frame_idx = 0
    active_penguins = {}
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,165,0)]
    
    text_annotations = []
    crops_saved = set()
    
    local_mapping = {}
    if prev_points is None: prev_points = {}
    if prev_trajectories is None: prev_trajectories = {}
    
    print("Phase 1: Tracking and Mask Generation (SAM 3)...")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for response in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
            out = response["outputs"]
            obj_ids = out.get('out_obj_ids', [])
            masks = out.get('out_binary_masks', [])
            
            frame = frames[frame_idx]
            overlay = frame.copy()
            current_frame_penguins = {}
            frame_annotations = []
            
            if frame_idx == 0 and prev_points:
                curr_points = {}
                for idx, raw_obj_id in enumerate(obj_ids):
                    mask = masks[idx]
                    if torch.is_tensor(mask): mask = mask.cpu().numpy()
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        curr_points[int(raw_obj_id)] = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                
                if curr_points:
                    from scipy.optimize import linear_sum_assignment
                    prev_pids = list(prev_points.keys())
                    curr_raw_ids = list(curr_points.keys())
                    
                    cost_matrix = np.zeros((len(prev_pids), len(curr_raw_ids)))
                    for r, pp_id in enumerate(prev_pids):
                        px, py = prev_points[pp_id]
                        for c, c_raw_id in enumerate(curr_raw_ids):
                            cx, cy = curr_points[c_raw_id]
                            cost_matrix[r, c] = np.hypot(px - cx, py - cy)
                            
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] < max_dist_pixels:
                            local_mapping[curr_raw_ids[c]] = int(prev_pids[r].split('_')[1])
                            
            for idx, raw_obj_id in enumerate(obj_ids):
                raw_id = int(raw_obj_id)
                if raw_id in local_mapping:
                    obj_id = local_mapping[raw_id]
                else:
                    obj_id = next_obj_id
                    local_mapping[raw_id] = obj_id
                    next_obj_id += 1
                    
                mask = masks[idx]
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                mask = mask.astype(bool)
                
                if int(obj_id) not in crops_saved:
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        y1, y2 = max(0, min(y_coords)-15), min(frame.shape[0], max(y_coords)+15)
                        x1, x2 = max(0, min(x_coords)-15), min(frame.shape[1], max(x_coords)+15)
                        crop = frame[y1:y2, x1:x2]
                        temp_path = f"/tmp/penguin_crop_{obj_id}.jpg"
                        cv2.imwrite(temp_path, crop)
                        crops_saved.add(int(obj_id))
                
                pid = f"obj_{obj_id}"
                    
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0: continue
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                area = len(y_coords)
                
                if pid not in trajectory_history:
                    if pid in prev_trajectories:
                        trajectory_history[pid] = list(prev_trajectories[pid])
                    else:
                        trajectory_history[pid] = []
                
                current_frame_penguins[pid] = (cx, cy)
                trajectory_history[pid].append((cx, cy))
                
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
                
                if frame_idx % int(fps) == 0:
                    second = frame_idx // int(fps)
                    vx, vy = 0, 0
                    if pid in active_penguins:
                        vx = cx - active_penguins[pid][0]
                        vy = cy - active_penguins[pid][1]
                        
                    if pid not in tracks:
                        tracks[pid] = []
                        
                    tracks[pid].append({
                        "second": second,
                        "x": cx,
                        "y": cy,
                        "area": float(area),
                        "vx": float(vx),
                        "vy": float(vy)
                    })
            
            text_annotations.append(frame_annotations)
            active_penguins = current_frame_penguins
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            out_temp.write(frame)
            frame_idx += 1
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    out_temp.release()
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    with open(temp_data, 'w') as f:
        json.dump({
            "tracks": tracks,
            "text_annotations": text_annotations,
            "crops_saved": list(crops_saved)
        }, f)
        
    print("Phase 2: SAM 3 process completed.")