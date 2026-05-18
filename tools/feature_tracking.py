import os
import sys
import glob
import multiprocessing
import subprocess
import json
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from tools.tracking_config import CHUNK_DURATION_SEC, MAX_DIST_REL_WIDTH
from tools.tracking_utils import create_lightweight_proxy_video
from tools.tracking_sam3 import _run_sam3_subprocess
from tools.tracking_siglip import get_penguin_species_batch
from tools.tracking_render import render_final_video

def extract_tracks(video_path, output_json="tracks.json", output_video="annotated.mp4", model_name="gemma4:26b", **kwargs):
    proxy_video = output_video + ".proxy.mp4"
    temp_output_video = output_video.replace(".mp4", "_temp_raw.mp4")
    
    chunk_files = []
    temp_videos = []
    active_process = None
    
    if os.path.exists(proxy_video): os.remove(proxy_video)
    
    create_lightweight_proxy_video(video_path, proxy_video)
    
    print(f"Chunking video into {CHUNK_DURATION_SEC}-second segments...")
    chunk_pattern = proxy_video + "_chunk_%03d.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-i', proxy_video,
        '-c', 'copy', '-map', '0', '-segment_time', str(CHUNK_DURATION_SEC), 
        '-reset_timestamps', '1',
        '-f', 'segment', chunk_pattern
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    chunk_files = sorted(glob.glob(proxy_video + "_chunk_*.mp4"))
    
    cap_proxy = cv2.VideoCapture(proxy_video)
    fps = cap_proxy.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap_proxy.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_proxy.release()
    
    max_dist_pixels = width * MAX_DIST_REL_WIDTH
    
    all_tracks = {}
    all_text_annotations = []
    crops_saved = set()
    
    print("Spawning isolated process for SAM 3 tracking for each chunk...")
    ctx = multiprocessing.get_context("spawn")
    
    next_obj_id = 0
    prev_points = {}
    global_trajectories = {}
    
    try:
        for i, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {i+1}/{len(chunk_files)}...")
            chunk_temp_vid = chunk_file + ".temp.mp4"
            chunk_temp_data = chunk_file + ".temp.data.json"
            temp_videos.append(chunk_temp_vid)
            
            if os.path.exists(chunk_temp_vid): os.remove(chunk_temp_vid)
            if os.path.exists(chunk_temp_data): os.remove(chunk_temp_data)
            
            p = ctx.Process(target=_run_sam3_subprocess, args=(chunk_file, chunk_temp_vid, chunk_temp_data, next_obj_id, prev_points, global_trajectories))
            active_process = p
            p.start()
            p.join()
            active_process = None
            
            if p.exitcode != 0 or not os.path.exists(chunk_temp_data):
                print(f"SAM 3 Subprocess failed or crashed on chunk {i+1}. Tracking aborted.")
                return None, None
                
            with open(chunk_temp_data, 'r') as f:
                data = json.load(f)
                
            frame_offset = len(all_text_annotations)
            second_offset = frame_offset // int(fps)
            
            if i > 0 and len(all_tracks) > 0 and len(data["tracks"]) > 0:
                max_second_prev = -1
                for pid, t_list in all_tracks.items():
                    if len(t_list) > 0:
                        max_second_prev = max(max_second_prev, t_list[-1]["second"])
                
                prev_points = {}
                for pid, t_list in all_tracks.items():
                    if len(t_list) > 0 and t_list[-1]["second"] >= max_second_prev - 2:
                        prev_points[pid] = (t_list[-1]["x"], t_list[-1]["y"])
                        
                curr_points = {}
                for pid, t_list in data["tracks"].items():
                    if len(t_list) > 0 and t_list[0]["second"] <= 2:
                        curr_points[pid] = (t_list[0]["x"], t_list[0]["y"])
                        
                if prev_points and curr_points:
                    prev_pids = list(prev_points.keys())
                    curr_pids = list(curr_points.keys())
                    
                    cost_matrix = np.zeros((len(prev_pids), len(curr_pids)))
                    for r, pp_id in enumerate(prev_pids):
                        px, py = prev_points[pp_id]
                        for c, cp_id in enumerate(curr_pids):
                            cx, cy = curr_points[cp_id]
                            cost_matrix[r, c] = np.hypot(px - cx, py - cy)
                            
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    mapping = {}
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] < max_dist_pixels:
                            mapping[curr_pids[c]] = prev_pids[r]
                    
                    new_tracks = {}
                    for pid, t_list in data["tracks"].items():
                        mapped_pid = mapping.get(pid, pid)
                        if mapped_pid not in new_tracks:
                            new_tracks[mapped_pid] = []
                        new_tracks[mapped_pid].extend(t_list)
                    data["tracks"] = new_tracks
                    
                    for frame_ann in data["text_annotations"]:
                        for ann in frame_ann:
                            pid_str = f"obj_{ann['obj_id']}"
                            if pid_str in mapping:
                                ann["obj_id"] = int(mapping[pid_str].split("_")[1])
                                
                    new_crops_saved = set()
                    for obj_id in data["crops_saved"]:
                        pid_str = f"obj_{obj_id}"
                        if pid_str in mapping:
                            new_obj_id = int(mapping[pid_str].split("_")[1])
                            old_crop = f"/tmp/penguin_crop_{obj_id}.jpg"
                            new_crop = f"/tmp/penguin_crop_{new_obj_id}.jpg"
                            if os.path.exists(old_crop) and not os.path.exists(new_crop):
                                os.rename(old_crop, new_crop)
                            new_crops_saved.add(new_obj_id)
                        else:
                            new_crops_saved.add(obj_id)
                    data["crops_saved"] = list(new_crops_saved)

            for pid, t_list in data["tracks"].items():
                if pid not in all_tracks:
                    all_tracks[pid] = []
                for item in t_list:
                    item["second"] += second_offset
                    all_tracks[pid].append(item)
                    
                obj_id_int = int(pid.split("_")[1])
                if obj_id_int >= next_obj_id:
                    next_obj_id = obj_id_int + 1
                    
            all_text_annotations.extend(data["text_annotations"])
            crops_saved.update(data["crops_saved"])
            
            prev_points = {}
            global_trajectories = {}
            
            if len(all_tracks) > 0:
                max_second_prev = -1
                for pid, t_list in all_tracks.items():
                    if len(t_list) > 0:
                        max_second_prev = max(max_second_prev, t_list[-1]["second"])
                
                for pid, t_list in all_tracks.items():
                    if len(t_list) > 0 and t_list[-1]["second"] >= max_second_prev - 2:
                        prev_points[pid] = (t_list[-1]["x"], t_list[-1]["y"])
                        
                for pid, t_list in all_tracks.items():
                    traj = [(pt["x"], pt["y"]) for pt in t_list[-15:]]
                    global_trajectories[pid] = traj

        print("Phase 3: Semantic Identification (SigLIP Batch Processing)...")
        semantic_id_map = {}
        species_counters = {}
        
        compact_annotation = kwargs.get("compact_annotation", False)
        species_to_compact_id = {}
        next_compact_id = 1
        
        crop_batch = []
        for obj_id in crops_saved:
            crop_path = f"/tmp/penguin_crop_{obj_id}.jpg"
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
        
        with open(output_json, 'w') as f:
            json.dump(final_tracks, f, indent=4)
            
        print(f"Tracking complete. Data saved to {output_json} and {output_video}.")
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
