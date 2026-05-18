import cv2
import subprocess

def render_final_video(temp_videos, all_text_annotations, semantic_id_map, temp_output_video, output_video, fps):
    cap_temp = cv2.VideoCapture(temp_videos[0])
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(temp_output_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    for chunk_temp_vid in temp_videos:
        cap_temp = cv2.VideoCapture(chunk_temp_vid)
        while True:
            ret, frame = cap_temp.read()
            if not ret: break
            
            if frame_idx < len(all_text_annotations):
                for ann in all_text_annotations[frame_idx]:
                    obj_id = ann["obj_id"]
                    name = semantic_id_map.get(obj_id, f"obj_{obj_id}")
                    if name != "NOT_PENGUIN":
                        cv2.putText(frame, name, (ann["cx"], ann["cy"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
            out_vid.write(frame)
            frame_idx += 1
            
        cap_temp.release()
    out_vid.release()
    
    print("Converting video for web compatibility (H.264)...")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', temp_output_video,
        '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
        '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
        output_video
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
