import cv2

from tools.tracking_results import TrackingError
from tools.tracking_utils import run_ffmpeg, validate_video


def render_final_video(temp_videos, all_text_annotations, semantic_id_map, temp_output_video, output_video, fps):
    if not temp_videos or not all_text_annotations:
        raise TrackingError("Final rendering requires worker videos and frame annotations.")
    _, width, height = validate_video(temp_videos[0], "First worker video")
    out_vid = cv2.VideoWriter(str(temp_output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_idx = 0
    try:
        if not out_vid.isOpened():
            raise TrackingError(f"Cannot open the final video writer at {temp_output_video}.")
        for chunk_temp_vid in temp_videos:
            chunk_fps, chunk_width, chunk_height = validate_video(chunk_temp_vid, "Worker video")
            if (chunk_width, chunk_height) != (width, height) or abs(chunk_fps - fps) > 0.01:
                raise TrackingError(f"Worker video dimensions or frame rate differ: {chunk_temp_vid}.")
            cap_temp = cv2.VideoCapture(str(chunk_temp_vid))
            try:
                if not cap_temp.isOpened():
                    raise TrackingError(f"Cannot open worker video {chunk_temp_vid}.")
                while True:
                    ret, frame = cap_temp.read()
                    if not ret:
                        break
                    if frame_idx >= len(all_text_annotations):
                        raise TrackingError("Worker video contains more frames than its annotation data.")
                    for ann in all_text_annotations[frame_idx]:
                        obj_id = ann["obj_id"]
                        name = semantic_id_map.get(obj_id, f"obj_{obj_id}")
                        if name != "NOT_PENGUIN":
                            cv2.putText(frame, name, (ann["cx"], ann["cy"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    out_vid.write(frame)
                    frame_idx += 1
            finally:
                cap_temp.release()
    finally:
        out_vid.release()

    if frame_idx != len(all_text_annotations):
        raise TrackingError(f"Worker videos contain {frame_idx} frames for {len(all_text_annotations)} annotations.")
    validate_video(temp_output_video, "Uncompressed final render", expected_frames=frame_idx)
    print("Converting video for web compatibility (H.264)...")
    run_ffmpeg([
        'ffmpeg', '-y', '-i', str(temp_output_video), '-vcodec', 'libx264', '-crf', '23',
        '-preset', 'fast', '-movflags', '+faststart', '-pix_fmt', 'yuv420p', str(output_video),
    ], "Final H.264 encoding")
    validate_video(output_video, "Final H.264 encoding", expected_frames=frame_idx)
