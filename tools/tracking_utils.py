import subprocess
from PIL import Image
from tools.tracking_config import TARGET_FPS, TARGET_HEIGHT, CHUNK_DURATION_SEC

def create_lightweight_proxy_video(input_path, output_path, target_fps=TARGET_FPS, target_height=TARGET_HEIGHT):
    """
    Creates a lightweight version of the video for SAM using FFmpeg to save VRAM.
    """
    print(f"Optimizing video to save VRAM (max height: {target_height}p, {target_fps} FPS)...")
    
    gop_size = target_fps * CHUNK_DURATION_SEC 
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f"scale=-2:'min({target_height},ih)'", 
        '-r', str(target_fps),
        '-g', str(gop_size), 
        '-keyint_min', str(gop_size),
        '-sc_threshold', '0',
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Proxy video ready.")

def pad_to_square(image):
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (128, 128, 128))
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image
