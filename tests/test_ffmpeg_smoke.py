"""Real CPU encoder check for fractional FPS, with no model weights or OpenCV."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import tracking_utils


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe not installed")
class FfmpegSmokeTests(unittest.TestCase):
    def test_fractional_proxy_and_short_segments_keep_all_frames(self):
        with tempfile.TemporaryDirectory(prefix="penguin video ") as temporary:
            root = Path(temporary)
            source, proxy = root / "source.mp4", root / "proxy.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=black:s=160x120:r=5:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True, capture_output=True)
            # Decoding/model checks have their own tests; inspect real encoded
            # metadata here through ffprobe so this test stays CPU-only.
            with mock.patch.object(tracking_utils, "validate_video"), mock.patch.object(tracking_utils, "CHUNK_DURATION_SEC", .8):
                tracking_utils.create_lightweight_proxy_video(source, proxy, target_fps=2.5, target_height=96)
            pattern = str(root / "chunk-%03d.mp4")
            tracking_utils.run_ffmpeg([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(proxy), "-c", "copy",
                "-segment_time", "0.8", "-reset_timestamps", "1", "-f", "segment", pattern,
            ], "Test segmentation")

            def metadata(path):
                completed = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=avg_frame_rate,nb_read_frames,height", "-of", "json", str(path),
                ], check=True, capture_output=True, text=True)
                return json.loads(completed.stdout)["streams"][0]

            info = metadata(proxy)
            self.assertEqual(info["avg_frame_rate"], "5/2")
            self.assertEqual(info["height"], 96)
            self.assertEqual(int(info["nb_read_frames"]), 5)
            chunks = sorted(root.glob("chunk-*.mp4"))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(sum(int(metadata(path)["nb_read_frames"]) for path in chunks), 5)


if __name__ == "__main__":
    unittest.main()
