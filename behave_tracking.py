import os
import json
from tools.feature_tracking import extract_tracks

class TrackingMixin:
    def _load_and_remap_tracks(self, json_path, output_dir):
       
        if not os.path.exists(json_path):
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_tracks = json.load(f)

        mapping_path = os.path.join(output_dir, "tracks_species_mapping.txt")
        if not os.path.exists(mapping_path):
            self.tracks = raw_tracks
            return self.tracks

        species_mapping = {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        key_part, value_part = line.split(":", 1)
                        key_part = key_part.strip()
                        if key_part.isdigit():
                            species_mapping[key_part] = value_part.strip()
        except Exception as e:
            print(f"Warning: Failed to parse species mapping file: {e}")
            self.tracks = raw_tracks
            return self.tracks

        
        remapped_tracks = {}
        for old_key, data_list in raw_tracks.items():
            if "_" in old_key:
                class_id, instance_id = old_key.split("_", 1)
                if class_id in species_mapping:
                    new_key = f"{species_mapping[class_id]}_{instance_id}"
                else:
                    new_key = old_key
            else:
                if old_key in species_mapping:
                    new_key = f"{species_mapping[old_key]}_1"
                else:
                    new_key = old_key
            
            remapped_tracks[new_key] = data_list

        self.tracks = remapped_tracks
        print(f"Successfully remapped {len(self.tracks)} tracks using species mapping.")
        return self.tracks

    def run_tracking(self, video_name, output_dir=None, compact_annotation=False):
        video_path = os.path.join(self.videos_dir, video_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video {video_path} not found.")

        if output_dir is None:
            output_dir = os.path.join(self.videos_dir, os.path.splitext(video_name)[0] + "_output")
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "tracks.json")
        annotated_video_path = os.path.join(output_dir, "annotated.mp4")
        extract_tracks(video_path, json_path, annotated_video_path, model_name=self.model, compact_annotation=compact_annotation)        
        self.tracks = self._load_and_remap_tracks(json_path, output_dir)
        return self.tracks, annotated_video_path