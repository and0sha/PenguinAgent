import os
import glob
import json
import ollama
import torch
import gc

class AnalysisMixin:
    def analyze_behavior(self, specific_id=None, query=None, output_dir=None):
        if not getattr(self, 'tracks', None):
            return "No tracks available. Please run tracking first."
            
        summary = "Penguin tracking summary:\n"
        additional_context = ""
        
        if specific_id:
            # Dynamically gather context about this penguin from all reports and logs
            import glob
            context_files = []
            if output_dir and os.path.exists(output_dir):
                context_files.extend(glob.glob(os.path.join(output_dir, "*.md")))
                context_files.extend(glob.glob(os.path.join(output_dir, "*.json")))
            else:
                context_files.extend(glob.glob("*.md"))
                context_files.extend(glob.glob("*.json"))
                context_files.extend(glob.glob(os.path.join(self.videos_dir, "*.json")))
                context_files.extend(glob.glob("sam3/agent_output/**/*.json", recursive=True))
                context_files.extend(glob.glob("sam3/agent_output/**/*.md", recursive=True))
            
            extracted_info = []
            for filepath in set(context_files):
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            if specific_id in content:
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if specific_id in line:
                                        start = max(0, i - 10)
                                        end = min(len(lines), i + 10)
                                        chunk = "\n".join(lines[start:end])
                                        chunk_info = f"--- Context from {filepath} ---\n{chunk}\n"
                                        if chunk_info not in extracted_info:
                                            extracted_info.append(chunk_info)
                    except Exception:
                        pass
            
            if extracted_info:
                additional_context = f"\nHistorical Context & Logs for {specific_id} from previous models and reports:\n"
                additional_context += "\n".join(extracted_info)[:12000] # Limit to avoid exceeding context window
                
            if specific_id in self.tracks:
                data = self.tracks[specific_id]
                summary += f"- {specific_id} tracked for {len(data)} seconds. Velocity vectors varying.\n"
        else:
            for pid, data in self.tracks.items():
                summary += f"- {pid} tracked for {len(data)} seconds.\n"
            
        if query and specific_id:
            prompt = f"You are analyzing data for penguin {specific_id}.\n{additional_context}\nData Summary:\n{summary}\nUser Query: {query}"
        else:
            prompt = f"Analyze the following penguin tracking data and provide a brief 3x100 anomaly score (Uniqueness, Anormality, Rarity). Data:\n{summary}\n{additional_context}"
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': '<|think|> You are PenguinAgent, an autonomous ornithology AI running on Gemma 4.'},
                {'role': 'user', 'content': prompt}
            ], options={"num_ctx": 8192}, keep_alive=0)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with Ollama: {e}"