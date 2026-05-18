import os
import json
import gc
import torch
import ollama
import numpy as np

class HuddleMixin:
    def generate_huddle_paper(self, video_name, output_dir=None, pe=None, beta=None, tp=None, t_inf=None):
        video_path = os.path.abspath(os.path.join(self.videos_dir, video_name))
        if output_dir is None:
            output_dir = os.path.join(self.videos_dir, os.path.splitext(video_name)[0] + "_output")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "tracks.json")

        if getattr(self, 'tracks', None) is None:
            if os.path.exists(json_path):
                self.tracks = self._load_and_remap_tracks(json_path, output_dir)
            else:
                return "No tracks available. Please run tracking first."

        summary = f"Total penguins tracked: {len(self.tracks)}\n"
        for pid, data in self.tracks.items():
            summary += f"- {pid}: tracked for {len(data)} seconds.\n"

        env_context = ""
        if pe is not None and beta is not None and tp is not None and t_inf is not None:
            env_context = f"\nUser-Provided Environmental Boundary Conditions:\n"
            env_context += f"- Péclet number ($Pe$): {pe} (Wind strength)\n"
            env_context += f"- Noise level ($\\beta$): {beta} (Random perturbation)\n"
            env_context += f"- Temperature at huddle boundary ($T_p$): {tp} °C\n"
            env_context += f"- Ambient temperature away from huddle ($T_\\infty$): {t_inf} °C\n"

       

        generated_charts = []
        def generate_custom_chart(python_code: str) -> str:
    
            try:
                print("📊 Gemma 4 is generating a custom chart...")
                import matplotlib.pyplot as plt
                import glob
                
                chart_filename = f"huddle_chart_{len(generated_charts) + 1}.png"
                output_chart_path = os.path.join(output_dir, chart_filename)
                
                local_vars = {
                    'tracking_data': self.tracks,
                    'output_chart_path': output_chart_path,
                    'video_path': video_path,
                    'plt': plt,
                    'np': np,
                    'json': json,
                    'os': os
                }
                
                pngs_before = set(glob.glob(os.path.join(output_dir, "*.png")))
                
                original_cwd = os.getcwd()
                os.chdir(output_dir)
                try:
                    exec(python_code, {}, local_vars)
                finally:
                    plt.close('all')
                    os.chdir(original_cwd)
                
                pngs_after = set(glob.glob(os.path.join(output_dir, "*.png")))
                new_pngs = list(pngs_after - pngs_before)
                
                if new_pngs:
                    for npng in new_pngs:
                        if npng not in generated_charts:
                            generated_charts.append(npng)
                    return f"SUCCESS: Chart successfully generated and saved."
                elif os.path.exists(output_chart_path):
                    if output_chart_path not in generated_charts:
                        generated_charts.append(output_chart_path)
                    return f"SUCCESS: Chart successfully generated and saved to {output_chart_path}."
                else:
                    return "ERROR: The code executed, but no file was found at 'output_chart_path'. Ensure you used plt.savefig(output_chart_path)."
            except Exception as e:
                return f"ERROR: Chart generation failed with exception: {e}"

        prompt = f'''
You are an applied mathematician and ethologist running on Gemma 4. Write a scientific paper analyzing the provided penguin tracking data, strictly following the mathematical methodology of 'Modeling Huddling Penguins' (Waters, Blanchette, Kim, 2012).

Here is the tracking summary:
{summary}
{env_context}
Your task is to analyze this data through the lens of the Waters 2012 model.
CRITICAL INSTRUCTIONS:
1. You MUST use the `generate_custom_chart` tool to create plot: a graph of the penguin movement dynamics in the huddle based on the provided tracking data, illustrating the trajectories, velocities, or overall flow of the penguins.

Structure the report EXACTLY as follows:
# Abstract
Briefly summarize the observation of the huddle, the estimated wind conditions, and how individual movements contributed to the macroscopic huddle shape based on the Waters 2012 model.

# 1. Introduction
Discuss the biological imperative of huddling (minimizing cold-exposed body surface area). Mention the fundamental hypothesis: equal access to warmth is achieved through the selfish behavior of individuals minimizing their own heat loss $H(p)$.

# 2. Methods: Kinematic to Thermodynamic Mapping
Explain the mathematical model. Mention the 2D inviscid irrotational flow (Schwarz-Christoffel transformation $z = sc(\\\\zeta)$ and Joukowsky transform $J(\\\\zeta) = U(\\\\zeta + 1/\\\\zeta)$) and the advection-diffusion equation for temperature $\\nabla^2 T' = Pe \\, \\mathbf{{u}}' \\cdot \\nabla T'$ which yields the turbulent Péclet number $Pe = UL/D$.
Explain how you inferred the wind flow and local heat loss based on the tracking data.

# 3. Results and Discussion
- **Huddle Shape & Wind (Deterministic Case):** Discuss the Aspect Ratio (Width/Length) of the observed huddle. Relate it to the theoretical scaling law derived in the paper: $\\delta/L \\sim Pe^{{-1/3}}$. Given the Aspect Ratio calculated by the tool, estimate the Péclet number $Pe$ and discuss the wind strength.
- **Effects of Random Perturbations:** Analyze the noise ($\\beta$) in the trajectories. Discuss how a purely deterministic model ($\\beta=0$) yields overly elongated huddles, while realistic huddles require a noise level around $\\beta \\approx 0.25$ to match observations.
- **Equal Access:** Discuss if the "movers" successfully rotated from the windward to the leeward side, achieving an even distribution of heat loss.

# 4. Conclusion
Summarize the findings. Note that huddling dynamics can be accurately modeled without assuming altruism.

Use formal academic language. Output ONLY the markdown report.
'''
        try:
            
            messages = [{'role': 'system', 'content': '<|think|> You are an autonomous AI generating a scientific report on Huddling Penguins based on Waters 2012. Use tools to calculate thermodynamics and generate charts.'}]
            messages.append({'role': 'user', 'content': prompt})
            
            max_turns = 3
            report_content = ""
            
            for turn in range(max_turns):
                print(f"Turn {turn+1}: Sending prompt to Gemma 4 for Huddling Report...")
                response = ollama.chat(
                    model=self.model, 
                    messages=messages, 
                    options={"num_ctx": 16384},
                    tools=[generate_custom_chart],
                    keep_alive=0 if turn == max_turns - 1 else "5m"
                )
                
                if response.get("message", {}).get("tool_calls"):
                    print("Gemma 4 decided to use tools!")
                    messages.append(response['message'])
                    
                    for tool in response['message']['tool_calls']:
                        func_name = tool['function']['name']
                        args = tool['function']['arguments']
                        
                        if func_name == 'generate_custom_chart':
                             code = args.get('python_code', '')
                             result = generate_custom_chart(code)
                             messages.append({'role': 'tool', 'content': result})
                else:
                    report_content = response['message']['content']
                    break
            
            if not report_content:
                messages.append({'role': 'user', 'content': 'Please finish generating the Markdown report now without using any more tools.'})
                final_res = ollama.chat(model=self.model, messages=messages, options={"num_ctx": 16384}, keep_alive=0)
                report_content = final_res['message']['content']
                
            import requests
            try:
                ollama.chat(model=self.model, messages=[{'role': 'user', 'content': 'unload'}], keep_alive=0)
            except: pass
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import base64
            if generated_charts:
                report_content += "\n\n## Generated Custom Charts\n"
                for c_path in generated_charts:
                    if os.path.exists(c_path):
                        with open(c_path, "rb") as img_file:
                            b64_string = base64.b64encode(img_file.read()).decode()
                        report_content += f"<img src='data:image/png;base64,{b64_string}' alt='Generated Chart' style='max-width: 100%; height: auto; margin-bottom: 20px;'/>\n"

            import json
            tracks_json = json.dumps(self.tracks) if hasattr(self, 'tracks') else "{}"
            safe_env_context = env_context.replace("'", "\\'").replace("\\n", "<br>")

            # Read template and inject data
            template_path = os.path.join(os.path.dirname(__file__), "huddle_simulation_template.html")
            with open(template_path, 'r', encoding='utf-8') as f:
                simulation_template = f.read()
            
            simulation_html = simulation_template.replace("{safe_env_context}", safe_env_context).replace("{tracks_json}", tracks_json)
            sim_path = os.path.join(output_dir, "huddle_simulation.html")
            with open(sim_path, "w") as f:
                f.write(simulation_html)
                
            report_content += f"\n\n## Interactive Simulation\nThe interactive huddle dynamics simulation has been extracted and saved to: `{sim_path}`."


            report_path = os.path.join(output_dir, "final_huddle_report.md")
            with open(report_path, "w") as f:
                f.write(report_content)
                
            return report_content
        except Exception as e:
            return f"Error generating report via Ollama: {e}"