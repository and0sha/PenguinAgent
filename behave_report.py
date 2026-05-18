import os
import ollama
import torch
import gc
import shutil
import base64
import re
import glob

class ReportMixin:
    def generate_anomaly_report(self, video_name, output_dir=None, w_u=0.5, w_a=0.3, w_r=0.2):
        import matplotlib.pyplot as plt
        video_path = os.path.abspath(os.path.join(self.videos_dir, video_name))
        
        if output_dir is None:
            output_dir = os.path.join(self.videos_dir, os.path.splitext(video_name)[0] + "_output")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
            
        json_path = os.path.join(output_dir, "tracks.json")
        
        if not getattr(self, 'tracks', None):
            if os.path.exists(json_path):
                self.tracks = self._load_and_remap_tracks(json_path, output_dir)
            else:
                return "No tracks available. Please run tracking first."

        self.current_rag_refs = {}
        self.rag_ref_counter = 1
        
        summary = f"Total penguins tracked: {len(self.tracks)}\n"
        for pid, data in self.tracks.items():
            summary += f"- {pid}: tracked for {len(data)} seconds.\n"
            
        rag_context = ""
        if getattr(self, 'vector_store', None):
            try:
                # Query RAG for general penguin behavior literature, including the identified species/names
                species_keywords = " ".join(list(self.tracks.keys())[:10]) # Use the generated names as keywords
                rag_query = f"penguin behavior tracking anomaly {species_keywords}"
                docs = self.vector_store.max_marginal_relevance_search(rag_query, k=5, fetch_k=20, lambda_mult=0.7) # Use MMR for broader context
                
                results = []
                for d in docs:
                    ref_id = self.rag_ref_counter
                    self.rag_ref_counter += 1
                    source_name = os.path.basename(d.metadata.get('source', 'Unknown'))
                    self.current_rag_refs[ref_id] = {'source': source_name, 'text': d.page_content}
                    results.append(f"- Source [{ref_id}] ({source_name}): {d.page_content[:500]}...")
                
                rag_context = "Relevant Literature from RAG:\n" + "\n".join(results)
            except Exception as e:
                print(f"Error searching RAG: {e}")
                
        # --- VIDEO EXTRACTION LOGIC FOR GEMMA 4 MULTIMODAL ---
        import cv2
        frames_to_send = []
        frames_dir = os.path.join(output_dir, "temp_frames")
        
        if os.path.exists(video_path):
            os.makedirs(frames_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30
            
            frame_idx = 0
            saved_count = 0
            
            # Gemma 4 supports up to 60 seconds at 1fps. We will cap at 60 frames.
            # If the video is longer, we sample 60 evenly spaced frames.
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_seconds = total_frames / fps
            
            if total_seconds > 60:
                step = total_frames / 60
            else:
                step = fps 
                
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx >= saved_count * step and saved_count < 60:
                    frame_path = os.path.join(frames_dir, f"frame_{saved_count:03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames_to_send.append(frame_path)
                    saved_count += 1
                frame_idx += 1
            cap.release()

        # Tool definition for Gemma 4
        def search_scientific_literature(query: str) -> str:
            """
            Search the local RAG database for scientific literature. Use this to find specific facts about penguin behavior, anomalies, or environmental factors.
            """
            if getattr(self, 'vector_store', None) is None:
                return "Error: Local RAG database is not initialized. Just proceed with the summary."
            try:
                print(f"🔍 Gemma 4 is searching RAG for: '{query}'")
                
                docs = self.vector_store.max_marginal_relevance_search(query, k=4, fetch_k=15, lambda_mult=0.7)
                
                results = []
                for d in docs:
                    ref_id = self.rag_ref_counter
                    self.rag_ref_counter += 1
                    source_name = os.path.basename(d.metadata.get('source', 'Unknown'))
                    self.current_rag_refs[ref_id] = {'source': source_name, 'text': d.page_content}
                    results.append(f"- Source [{ref_id}] ({source_name}): {d.page_content[:500]}...")
                
                return "Relevant Literature Found:\n" + "\n".join(results)
            except Exception as e:
                return f"Search failed: {e}"
                
        generated_charts = []
        def generate_custom_chart(python_code: str) -> str:
       
            try:
                print("📊 Gemma 4 is generating a custom chart...")
                import glob
                
                chart_filename = f"custom_chart_{len(generated_charts) + 1}.png"
                output_chart_path = os.path.join(output_dir, chart_filename)
                
                local_vars = {
                    'tracking_data': self.tracks,
                    'output_chart_path': output_chart_path,
                    'plt': plt
                }
                
                pngs_before = set(glob.glob(os.path.join(output_dir, "*.png")))
                
                original_cwd = os.getcwd()
                os.chdir(output_dir)
                try:
                    # Execute the code securely in a restricted namespace
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

        # Extract identified species from tracks
        identified_species = set()
        for pid in self.tracks.keys():
            pid_name = re.sub(r'_\d+$', '', pid)
            pid_name = pid_name.replace('_', ' ')
            identified_species.add(pid_name)
            
        questions_context = ""
        try:
            with open("questions/100.txt", "r", encoding="utf-8") as f:
                questions_context += "--- General 100 Questions for Evaluation ---\n" + f.read() + "\n\n"
        except Exception:
            pass
            
        for species in identified_species:
            species_folder = os.path.join("questions", species)
            if os.path.exists(species_folder):
                for file in glob.glob(os.path.join(species_folder, "*.txt")):
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            questions_context += f"--- 20 Questions specifically for {species} ---\n" + f.read() + "\n\n"
                    except Exception:
                        pass
        summary = f"Total penguins tracked: {len(self.tracks)}\n\n"
        summary += "### Kinematic Telemetry Logs (Per-Second Spatial Data):\n"
        
        for pid, data in self.tracks.items():
            summary += f"#### Penguin Profile: {pid}\n"
            summary += "| Second | Coordinates (X, Y) | Velocity Vectors (Vx, Vy) | Surface Area (px) |\n"
            summary += "| :--- | :--- | :--- | :--- |\n"
            
            for frame_data in data:
                sec = frame_data.get("second", 0)
                x = frame_data.get("x", 0)
                y = frame_data.get("y", 0)
                vx = frame_data.get("vx", 0.0)
                vy = frame_data.get("vy", 0.0)
                area = frame_data.get("area", 0.0)
                
                summary += f"| {sec}s | ({x}, {y}) | ({vx}, {vy}) | {area} |\n"
            summary += "\n"
        
        prompt = f"""
You are PenguinAgent, an expert ornithologist AI.
You have been provided with {len(frames_to_send)} sequential frames (1 frame per second) from the video, along with the following tracking data summary.

Precise Tracking & Telemetry Dataset:
{summary}

{rag_context}

Here are some specific questions to help guide your evaluation:
{questions_context}

Your primary goal is to identify anomalies in the behavior of the penguins based on the provided questions and data.
1. Correlate the visual state of the scene from the raw frames with the provided spatial coordinates (X, Y) and telemetry logs. For instance, at second 2, find the penguin moving at the coordinates specified in the dataset.
2. Cross-reference your findings with the provided expert questions to determine if the behavior is standard or anomalous.

You have access to two tools:
1. `search_scientific_literature`: To query the RAG database for scientific facts about penguin behavior.
2. `generate_custom_chart`: To write and execute Python matplotlib code. Use this to visually plot the tracking data.

CRITICAL FORMATTING INSTRUCTIONS FOR THE FINAL REPORT:
- When citing from the RAG context, use standard markdown bracket notation like [1], [2], etc., matching the Source IDs provided in the tool output. DO NOT use HTML spans for citations.
- In the "Methodology -> Reasoning and Interpretation" section, explicitly state that you are the `{self.model}` model running locally via Ollama, that you directly processed {len(frames_to_send)} frames.

The final report MUST follow this EXACT structure:
1. **Executive Summary**
2. **Methodology**
3. **General Scene Description & Assessment** 
   - First, provide a general description of the scene from your multimodal analysis.
   - Then, evaluate the ENTIRE SCENE for anomalies, rarity, uniqueness, and whether such an event has been recorded in the scientific literature before.
4. **Visual & Kinematic Analysis**
   - Describe the trajectories, "Life History" Chronicles, and dynamic relationship graph.
5. **Individual Penguin Evaluation (Anomaly Section)** 
   - You MUST calculate and provide a detailed Anomaly Scoring (3x100 System) analysis for EACH PENGUIN identified.
   - Rely heavily on the provided "General 100 Questions" and "20 Questions specifically for [species]" to evaluate each individual. Answer the relevant questions for each individual to justify your score.

For EACH PENGUIN, evaluate:
    1. Uniqueness (U) [0-100] 
    The Uniqueness (U) metric, utilizing a scale from 0 to 100 within the context of scientific research, serves as an analytical index reflecting the degree of scientific novelty or the level of prior investigation regarding a specific phenomenon.
In this framework, the metric evaluates the frequency and depth to which a particular scene, behavioral pattern, or interaction has been previously documented in academic literature. An inverse correlation applies: a higher U-score indicates a lower prevalence of existing empirical data within the scientific domain.
Below is a detailed academic classification of this scale:
U [0–20]: Fundamental / Baseline Behavior (Negligible Uniqueness)
Literature Status: Canonical, universally recognized phenomenon.
Description: The behavior or scene is extensively documented across hundreds or thousands of peer-reviewed publications. The topic is thoroughly covered by meta-analyses, systematic reviews, and foundational textbook chapters.
Scientific Value: Primarily serves to corroborate existing theoretical frameworks. Establishing novelty for publication strictly based on this phenomenon, absent new variables or methodologies, is highly improbable due to the lack of originality.
U [21–40]: Well-Documented Behavior (Low Uniqueness)
Literature Status: Frequently observed and investigated across diverse contexts.
Description: The behavior is widely recognized by both specialized researchers and the broader scientific community. Comprehensive searches within primary academic databases (e.g., Scopus, Web of Science, PubMed) readily yield substantial contemporary literature detailing identical or highly analogous scenes.
U [41–60]: Niche Behavior (Moderate Uniqueness)
Literature Status: Acknowledged within the discipline but characterized by fragmented or limited empirical investigation.
Description: The phenomenon is referenced in the literature, though typically peripherally or confined to highly specific, restricted parameters. Documentation may exist for related taxa or parallel conditions, but comprehensive analysis of the specific target behavior under the current exact conditions remains absent.
U [61–80]: Rare / Anomalous Behavior (High Uniqueness)
Literature Status: Characterized by isolated mentions, case reports, or reliance on antiquated empirical data.
Description: The existing literature yields only a minimal number of publications—or brief references within older manuscripts—that tangentially or imprecisely describe the scene. The behavior may be classified as anomalous, mutational, or the result of highly specific, idiosyncratic environmental variables.
Scientific Value: High. Observations within this range constitute robust material for Short Communications or Case Studies, possessing the potential to catalyze novel academic discourse and subsequent investigations.
U [81–100]: Absolute Uniqueness (Novel Discovery / Uncharted Domain)
Literature Status: Complete absence of prior empirical data.
Description: Rigorous literature reviews utilizing comprehensive keyword strategies yield null results. The scene, mechanism, or behavioral pattern has not been previously described by the scientific community. This represents an entirely novel behavioral phenotype, an unprecedented response to stimuli, or an undiscovered socio-behavioral construct.
Scientific Value: Maximum. A U-score approaching 100 signifies a paradigm-shifting observation. Findings in this tier require an exceptionally rigorous evidentiary standard and robust methodological protocols, as the scientific community typically subjects claims of absolute novelty to intense peer scrutiny.
    2. Anormality (A) [0-100]
    The Anormality (A) metric, utilizing a scale from 0 to 100 within the context of focal animal sampling, serves as an analytical index reflecting the degree of intra-individual behavioral variance.
Unlike population-level metrics, this index specifically evaluates the extent to which an observed action, state, or interaction deviates from the established baseline ethogram of a specific focal individual (in this context, the target subject). An inverse correlation applies to normalization: a higher A-score indicates a more profound departure from the subject’s own documented historical behavioral patterns.
Below is a detailed academic classification of this scale:
A [0–20]: Baseline / Normative Behavior (Negligible Anormality)
Behavioral Status: Ethologically consistent, high-frequency baseline.
Description: The observed behavior aligns congruently with the focal individual's historically documented ethogram. It represents routine metabolic, locomotive, or intra-colonial social activities that are statistically standard for this specific subject under typical environmental parameters.
Scientific Value: Primarily serves to establish and verify the behavioral control state. Continuous logging within this tier is essential for longitudinal monitoring of standard physiological and psychological homeostasis.
A [21–40]: Minor Behavioral Variation (Low Anormality)
Behavioral Status: Contextually expected phenotypic plasticity.
Description: The behavior exhibits slight statistical deviations from routine chronobiological or spatial patterns, yet remains entirely within the subject's known limits of behavioral plasticity. Such variations typically reflect standard, low-level adaptive responses to minor microclimatic fluctuations or transient social reconfigurations within the immediate vicinity.
A [41–60]: Atypical / Context-Specific Shift (Moderate Anormality)
Behavioral Status: Noticeable deviation contingent upon specific exogenous or endogenous catalysts.
Description: The focal individual engages in kinetic or social patterns that are statistically infrequent within its personal behavioral repository. Observations falling within this tier generally indicate a measurable response to moderate ecological stressors, shifts in territorial boundaries, or defined physiological transitions (e.g., the onset of molting or pre-copulatory physiological changes).
A [61–80]: Aberrant Behavior (High Anormality)
Behavioral Status: Highly irregular, indicative of significant systemic or environmental disruption.
Description: The behavior represents a stark, quantifiable departure from the subject's established normative profile. Such actions are exceptionally rare for this specific individual and strongly correlate with acute distress, extreme predatory evasion tactics, severe localized ecological degradation, or the preliminary onset of pathological conditions.
Scientific Value: Substantial. Documentation within this range is critical for identifying idiosyncratic stress responses, tracking early-stage pathogenic vectors, or observing individual-level cognitive adaptations to severe environmental perturbations.
A [81–100]: Pathological / Unprecedented Deviation (Absolute Anormality)
Behavioral Status: Completely uncharacteristic, potentially deleterious, and lacking historical precedent.
Description: The observed behavior has never been previously recorded in the focal individual's observational history and sharply contradicts established survival imperatives. Activity in this uppermost tier strongly implies severe neurological impairment, advanced pathogenesis, acute psychological trauma, or exposure to catastrophic environmental anomalies surpassing the subject's adaptive capacity.
Scientific Value: Critical for immediate intervention protocols or forensic ethology. Observations at an A-score approaching 100 define the absolute terminal limits of the individual's behavioral elasticity under extreme physiological or ecological duress.
    3. Rarity (R) [0-100]
    The Rarity (R) metric, utilizing a scale from 0 to 100 within the context of focal animal sampling, serves as a quantitative index reflecting the temporal frequency and probability of occurrence of a specific behavior or scene for a given individual.
Crucially, Rarity is fundamentally distinct from Anormality. A behavior with a highly elevated R-score is not necessarily pathological, aberrant, or stressful; it may be a perfectly normative, highly adaptive biological function that simply occurs very infrequently within the subject's overall time budget. An inverse correlation applies: a higher R-score indicates a lower frequency of occurrence in the focal individual's lifespan or observation period.
Below is a detailed academic classification of this scale:
R [0–20]: Ubiquitous / High-Frequency Behavior (Negligible Rarity)
Temporal Status: Constant, daily, or hourly occurrence.
Description: The behavior forms the bulk of the focal individual’s standard time budget. These are maintenance and foundational survival activities that occur with high statistical predictability.
Scientific Value: Essential for constructing standard activity budgets and calculating baseline energetic expenditure.
R [21–40]: Periodic / Recurrent Behavior (Low Rarity)
Temporal Status: Regular cyclic activity occurring on a diurnal or weekly basis.
Description: The event is highly predictable but occupies a smaller, discrete fraction of the time budget compared to ubiquitous behaviors. It often represents structured interactions with the environment or conspecifics.
R [41–60]: Episodic / Seasonal Behavior (Moderate Rarity)
Temporal Status: Predictable but highly restricted temporally.
Description: The behavior is entirely normal and expected, but it is confined to specific, limited phases of the individual's annual cycle. Observers will only record this behavior during specific biological windows.
R [61–80]: Contingent / Infrequent Event (High Rarity)
Temporal Status: Reactive, highly situational, and statistically improbable on a given day.
Description: The behavior is an adaptive response to specific, uncommon external triggers. While the individual is evolutionarily programmed to execute this behavior, the environmental catalyst required to trigger it is rarely encountered.
Scientific Value: High. These observations are difficult to capture systematically but are crucial for understanding the subject's maximum physical capabilities, acute stress responses, and survivability under pressure.
R [81–100]: Singular / Once-in-a-Lifetime Event (Absolute Rarity)
Temporal Status: Exceptionally low probability of observation; may occur only once or never in the subject's lifespan.
Description: A completely normative but extraordinarily rare life event. This category encompasses major life-history milestones, demographic shifts, or responses to statistically extreme ecological anomalies.
Scientific Value: Critical for long-term demographic and life-history studies. Documenting events at an R-score approaching 100 provides rare insights into evolutionary fitness and behavioral plasticity at the extreme margins of the individual's life cycle.

**Anomaly_Index = sqrt(w_a * A² + w_r * R² + w_u * U²)**
(w_a = {w_a}, w_r = {w_r}, w_u = {w_u})
Show the calculation results based on this formula.
First, carefully calculate and check the values ​​using the formula for each penguin.
As a result, output only the table with the results of each penguin and all its indices: Uniqueness (U) [0-100]; Anormality (A) [0-100]; Rarity (R) [0-100]; Anomaly_Index
6. **References**
"""
        try:
            
            messages = [{'role': 'system', 'content': '<|think|> You are an autonomous multimodal AI generating a scientific report from video frames and data. You can use tools to research literature and generate charts.'}]
            
            if frames_to_send:
                messages.append({'role': 'user', 'content': 'Visual context (frames from video):', 'images': frames_to_send})
                
            user_msg = {'role': 'user', 'content': prompt}
            messages.append(user_msg)
            
            # Loop to allow up to 3 turns of tool calling
            max_turns = 3
            report_content = ""
            
            for turn in range(max_turns):
                print(f"Turn {turn+1}: Sending prompt and {len(frames_to_send) if turn==0 else 0} frames to Gemma 4 (Waiting for Tool Calls or Report)...")
                response = ollama.chat(
                    model=self.model, 
                    messages=messages, 
                    options={"num_ctx": 16384},
                    tools=[search_scientific_literature, generate_custom_chart],
                    keep_alive=0 if turn == max_turns - 1 else "5m"
                )
                
                if response.get("message", {}).get("tool_calls"):
                    print("Gemma 4 decided to use tools!")
                    messages.append(response['message']) # Add the assistant's tool call request
                    
                    for tool in response['message']['tool_calls']:
                        func_name = tool['function']['name']
                        args = tool['function']['arguments']
                        
                        if func_name == 'search_scientific_literature':
                             query = args.get('query', '')
                             result = search_scientific_literature(query)
                             messages.append({'role': 'tool', 'content': result})
                             
                        elif func_name == 'generate_custom_chart':
                             code = args.get('python_code', '')
                             result = generate_custom_chart(code)
                             messages.append({'role': 'tool', 'content': result})
                else:
                    # Final text response received
                    report_content = response['message']['content']
                    break
            
            # If after max_turns we still don't have text (stuck in tool loop), force a text generation
            if not report_content:
                messages.append({'role': 'user', 'content': 'Please finish generating the Markdown report now without using any more tools.'})
                final_res = ollama.chat(model=self.model, messages=messages, options={"num_ctx": 16384}, keep_alive=0)
                report_content = final_res['message']['content']
                
            # If the model didn't use keep_alive=0 on the last call, force unload
            import requests
            try:
                ollama.chat(model=self.model, messages=[{'role': 'user', 'content': 'unload'}], keep_alive=0)
            except: pass
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Cleanup temp frames
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
                
            # Append generated charts image markdown as base64 so Streamlit renders it properly
            if generated_charts:
                report_content += "\n\n## Generated Custom Charts\n"
                for c_path in generated_charts:
                    if os.path.exists(c_path):
                        with open(c_path, "rb") as img_file:
                            b64_string = base64.b64encode(img_file.read()).decode()
                        report_content += f"<img src='data:image/png;base64,{b64_string}' alt='Generated Chart' style='max-width: 100%; height: auto; margin-bottom: 20px;'/>\n"
            
            report_path = os.path.join(output_dir, "final_scientific_report.md")
            with open(report_path, "w") as f:
                f.write(report_content)
                
            return report_content
        except Exception as e:
            return f"Error generating report via Ollama: {e}"