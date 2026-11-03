import urllib.request
import streamlit as st
import logging
import warnings


st_loggers = [
    "streamlit.runtime.scriptrunner.script_run_context",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.scriptrunner.script_run_context",
]
for logger_name in st_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
import json
import html
from pathlib import Path
from behave_ai import BehaviorAgent
from interactive_mode import handle_interactive_query
from tools.gui_tracking_state import (
    begin_tracking_attempt, current_tracking_result, report_for_result,
    reset_session_analysis_state, save_uploaded_video, store_report_for_result,
    sync_tracking_context,
)

st.set_page_config(page_title="PenguinAgent", layout="wide", page_icon="🐧")

# Initialize Agent
if "agent" not in st.session_state:
    st.session_state.agent = BehaviorAgent()

# Helper to get Ollama models
@st.cache_data(ttl=60)
def get_ollama_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
            models_data = json.loads(response.read())["models"]
        return [model["name"] for model in models_data if isinstance(model.get("name"), str)]
    except Exception:
        return []

st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    h1, h2, h3 {
        color: #4CAF50 !important;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .rag-tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: #4CAF50;
        font-weight: bold;
        font-size: 0.85em;
        vertical-align: super;
        padding: 0 4px;
        margin: 0 1px;
        background-color: rgba(76, 175, 80, 0.15);
        border-radius: 4px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        transition: background-color 0.2s;
    }
    .rag-tooltip:hover {
        background-color: rgba(76, 175, 80, 0.3);
    }
    .rag-tooltip .rag-tooltiptext {
        visibility: hidden;
        width: 400px;
        max-height: 300px;
        overflow-y: auto;
        word-wrap: break-word;
        background-color: #1e1e1e;
        color: #e0e0e0;
        text-align: left;
        border-radius: 12px;
        padding: 16px;
        position: absolute;
        z-index: 999999;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.2s, visibility 0.2s;
        font-size: 0.95rem;
        font-weight: normal;
        line-height: 1.6;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .rag-tooltip .rag-tooltiptext strong {
        color: #4CAF50;
        font-size: 1.05em;
        display: block;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(76, 175, 80, 0.3);
        padding-bottom: 4px;
    }
    .rag-tooltip .rag-tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -8px;
        border-width: 8px;
        border-style: solid;
        border-color: #1e1e1e transparent transparent transparent;
    }
    .rag-tooltip:hover .rag-tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    /* Ensure Streamlit markdown containers don't clip our tooltips */
    [data-testid="stMarkdownContainer"] {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐧 PenguinAgent: AI Antarctic Researcher")

with st.sidebar:
    st.header("Control Panel")
    
    # Keep the configured model selected while allowing any locally installed model.
    gemma_models = get_ollama_models()
    configured_model = st.session_state.agent.model
    if configured_model not in gemma_models:
        gemma_models.insert(0, configured_model)

    selected_model = st.selectbox("Select Model (via Ollama)", gemma_models, index=gemma_models.index(configured_model))
    st.session_state.agent.model = selected_model

    st.markdown("---")
    st.subheader("Knowledge Base")
    
    if "rag_dir" not in st.session_state:
        st.session_state.rag_dir = st.session_state.agent.rag_dir

    rag_dir_input = st.text_input("Documents Folder", value=st.session_state.rag_dir, help="Specify the local path to the folder containing your scientific PDFs.")
    st.session_state.rag_dir = rag_dir_input
    rag_directory_valid = True
    try:
        st.session_state.agent.rag_dir = rag_dir_input
    except (OSError, ValueError, RuntimeError) as exc:
        rag_directory_valid = False
        st.error(f"Invalid scientific PDF directory: {exc}")

    if st.button("Initialize Local RAG"):
        if rag_directory_valid:
            with st.spinner(f"Indexing PDFs from {st.session_state.rag_dir}..."):
                try:
                    if st.session_state.agent.initialize_rag() is not True:
                        raise RuntimeError("The scientific index was not initialized.")
                    st.success("RAG Initialized!")
                except Exception as exc:
                    st.error(f"Cannot initialize scientific RAG: {exc}")
    # Video Upload (Single Source of Truth)
    st.subheader("1. Upload Video")
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        try:
            upload = save_uploaded_video(
                uploaded_file.name, uploaded_file.getbuffer(), st.session_state.agent.videos_dir,
            )
            if upload.changed or st.session_state.get("current_video") != upload.filename:
                reset_session_analysis_state(st.session_state)
            st.session_state.current_video = upload.filename
            st.caption(f"Loaded: {upload.filename}")
        except Exception as exc:
            st.session_state.pop("current_video", None)
            begin_tracking_attempt(st.session_state, st.session_state.agent)
            st.error(f"Cannot save uploaded video: {exc}")
    else:
        if st.session_state.pop("current_video", None) is not None:
            begin_tracking_attempt(st.session_state, st.session_state.agent)
        else:
            st.session_state.agent.invalidate_tracking_state()
        st.info("Please upload a video to begin analysis.")
        
    st.subheader("2. Analyze Video")

    if "current_video" in st.session_state:
        default_out = os.path.join(st.session_state.agent.videos_dir, os.path.splitext(st.session_state.current_video)[0] + "_output")
    else:
        default_out = ""

    output_dir = st.text_input("Output Directory", value=default_out, help="Where to save the JSON tracks, annotated video, and report.")
    st.session_state.output_dir = output_dir if output_dir else default_out

    current_video = st.session_state.get("current_video")
    if current_video:
        try:
            sync_tracking_context(
                st.session_state, st.session_state.agent, current_video, st.session_state.output_dir,
            )
        except Exception as exc:
            begin_tracking_attempt(st.session_state, st.session_state.agent)
            st.error(f"Cannot select this video analysis: {exc}")
    active_result = current_tracking_result(
        st.session_state, st.session_state.agent, current_video, st.session_state.output_dir,
    )
    
    compact_annotation = st.checkbox(
        "Compact annotation",
        value=True, 
        help="Instead of cumbersome species names, use the '1_1', '2_1' format in videos. The correspondence map will be saved in a text file tracks_species_mapping.txt."
    )

    if st.button("Start processing and annotating video"):
        if "current_video" not in st.session_state:
            st.error("Upload a video first!")
        else:
            previous_run_id = active_result.run_id if active_result else None
            begin_tracking_attempt(st.session_state, st.session_state.agent)
            active_result = None
            with st.spinner(f"Extracting sub-pixel masks & tracks for {st.session_state.current_video}..."):
                try:
                    st.session_state.agent.run_tracking(
                        st.session_state.current_video, 
                        output_dir=st.session_state.output_dir,
                        compact_annotation=compact_annotation
                    )
                    active_result = current_tracking_result(
                        st.session_state, st.session_state.agent,
                        st.session_state.current_video, st.session_state.output_dir,
                    )
                    if active_result is None or active_result.run_id == previous_run_id:
                        raise RuntimeError("Tracking did not produce a new validated result.")
                    st.session_state.annotated_video = str(active_result.annotated_video_path)
                    st.success("Tracks and annotated video successfully generated!")
                except Exception as e:
                    active_result = None
                    begin_tracking_attempt(st.session_state, st.session_state.agent)
                    st.error(f"Error during tracking: {e}")
                
    st.divider()
    st.subheader("📄 Generating a report")

    report_type = st.radio(
        "Select the type of research to generate the report:",
        options=[
            "Search for anomalies", 
            "Huddling Analysis"
        ],
        help="Choose your research focus before generating your article text."
    )

    # Context parameters for Anomaly Detection
    anomaly_params = {}
    if report_type == "Search for anomalies":
        st.markdown("**Anomaly Index Coefficients:**")
        st.caption("These user-selected weights define an illustrative index; they are not calibrated anomaly probabilities or validated biological scores.")
        c1, c2, c3 = st.columns(3)
        with c1:
            anomaly_params['w_u'] = st.number_input("Uniqueness (w_u)", value=0.5, min_value=0.0, step=0.1, format="%.2f")
        with c2:
            anomaly_params['w_a'] = st.number_input("Anormality (w_a)", value=0.3, min_value=0.0, step=0.1, format="%.2f")
        with c3:
            anomaly_params['w_r'] = st.number_input("Rarity (w_r)", value=0.2, min_value=0.0, step=0.1, format="%.2f")

    # Context parameters for Huddling analysis
    huddle_params = {}
    if report_type == "Huddling Analysis":
        st.caption("The viewer replays tracked observations. Temperature, wind and heat loss are not measured from this video.")
        if st.checkbox("Include optional thermodynamic assumptions", value=False,
                       help="Include explicitly chosen model inputs in the report; these are not video measurements."):
            st.markdown("**User-supplied model assumptions:**")
            huddle_params['pe'] = st.number_input("Assumed Peclet number (Pe)", value=100.0, min_value=0.0, step=10.0,
                                                  help="Dimensionless model input; no wind measurement is inferred.")
            huddle_params['beta'] = st.number_input("Assumed model noise (β)", value=0.25, min_value=0.0, step=0.05, format="%.2f")
            c1, c2 = st.columns(2)
            with c1:
                huddle_params['tp'] = st.number_input("Assumed crowd-edge temperature (Tp) °C", value=-10.0, step=1.0)
            with c2:
                huddle_params['t_inf'] = st.number_input("Assumed ambient temperature (T∞) °C", value=-40.0, step=1.0)

    if st.button("Generate Scientific Report", type="primary"):
        active_result = current_tracking_result(
            st.session_state, st.session_state.agent,
            st.session_state.get("current_video"), st.session_state.output_dir,
        )
        if active_result is None:
             st.error("No validated analysis for this video and output directory. Run tracking first.")
        else:
             st.session_state.pop("report", None)
             st.session_state.pop("report_run_id", None)
             with st.spinner(f"The agent writes a report: {report_type}..."):
                 try:
                     if report_type == "Search for anomalies":
                         paper = st.session_state.agent.generate_anomaly_report(
                             st.session_state.current_video, 
                             output_dir=st.session_state.output_dir,
                             **anomaly_params
                         )
                     elif report_type == "Huddling Analysis":
                         paper = st.session_state.agent.generate_huddle_paper(
                             st.session_state.current_video, 
                             output_dir=st.session_state.output_dir,
                             **huddle_params
                         )
                     
                     validated_result = current_tracking_result(
                         st.session_state, st.session_state.agent,
                         st.session_state.current_video, st.session_state.output_dir,
                     )
                     if validated_result is None or validated_result.run_id != active_result.run_id:
                         active_result = validated_result
                         raise RuntimeError("The tracking result changed while generating the report.")
                     store_report_for_result(st.session_state, paper, active_result, st.session_state.agent.current_rag_refs)
                     st.success("The report has been successfully generated! View the results below.")
                 except Exception as e:
                     st.error(f"Error generating report: {e}")

active_result = current_tracking_result(
    st.session_state, st.session_state.agent,
    st.session_state.get("current_video"), st.session_state.output_dir,
)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Analysis (SAM 3 output)")
    if "current_video" in st.session_state:
        video_path = os.path.join(st.session_state.agent.videos_dir, st.session_state.current_video)
        
        if active_result is not None:
            st.video(str(active_result.annotated_video_path), format="video/mp4")
            st.caption("Explicitly showing SAM 3 tracking, masks, and trajectories")
            tracks = active_result.tracks
            st.write("Tracking JSON generated (sample):")
            st.json({k: tracks[k][:2] for k in list(tracks.keys())[:3]})
        else:
            st.video(video_path)
            st.caption("Original Video. Click 'Start processing and annotating video' to generate tracks and masks.")
    else:
        st.write("Upload a video in the sidebar to view it here.")
        
    report_text = report_for_result(st.session_state, active_result)
    if report_text is not None:
        st.subheader("Generated Scientific Report")
        
        import re
        def replace_citation(match):
            ref_id = int(match.group(1))
            report_refs = st.session_state.get("report_rag_refs", {})
            if ref_id in report_refs:
                ref_data = report_refs[ref_id]
                source = html.escape(str(ref_data['source']))
                text = html.escape(str(ref_data['text'])).replace('\n', '<br/>')
                return f'<span class="rag-tooltip">[{ref_id}]<span class="rag-tooltiptext"><strong>{source}</strong>{text}</span></span>'
            return match.group(0) # If no ref found, leave it as is.
            
        processed_report = re.sub(r'\[(\d+)\]', replace_citation, report_text)
        
        sim_match = re.search(r'The interactive huddle dynamics simulation has been extracted and saved to: `([^`]+)`', processed_report)
        if sim_match:
            processed_report = processed_report.replace(sim_match.group(0), "_The tracked-observation viewer is rendered below._")
            
        st.markdown(processed_report, unsafe_allow_html=True)
        
        if sim_match:
            sim_path = Path(active_result.output_dir) / "huddle_simulation.html"
            try:
                simulation_matches_run = Path(sim_match.group(1)).resolve() == sim_path.resolve()
            except (OSError, ValueError):
                simulation_matches_run = False
            if simulation_matches_run and sim_path.is_file():
                st.caption("Tracked-observation viewer: positions come from the video analysis; this is not a thermodynamic simulation.")
                with open(sim_path, 'r', encoding='utf-8') as f:
                    sim_html = f.read()
                import streamlit.components.v1 as components
                components.html(sim_html, height=800, scrolling=False)

with col2:
    st.subheader("Chat")
    if active_result is None:
        st.caption("No video analysis context; scientific literature only.")
    
    st.markdown("_Use the '🏷️ Add Tags' button to auto-paste penguin indicators, or manually type @tags._")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            
    if "pending_tag" not in st.session_state:
        st.session_state.pending_tag = ""

    if "last_copied_tag" not in st.session_state:
        st.session_state.last_copied_tag = ""

    # Tag Selector UI
    with st.popover("🏷️ Add Tags"):
        st.write("Select a tag to copy it to the clipboard.")

        penguin_options = ["None"]
        if active_result is not None and active_result.tracks:
            penguin_options.extend(active_result.tracks.keys())

        penguin_tag = st.selectbox("Penguins (Species & ID)", penguin_options)
        if st.button("Add Penguin Tag", disabled=active_result is None):
            if penguin_tag != "None":
                st.session_state.pending_tag = f"@{penguin_tag} "
                st.session_state.last_copied_tag = f"@{penguin_tag} "
                st.rerun()

    if st.session_state.last_copied_tag:
        st.info(f"📋 Copied to clipboard: **{st.session_state.last_copied_tag.strip()}** - You can now paste it in the chat box below.")

    if st.session_state.pending_tag:
        tag_to_insert = st.session_state.pending_tag
        st.session_state.pending_tag = ""  

        tag_json = json.dumps(tag_to_insert.strip()).replace("<", "\\u003c")
        js_code = f"""
        <script>
        const textToCopy = {tag_json};
        const parentWindow = window.parent;
        const parentDoc = parentWindow.document;

        // Try Clipboard API
        if (parentWindow.navigator.clipboard && parentWindow.isSecureContext) {{
            parentWindow.navigator.clipboard.writeText(textToCopy).catch(err => console.error('Clipboard copy failed', err));
        }} else {{
            // Fallback to execCommand
            const textArea = parentDoc.createElement("textarea");
            textArea.value = textToCopy;
            textArea.style.position = "absolute";
            textArea.style.left = "-999999px";
            parentDoc.body.prepend(textArea);
            textArea.select();
            try {{
                parentDoc.execCommand('copy');
            }} catch (error) {{
                console.error('Fallback copy failed', error);
            }} finally {{
                textArea.remove();
            }}
        }}
        </script>
        """
        import streamlit.components.v1 as components
        components.html(js_code, height=0, width=0)                
    prompt_typing = st.chat_input("Ask PenguinAgent...")
    
    if prompt_typing:
        active_result = current_tracking_result(
            st.session_state, st.session_state.agent,
            st.session_state.get("current_video"), st.session_state.output_dir,
        )
        full_prompt = prompt_typing
        st.session_state.setdefault("messages", [])
            
        st.session_state.messages.append({"role": "user", "content": full_prompt})
        with st.chat_message("user"):
            st.markdown(full_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner(f"{st.session_state.agent.model} is thinking..."):
                try:
                    response = handle_interactive_query(
                        st.session_state.agent, full_prompt,
                        output_dir=active_result.output_dir if active_result is not None else None,
                    )
                    if not isinstance(response, str) or not response.strip():
                        raise RuntimeError("The assistant returned no answer.")
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as exc:
                    message = f"Error answering this question: {exc}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message, "error": True})
        st.rerun()
