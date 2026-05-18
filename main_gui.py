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
from st_keyup import st_keyup
import os
import cv2
import json
from behave_ai import BehaviorAgent
from interactive_mode import handle_interactive_query

# Initialize Agent
if "agent" not in st.session_state:
    st.session_state.agent = BehaviorAgent()

# Helper to get Ollama models
@st.cache_data(ttl=60)
def get_ollama_models():
    try:
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        models_data = json.loads(r.read())["models"]
        # Only list gemma models for this hackathon
        return [m["name"] for m in models_data if "gemma" in m["name"].lower()]
    except Exception:
        return ["gemma4:26b"] 

st.set_page_config(page_title="PenguinAgent", layout="wide", page_icon="🐧")

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
    
    # Gemma 4 Model Selection via Ollama
    gemma_models = get_ollama_models()
    if not gemma_models:
        gemma_models = ["gemma4:26b"]

    selected_model = st.selectbox("Select Gemma 4 Model (via Ollama)", gemma_models)
    st.session_state.agent.model = selected_model

    st.markdown("---")
    st.subheader("Knowledge Base")
    
    if "rag_dir" not in st.session_state:
        st.session_state.rag_dir = st.session_state.agent.rag_dir

    rag_dir_input = st.text_input("Documents Folder", value=st.session_state.rag_dir, help="Specify the local path to the folder containing your scientific PDFs.")
    st.session_state.rag_dir = rag_dir_input
    st.session_state.agent.rag_dir = rag_dir_input

    if st.button("Initialize Local RAG"):
        with st.spinner(f"Indexing PDFs from {st.session_state.rag_dir}..."):
            st.session_state.agent.initialize_rag()
            st.success("RAG Initialized!")            
    # Video Upload (Single Source of Truth)
    st.subheader("1. Upload Video")
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        video_save_path = os.path.join(st.session_state.agent.videos_dir, uploaded_file.name)
        # Avoid rewriting if it's the exact same file to save time, but overwrite if needed
        with open(video_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.current_video = uploaded_file.name
        st.success(f"Loaded: {uploaded_file.name}")
    else:
        st.info("Please upload a video to begin analysis.")
        
    st.subheader("2. Analyze Video")

    if "current_video" in st.session_state:
        default_out = os.path.join(st.session_state.agent.videos_dir, os.path.splitext(st.session_state.current_video)[0] + "_output")
    else:
        default_out = ""

    output_dir = st.text_input("Output Directory", value=default_out, help="Where to save the JSON tracks, annotated video, and report.")
    st.session_state.output_dir = output_dir if output_dir else default_out
    
    compact_annotation = st.checkbox(
        "Сompact annotation", 
        value=True, 
        help="Instead of cumbersome species names, use the '1_1', '2_1' format in videos. The correspondence map will be saved in a text file tracks_species_mapping.txt."
    )

    if st.button("Start processing and annotating video"):
        if "current_video" not in st.session_state:
            st.error("Upload a video first!")
        else:
            with st.spinner(f"Extracting sub-pixel masks & tracks for {st.session_state.current_video}..."):
                try:
                    tracks, annotated_video = st.session_state.agent.run_tracking(
                        st.session_state.current_video, 
                        output_dir=st.session_state.output_dir,
                        compact_annotation=compact_annotation
                    )
                    st.session_state.annotated_video = annotated_video
                    st.success("Tracks and annotated video successfully generated!")
                except Exception as e:
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
        c1, c2, c3 = st.columns(3)
        with c1:
            anomaly_params['w_u'] = st.number_input("Uniqueness (w_u)", value=0.5, step=0.1, format="%.2f")
        with c2:
            anomaly_params['w_a'] = st.number_input("Anormality (w_a)", value=0.3, step=0.1, format="%.2f")
        with c3:
            anomaly_params['w_r'] = st.number_input("Rarity (w_r)", value=0.2, step=0.1, format="%.2f")

    # Context parameters for Huddling analysis
    huddle_params = {}
    if report_type == "Huddling Analysis":
        st.markdown("**Thermodynamic model parameters (Waters 2012):**")
        huddle_params['pe'] = st.number_input("Wind force (Peclet number, Pe)", value=100.0, step=10.0, help="The ratio of convective to diffusive transfer. A realistic value is ~100.")
        huddle_params['beta'] = st.number_input("Noise/uncertainty level (β)", value=0.25, step=0.05, format="%.2f", help="How wrong are penguins? A realistic value is ~0.25.")
        c1, c2 = st.columns(2)
        with c1:
            huddle_params['tp'] = st.number_input("Temperature at the edge of the crowd (Tp) °C", value=-10.0, step=1.0)
        with c2:
            huddle_params['t_inf'] = st.number_input("Temperature away (T∞) °C", value=-40.0, step=1.0)

    if st.button("Generate Scientific Report", type="primary"):
        if "current_video" not in st.session_state:
             st.error("First, upload and process your video!")
        elif getattr(st.session_state.agent, 'tracks', None) is None:
             st.error("No tracking data! Run the analysis first.")
        else:
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
                     
                     st.session_state.report = paper
                     st.success("The report has been successfully generated! View the results below.")
                 except Exception as e:
                     st.error(f"Error generating report: {e}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Analysis (SAM 3 output)")
    if "current_video" in st.session_state:
        video_path = os.path.join(st.session_state.agent.videos_dir, st.session_state.current_video)
        
        out_dir = st.session_state.get("output_dir", os.path.join(st.session_state.agent.videos_dir, os.path.splitext(st.session_state.current_video)[0] + "_output"))
        annotated_path = os.path.join(out_dir, "annotated.mp4")
        json_path = os.path.join(out_dir, "tracks.json")
        
        if os.path.exists(annotated_path):
            st.video(annotated_path, format="video/mp4")
            st.caption("Explicitly showing SAM 3 tracking, masks, and trajectories")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    tracks = json.load(f)
                st.write("Tracking JSON generated (sample):")
                st.json({k: tracks[k][:2] for k in list(tracks.keys())[:3]})
        else:
            st.video(video_path)
            st.caption("Original Video. Click 'Start processing and annotating video' to generate tracks and masks.")
    else:
        st.write("Upload a video in the sidebar to view it here.")
        
    if "report" in st.session_state:
        st.subheader("Generated Scientific Report")
        report_text = st.session_state.report
        
        import re
        def replace_citation(match):
            ref_id = int(match.group(1))
            if hasattr(st.session_state.agent, 'current_rag_refs') and ref_id in st.session_state.agent.current_rag_refs:
                ref_data = st.session_state.agent.current_rag_refs[ref_id]
                source = ref_data['source']
                # Escape quotes and formatting for HTML insertion
                text = ref_data['text'].replace('"', '&quot;').replace('\n', '<br/>')
                return f'<span class="rag-tooltip">[{ref_id}]<span class="rag-tooltiptext"><strong>{source}</strong>{text}</span></span>'
            return match.group(0) # If no ref found, leave it as is.
            
        processed_report = re.sub(r'\[(\d+)\]', replace_citation, report_text)
        
        sim_match = re.search(r'The interactive huddle dynamics simulation has been extracted and saved to: `([^`]+)`', processed_report)
        if sim_match:
            processed_report = processed_report.replace(sim_match.group(0), "_The interactive simulation is rendered below._")
            
        st.markdown(processed_report, unsafe_allow_html=True)
        
        if sim_match:
            sim_path = sim_match.group(1)
            if os.path.exists(sim_path):
                with open(sim_path, 'r', encoding='utf-8') as f:
                    sim_html = f.read()
                import streamlit.components.v1 as components
                components.html(sim_html, height=800, scrolling=False)

with col2:
    st.subheader("Chat")
    
    st.markdown("_Use the '🏷️ Add Tags' button to auto-paste penguin indicators, or manually type @tags._")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if "pending_tag" not in st.session_state:
        st.session_state.pending_tag = ""

    if "last_copied_tag" not in st.session_state:
        st.session_state.last_copied_tag = ""

    # Tag Selector UI
    with st.popover("🏷️ Add Tags"):
        st.write("Select a tag to copy it to the clipboard.")

        penguin_options = ["None"]
        if hasattr(st.session_state.agent, 'tracks') and st.session_state.agent.tracks:
            penguin_options.extend(st.session_state.agent.tracks.keys())

        penguin_tag = st.selectbox("Penguins (Species & ID)", penguin_options)
        if st.button("Add Penguin Tag"):
            if penguin_tag != "None":
                st.session_state.pending_tag = f"@{penguin_tag} "
                st.session_state.last_copied_tag = f"@{penguin_tag} "
                st.rerun()

    if st.session_state.last_copied_tag:
        st.info(f"📋 Copied to clipboard: **{st.session_state.last_copied_tag.strip()}** - You can now paste it in the chat box below.")

    if st.session_state.pending_tag:
        tag_to_insert = st.session_state.pending_tag
        st.session_state.pending_tag = ""  

        js_code = f"""
        <script>
        const textToCopy = '{tag_to_insert.strip()}';
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
        full_prompt = prompt_typing
            
        st.session_state.messages.append({"role": "user", "content": full_prompt})
        with st.chat_message("user"):
            st.markdown(full_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner(f"{st.session_state.agent.model} is thinking..."):
                out_dir = st.session_state.get("output_dir")
                response = handle_interactive_query(st.session_state.agent, full_prompt, output_dir=out_dir)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
