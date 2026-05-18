# PenguinAgent 🐧

PenguinAgent is an AI-powered ornithologist application designed to automate the tracking, analysis, and behavioral reporting of Antarctic penguin colonies. It leverages state-of-the-art computer vision and large language models to extract kinematics from video footage, identify anomalies, and contextualize findings against scientific literature.

## Features

- **Video Telemetry & Tracking:** Employs SAM 3 (Segment Anything Model 3) to track individual penguins across video frames, extracting spatial coordinates, velocities, and area metrics  and we use SigLIP2 + linear probe on 18 penguin species for accurate species identification. 
- **Multimodal AI Analysis:** Uses Ollama (running the `gemma4:26b` model by default) to interpret video frames and telemetry data to generate comprehensive scientific reports and anomaly assessments.
- **Retrieval-Augmented Generation (RAG):** Integrates LangChain and FAISS to ground the AI's analysis in relevant scientific literature, providing verifiable citations.
- **Interactive Chat & Queries:** A Streamlit-based GUI allows researchers to interrogate the AI about specific penguins (e.g., `@penguin_1`), general behavioral literature, and complex dynamics.
- **Huddling & Social Dynamics:** Specialized tools to calculate and visualize colony huddling behavior, including density mapping and thermal/social simulations.
- **Custom Visualizations:** The agent autonomously generates matplotlib charts and graphs directly into its scientific reports.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/and0sha/PenguinAgent>
   cd PenguinAgent
   ```

2. **Install Python dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   **Note:** All packages required for SAM 3 to work must be pre-installed (see the `sam3` directory for its specific requirements or installation instructions).

3. **Install Ollama & Download Models:**
   PenguinAgent requires Ollama to run the local LLM.
   - Install Ollama from [ollama.com](https://ollama.com).
   - Pull the default Gemma model:
     ```bash
     ollama pull gemma4:26b
     ```
   *(Note: You can configure the specific model used in `config.env`)*

4. **Prepare Data Directories:**
   Ensure the following directories exist and contain relevant files:
   - `videos/`: Place your MP4 tracking videos here.
   - `data/`: Place scientific PDF literature here for the RAG database.

## Usage

1. **Start the application:**
   ```bash
   streamlit run main_gui.py
   ```

2. **Interface Navigation:**
   - **Main Dashboard:** Select videos, initialize the RAG database, and configure processing options.
   - **Track & Analyze:** Run the tracking algorithms on video files to generate `tracks.json` and annotated video.
   - **Scientific Reports:** Generate comprehensive markdown reports detailing behavioral anomalies and overall findings.
   - **Interactive Mode:** Ask the agent questions and mention specific tracks using the `@` symbol for targeted insights.

## Configuration

You can customize core paths and model selection by modifying the `config.env` file:
```env
OLLAMA_MODEL=gemma4:26b
SAM3_ROOT=./sam3
RAG_DATA_DIR=./data
VIDEOS_DIR=./videos
```

## Architecture

- **`main_gui.py`**: Streamlit frontend and entry point.
- **`behave_ai.py`**: The core `BehaviorAgent` class unifying tracking, RAG, reporting, and huddling mixins.
- **`behave_tracking.py`**, **`behave_report.py`**, **`behave_rag.py`**, **`behave_huddle.py`**: Modular mixins containing specialized agent skills.
- **`tools/`**: Sub-modules for computer vision tasks (SAM 3 tracking, SigLIP2, linear probe, etc.).

