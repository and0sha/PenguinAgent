# PenguinAgent 🐧

PenguinAgent is an AI-powered ornithologist application designed to automate the tracking, analysis, and behavioral reporting of Antarctic penguin colonies. It leverages state-of-the-art computer vision and large language models to extract kinematics from video footage, identify anomalies, and contextualize findings against scientific literature.

## Features

- **Video Telemetry & Tracking:** Uses SAM 3 to extract image-space positions, velocities and mask areas, followed by SigLIP2 and a linear probe for species predictions. Model predictions and cross-chunk identities require validation on the target footage.
- **Multimodal AI Analysis:** Uses Ollama (running the `gemma4:26b` model by default) to interpret video frames and telemetry data to generate comprehensive scientific reports and anomaly assessments.
- **Retrieval-Augmented Generation (RAG):** Uses LangChain and FAISS to retrieve excerpts from local scientific PDFs. Reports can cite these excerpts for review against the source papers.
- **Interactive Chat & Queries:** A Streamlit-based GUI allows researchers to interrogate the AI about specific penguins (e.g., `@penguin_1`), general behavioral literature, and complex dynamics.
- **Huddling & Social Dynamics:** Reports measured group geometry and shows recorded tracks in an offline observation viewer. Video pixels alone do not establish physical density, temperature, energy or thermodynamic quantities.
- **Custom Visualizations:** Reports can request predefined Matplotlib charts through a validated tool interface.

## Installation

1. **Clone the repository and its pinned SAM 3 source:**
   ```bash
   git clone --recurse-submodules https://github.com/and0sha/PenguinAgent
   cd PenguinAgent
   ```
   For an existing clone, run:
   ```bash
   git submodule sync -- sam3
   git submodule update --init -- sam3
   git -C sam3 rev-parse HEAD
   ```
   The bundled SAM 3 source is [Meta's official repository](https://github.com/facebookresearch/sam3), pinned to `847e1a3b15115a04c87c0760297f044f0555d970`. Keep this revision when restoring the dependency.

2. **Prepare a SAM 3 environment:**
   Ubuntu/Linux with an NVIDIA GPU is the recommended inference environment. On Windows, use WSL2 with an Ubuntu distribution and GPU support. Native Windows is experimental: it needs a compatible Triton setup and separate validation; installing PyTorch alone does not establish SAM 3 compatibility.

   Use Python 3.12 with the pinned SAM 3 revision and the CUDA 12.8 PyTorch wheels selected by the bootstrap. On Ubuntu 24.04, install the OS prerequisites and prepare an isolated environment:
   ```bash
   sudo apt-get update
   sudo apt-get install python3.12 python3.12-venv python3-pip ffmpeg git build-essential libgl1 libglib2.0-0t64
   bash scripts/setup_sam3_wsl.sh --venv "$HOME/.venvs/penguinagent" --gpu-check
   source "$HOME/.venvs/penguinagent/bin/activate"
   python -m pip install -r requirements.txt
   python -m pip check
   ```
   Despite its historical name, `setup_sam3_wsl.sh` supports native Linux as well as WSL2. It installs the tracking dependencies and editable SAM 3 source, checks imports and optionally runs a small CUDA matrix multiplication. It does not download model weights, install drivers or invoke `sudo`. Use `--dry-run` to inspect its commands. The additional `requirements.txt` step installs the GUI, Ollama client, plotting and RAG packages.

   The requirements pin the model runtime, including NumPy 1.26 and a compatible OpenCV version; application packages have bounded major versions. They are not a complete transitive lock. The bootstrap writes `sam3-bootstrap-freeze.txt` and an environment report inside the venv; after installing the full application, save `python -m pip freeze --all` as well. The [upstream installation documentation](https://github.com/facebookresearch/sam3/blob/847e1a3b15115a04c87c0760297f044f0555d970/README.md#installation) describes SAM 3's GPU requirements. SAM 3 inference has no CPU fallback in this application.

   Install FFmpeg and verify the video encoders:
   ```bash
   ffmpeg -version
   ffmpeg -encoders
   ```
   The video pipeline uses FFmpeg's `libx264` encoder and OpenCV's `mp4v` writer.

3. **Prepare SAM 3 weights and validate paths:**
   Request access to [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3), review its terms, then authenticate locally with `hf auth login`. With `SAM3_CHECKPOINT` empty, the upstream builder uses the Hugging Face cache or downloads `sam3.pt` and `config.json` on first inference. Keep credentials and model weights outside Git.

   For offline SAM 3 loading, provide an existing local `sam3.pt` using `SAM3_CHECKPOINT` in `config.env`; the builder then loads that file directly. Other application models, including SigLIP and Ollama, need their own local files or caches for a fully offline run.
   ```bash
   python -B scripts/check_sam3.py
   python -B -m unittest discover -s tests -p 'test_sam3_runtime.py' -v
   ```
   These checks use only the Python standard library and do not import ML packages or download weights. They validate source/checkpoint paths; run a short tracking clip in the target GPU environment to verify inference. To inspect an alternate config file, use `python -B scripts/check_sam3.py --config path/to/config.env`.

4. **Install Ollama & Download Models:**
   PenguinAgent requires Ollama to run the local LLM.
   - Install Ollama from [ollama.com](https://ollama.com).
   - Pull the default Gemma model:
     ```bash
     ollama pull gemma4:26b
     ```
   *(Note: You can configure the specific model used in `config.env`)*

5. **Prepare Data Directories:**
   Ensure the following directories exist and contain relevant files:
   - `videos/`: Place your MP4 tracking videos here.
   - `data/`: Place scientific PDF literature here for the RAG database.

## Updating an existing Ubuntu checkout

From your existing PenguinAgent directory, inspect local changes before pulling:

```bash
git status --short
git switch main
git pull --ff-only origin main
git submodule sync -- sam3
git submodule update --init --recursive
git -C sam3 rev-parse HEAD
```

Commit or back up any local source/configuration changes first. If Git reports a conflict or diverged history, resolve it before continuing; no reset or forced overwrite is needed. SAM 3 should remain at `847e1a3b15115a04c87c0760297f044f0555d970`.

Use the installation steps above to create or update the separate Python 3.12 environment on Ubuntu 24.04, then run the [short GPU smoke test](#short-gpu-smoke-test). `nvidia-smi` must work before the bootstrap's CUDA check. Hugging Face browser login does not authenticate the Linux environment: run `hf auth login` inside the activated venv, enter a token locally for an account with access to `facebook/sam3`, and verify with `hf auth whoami`.

See [AUDIT.md](AUDIT.md) for fixed issues, validation evidence and remaining Linux/GPU acceptance checks.

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
SAM3_CHECKPOINT=
RAG_DATA_DIR=./data
VIDEOS_DIR=./videos
```

`SAM3_ROOT` points to a SAM 3 repository checkout containing `sam3/model_builder.py` and `sam3/assets/bpe_simple_vocab_16e6.txt.gz`. `SAM3_CHECKPOINT` may point to a local `sam3.pt`; leave it empty for automatic Hugging Face loading. Relative configuration paths, video/data directories and SAM 3 paths are resolved from the PenguinAgent project directory, independent of the current working directory. Absolute paths are also supported. Environment variables override file values. `BehaviorAgent(config_path="custom.env")` passes the selected file to spawned SAM 3 workers without changing process-wide environment settings.

Use literal paths for these settings; `${VARIABLE}` interpolation is not supported. In Windows env files, prefer forward slashes (`C:/models/sam3`) or single quotes (`'C:\models\sam3'`): dotenv interprets escape sequences such as `\t` inside double quotes.

For example, an offline checkout can use:
```env
SAM3_ROOT=/opt/models/sam3
SAM3_CHECKPOINT=/opt/models/weights/sam3.pt
```

## Architecture

- **`main_gui.py`**: Streamlit frontend and entry point.
- **`behave_ai.py`**: The core `BehaviorAgent` class unifying tracking, RAG, reporting, and huddling mixins.
- **`behave_tracking.py`**, **`behave_report.py`**, **`behave_rag.py`**, **`behave_huddle.py`**: Modular mixins containing specialized agent skills.
- **`tools/`**: Sub-modules for computer vision tasks (SAM 3 tracking, SigLIP2, linear probe, etc.).

## Tracking runs and saved results

Each attempt has a unique run ID and private temporary video/crop files. A complete run is published under the selected output directory:

```text
video_output/
  tracking_manifest.json
  runs/
    <run_id>/
      tracks.json
      annotated.mp4
      tracks_species_mapping.txt  # only when compact labels are used
      run_manifest.json
      final_scientific_report.md  # generated separately
      final_huddle_report.md      # generated separately
```

`tracking_manifest.json` describes the latest attempt (`running`, `failed`, or `success`). It binds a successful run to the original video path and SHA-256 contents, processing parameters and artifact checksums. Reports and the UI validate that receipt instead of trusting cached tracks or fixed filenames. Replacing a video under the same filename also invalidates previous results.

Starting a rerun makes previous results unavailable as the current analysis. Failures in FFmpeg, SAM 3, classification, encoding, validation or publication propagate as errors. Previous completed run files are retained as history; a failed attempt never silently falls back to them, including after an application restart. A genuinely empty scene is a successful empty tracking result only when the video pipeline completed successfully.

Older loose `tracks.json`/`annotated.mp4` files without a manifest remain on disk but cannot prove which input produced them; reprocess the video to use it as a current verified analysis. JSON/video output paths returned by `run_tracking` now point inside the published run directory. The `output_dir` argument for tracking and reports still denotes the selected **base output directory**.

If a process stops abruptly, a `.tracking.lock` file may remain. The application then blocks reuse of results. Confirm that no tracking process is running and check the output directory before removing that lock and retrying; do not remove completed run files. If a failed status cannot be saved, the lock is deliberately retained to prevent old results from being accepted.

Video reports and chat are scoped to the current validated run. General scientific literature questions remain available without tracking and do not collect reports from other videos.

## Scientific context and reports

RAG discovers PDF files recursively, including uppercase `.PDF`, and fingerprints their contents along with index settings. A successful build publishes a complete index generation atomically. A failed initialization clears the active in-memory index and surfaces an error. New indexes use native FAISS vectors with JSON document records; legacy `index.pkl` caches are rebuilt without unpickling them. Only use trusted local index files. Embeddings run on CPU to leave GPU memory available for tracking and Ollama. If a crashed build leaves `data/faiss_index/.build.lock`, first confirm no indexing process is active before removing that lock.

Analysis receives bounded numeric evidence with timestamps, positions, velocities and mask areas, plus explicit sampling limits. Values use proxy pixels, pixels/second and square pixels; they are not calibrated physical measurements. Reports select frames by elapsed timestamp, including times above one minute. The huddle HTML is an offline viewer of observed detections, with gaps preserved, rather than a physical simulation. LLM-generated interpretations and citations still need scientific review.

## Track identities and velocities across chunks

SAM 3 object IDs belong to one chunk. Only the parent process assigns global video IDs, using one mapping for telemetry, classification crops and rendered text labels. It matches detections in the new chunk's first frame against animals visible in the preceding chunk's final frame. A constant-velocity prediction from the last two observed centroids helps preserve trajectories when animals cross; a single observation uses its last position.

An association must be a mutual nearest match inside the distance gate, with a clear advantage over competing candidates in both directions. Ties and near ties receive fresh global IDs. Later arrivals also receive fresh IDs, and IDs are never recycled after disappearance or an empty chunk. Crops are isolated per chunk before being mapped to global IDs, so a reused local ID cannot overwrite another animal's classification image.

This is conservative geometric association, not biological re-identification. Occlusion, indistinguishable crossings or abrupt motion can still split a real animal's trajectory or produce an incorrect association. Mask/trail colors are local visualization colors and can change between chunks; use the global text labels to follow track identity.

Telemetry contains every observed proxy frame. `second` is elapsed video time, including fractional seconds, calculated from the actual frame count and proxy FPS. After all chunks are assembled, `vx` and `vy` are recomputed as centroid displacement divided by elapsed seconds, in proxy-image pixels per second. A chunk boundary does not reset velocity. The first observation of a new global track uses a zero vector; a gap gives the mean velocity over the observation interval, without interpolation.

Before stitching, the parent checks that chunk input/output frame counts and FPS agree, and binds each telemetry sample to the same local object and centroid in its annotated frame. Contradictory worker data is an error, even when frame-count errors in separate chunks would cancel out in the final video.

The current manifest schema is version **2**, requiring these identity and velocity rules. Results from schema version 1 remain as history but must be reprocessed before reports or the UI accept them as the current analysis.

## Regression checks

The regression suite runs without SAM 3 weights or a GPU. Use a lightweight environment with Matplotlib for chart tests:
```bash
python -m venv .venv-tests
source .venv-tests/bin/activate
python -m pip install matplotlib
python -B -m unittest discover -s tests -v
```
On Windows, activate with `.venv-tests\Scripts\Activate.ps1`. The [regression workflow](.github/workflows/regression-tests.yml) runs these tests on Ubuntu and Windows without downloading model weights or checking out the SAM 3 submodule.

The suite includes video replacement with the same filename/size/mtime, cached and legacy data, failed reruns and restart recovery, partial outputs, FFmpeg/worker/classification errors, run publication, concurrent attempts and source/artifact replacement during validation. Real FFmpeg checks cover errors and fractional-FPS conversion/segmentation when FFmpeg and ffprobe are available; other pipeline tests use explicit ML/video stubs. Additional regressions cover corrupt training images, bounded classification batches, RAG cache invalidation, report evidence and offline viewer escaping.

Chunk regressions run the actual parent association, velocity calculation and renderer with synthetic worker results. They cover crossing trajectories, shuffled/reused local IDs, late arrivals, ambiguous matches, disappearance and empty chunks, crop/label consistency, twelve consecutive chunks, short chunks, sparse observations, FPS values of 0.5, 2 and 2.5, and contradictory worker output. Separate worker tests verify per-frame timestamps and that legacy global-ID arguments cannot trigger a second association.

Report charts use the structured `generate_tracking_chart` tool with `chart_type` set to `trajectory`, `speed`, `area`, or `population`, and optional `track_id` and `title` values. Custom Python chart code is no longer supported.

## Short GPU smoke test

After preparing the Linux GPU environment and obtaining SAM 3 model access, run the real model without loading Ollama or the RAG database:

```bash
python -B scripts/smoke_tracking.py --video videos/8.mp4 --output-dir ../outputs/gpu-smoke --seconds 12 --height 360 --stage sam3
python -B scripts/smoke_tracking.py --video videos/8.mp4 --output-dir ../outputs/gpu-smoke --seconds 12 --height 360 --chunk-seconds 4 --stage pipeline
```

The `sam3` stage checks inference on one clip with local IDs. The `pipeline` stage also runs SigLIP classification, global ID association across chunks, velocity calculation and final rendering. Each attempt creates a new `smoke-*` directory with artifacts, `summary.json` and CUDA peak memory measurements from the actual SAM 3 processes. Exit code 0 means tracks and GPU allocation were observed, 2 means the test is inconclusive (including zero detections), and 1 means failure. Successful execution still requires visual inspection of the resulting tracks to assess identity accuracy.

The optional `PENGUIN_TRACKING_FPS`, `PENGUIN_TRACKING_HEIGHT` and `PENGUIN_TRACKING_CHUNK_SECONDS` environment variables configure the regular pipeline and are inherited by spawned workers. `PENGUIN_SIGLIP_BATCH_SIZE` controls classification batch size (default 16); reduce it if classification runs out of GPU memory. The smoke CLI sets video sampling parameters from its arguments without editing `config.env` or the source defaults. `--help` works without installing ML packages. These preparation and regression checks do not establish that GPU inference has succeeded.

## License
This project is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
The SAM 3 submodule and weights have their own [SAM License](https://github.com/facebookresearch/sam3/blob/847e1a3b15115a04c87c0760297f044f0555d970/LICENSE).
