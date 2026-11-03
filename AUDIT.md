# PenguinAgent Final Audit

Date: November 3, 2026. Comparison baseline: `457cbcf54e5ea5d35b911a1e27a3b46b81a61a6b`.

## Outcome and verification scope

Confirmed defects in security, data provenance, track association, failure handling, reporting, RAG and classifier training have been fixed. This change includes fixes from the earlier work and the follow-up audit. The primary target remains Linux / Python 3.12 / NVIDIA CUDA; the application starts with `streamlit run main_gui.py`. The SAM 3 submodule remains at its original pinned commit, and its source code has not been modified.

This audit covers testable application logic. It does not prove that all bugs have been eliminated or establish model quality. A full dependency installation and execution of Streamlit, SAM 3, SigLIP, BGE-M3 and Ollama on the target Linux/GPU environment were not performed during this audit. Unit tests alone do not establish that the prepared runtime is operational.

## Original findings A01–A14

| Priority / ID | Fix | Verification and limitation |
|---|---|---|
| P1 · A01 | Removed execution of model-generated Python. Charts are generated only through `generate_tracking_chart` with validated arguments. | Tests cover forbidden fields/code/paths, types, nonnumeric values, data limits, actual PNG output and HTML escaping. |
| P1 · A02 | Restored `.gitmodules`, the SAM 3 checkout and support for `SAM3_ROOT`/`SAM3_CHECKPOINT`; the selected configuration file reaches the worker. | Tests cover paths, configuration precedence, instance isolation and propagation into the child process. Real inference requires separate acceptance testing. |
| P1 · A03 | Each result is bound to the source video hash, run ID, parameters and artifact checksums. The GUI resets derived state when the video, directory or file contents change. | Tests include replacement of a video with the same filename, size and mtime; reports revalidate the current run. |
| P1 · A04 | FFmpeg, SAM, classification, encoding and publication failures stop processing. Earlier results cannot be presented as a newly successful run. | Separate tests cover empty scenes, partial files, failure after an earlier success, application restart, concurrent attempts and disk errors. |
| P1 · A05 | Only the parent process assigns global IDs. Boundary association is one-to-one, uses visibility in adjacent boundary frames and rejects ambiguous matches. | Tests cover crossings, reused local IDs, new arrivals, disappearances, empty chunks and consecutive chunks. Geometry alone cannot guarantee biological identity. |
| P1 · A06 | Replaced the shared `/tmp` location with isolated run and chunk directories. Crop write failures are distinct from a “not a penguin” classification. | Tests cover failed writes, missing crops and prevention of one object's crop overwriting another's. |
| P2 · A07 | Fixed the `json` scope error in the huddle callback while retaining the safe chart tool. | Tests execute the actual callback with the external LLM replaced by a stub. |
| P2 · A08 | Text reports and HTML are written as UTF-8 through a temporary file and atomic replacement. | Tests cover Unicode and preservation of the previous file when publication fails. |
| P1 · A09 | Each report image includes its source frame index, timestamp and dimensions; the number of images is bounded. Removed the incorrect “one frame per second” claim. | A long-video case is tested. VFR source timestamps remain nominal (`frame_index/fps`); proxy coordinates are explicitly distinguished from source images, with no claim of metric calibration. |
| P1 · A10 | Analysis receives numeric observations, actual interval velocities, duration and information about data omitted from the preview. | Stationary and moving tracks produce different context; the user's global question is retained. Numeric U/A/R scores are not requested without baseline evidence. |
| P2 · A11 | Failed RAG initialization clears the previous store and raises an explicit error. The GUI reports success only after a valid load. | Tests cover builder failure, missing/corrupt files, loader errors and directory changes. |
| P2 · A12 | The PDF loader and fingerprint use the same recursive file inventory, including content hashes and index settings. | Tests cover nested `.PDF` files, deletion of all documents, changes preserving size/mtime and changes during a build. |
| P2 · A13 | Chat reads both report types only from the supplied current run, with bounded excerpts. | It does not scan other videos or historical runs; huddle context is available alongside anomaly context. |
| P2 · A14 | A training image's label is appended only after successful decoding; feature dimensions, counts and finite values are validated. | A corrupt image between classes/batches cannot shift the labels. Empty data and fewer than two readable classes cannot proceed to training. |

## Additional fixes from the follow-up audit

- **P1 — proxy duration.** Output-only `-r` added extra frames: reducing a two-second test clip from 5 to 2.5 FPS produced 7 frames instead of 5. The conversion now uses the `fps` filter and only the video stream. Real FFmpeg checks verify duration and preservation of frame counts during segmentation.
- **P1 — boundary velocities.** Timestamps use actual frame counts and FPS, including fractional seconds. Velocities are recomputed after global association as `dx/dt`, without resetting at every chunk. Contradictions between telemetry and frame annotations are errors.
- **P1 — non-penguin classification.** Confident “seal,” “iceberg” and other nuisance-class predictions now produce `NOT_PENGUIN` instead of a new species label. Probe probability columns are mapped through `classifier.classes_`.
- **P1 — classifier GPU memory.** Crops are processed in batches of 16, configurable with `PENGUIN_SIGLIP_BATCH_SIZE`; images are closed after each batch. Already-computed normalized embeddings replace a second vision-encoder call. This reduces operation count and bounds batch size; it is not a measured FPS improvement.
- **P2 — rendering and long videos.** Removed `gc.collect()`/`empty_cache()` from every SAM frame. Visual trails are limited to 30 points while all telemetry is retained. Chunks are sorted by numeric index, including indices beyond 999.
- **P1 — unsupported scientific claims.** Arbitrary animation and physical parameter estimates derived from group shape have been replaced by an offline observation viewer. It does not invent animals, wind, heat exchange or Pe estimates from aspect ratio. User parameters are labeled as assumptions. Reports bound cumulative tool context and literature excerpts.
- **P1 — RAG pickle.** New index generations contain `index.faiss` and JSON document records; the old `index.pkl` is never deserialized. A manifest verifies files before loading, and generation publication is atomic. Native FAISS indexes must also come from a trusted local directory.
- **P2 — working-directory independence.** Configuration, data and the builder are resolved relative to the project; separate agent instances do not modify shared dotenv state. The configured Ollama model remains selected in the GUI. Tag parsing does not intercept email addresses or repeat identical requests.
- **P2 — Linux preparation.** Completed the bootstrap, import/version checks and a separate CUDA matrix multiplication check. Seed dependencies account for SAM 3's NumPy<2 constraint, OpenCV, Transformers/SigLIP2, the serialized classifier and `pkg_resources`. These are direct pins and application constraints, not a fully tested transitive lock.
- **P2 — smoke diagnostics.** Corrupt worker measurements no longer prevent the final JSON summary from being written. Missing detections or missing evidence of CUDA execution produce `inconclusive`, not success.
- **Maintenance.** Added an Ubuntu/Windows regression workflow, `.gitignore` and LF line endings for Linux scripts. Removed 17 generated `.pyc` files from the Git index; source videos, the dataset and classifier were preserved.

## Verification performed

- Full regression suite: **252 tests, PASS** on Windows / Python 3.12. ML models and external services are replaced by stubs where explicitly documented in the tests.
- Real Matplotlib renders PNGs; Pillow decodes test images; FFmpeg/FFprobe encode, segment and decode video.
- A separate temporary environment exercised real FAISS 1.15.1, LangChain Community 0.4.2/Core 1.7.4 and NumPy 1.26.4: writing/reading the new format, exact metadata preservation, similarity/MMR search and rejection of modified JSON. It used deterministic test embeddings; BGE-M3 was not downloaded.
- CPU preparation of the first two seconds of the actual `videos/8.mp4` produced a 640×360 proxy at 2.5 FPS, with 5 frames and a duration of 2.000000 seconds. The entire proxy decoded without error. This was not GPU tracking.
- Bash checks bootstrap syntax and dry-run behavior; these tests do not install packages or execute CUDA. Windows Git Bash is not a substitute for full Linux testing.
- `--help` works for five CLI entry points from outside the project directory without ML dependencies. The SAM 3 preflight verifies source/tokenizer paths and the pinned SHA, not weights access or GPU availability.
- Application, tool, script and test Python sources pass AST parsing. `git diff --check` passes. The SAM 3 submodule working tree is clean.
- The viewer's JavaScript is tested through Node VM with a DOM adapter: timestamp changes, missing detections and safe handling of arbitrary IDs. This is a functional JavaScript check, not visual acceptance testing of Streamlit in a browser.

At the time of this audit, GitHub CI and Linux/GPU validation had not yet run. The Ubuntu/Windows workflow was prepared for execution after publication.

## Prioritized Linux acceptance checks

1. **P1 — full GPU smoke test.** In an isolated Linux venv, run the README bootstrap, install `requirements.txt`, run `pip check`, save a complete `pip freeze`, authenticate locally with Hugging Face and run both stages below. Inspect CUDA measurements and the annotated video. This is the primary remaining operational acceptance criterion.
2. **P1 — end-to-end GUI and models.** After the smoke test, open Streamlit, process two different videos, repeat a run with a controlled failure, generate both reports and test chat/RAG against the real services. Unit tests do not verify the integration of all installed packages.
3. **P2 — identity and species quality.** Annotate a small independent dataset containing crossings and occlusions; measure ID switches, false detections and species classification quality. Geometric association is conservative but does not replace re-identification. Metrics on training images are now explicitly labeled as training-set performance, not validation.
4. **P2 — scientific calibration.** Physical speeds and densities require camera geometry, scale and independent measurements. Matching source frames from VFR video requires actual presentation timestamps. A researcher must review LLM-generated text and citations.
5. **P3 — resources and reproducibility.** After a successful Linux run, lock the resolved transitive dependencies and model weight versions/hashes. Profile RAM/VRAM and latency: SAM still holds a chunk's frames in memory; strict provenance checks reread files, and previous result/index generations are retained. Safe history management and recovery from hung processes can be added separately; currently, a leftover lock must be removed manually after checking that no process is active.

Acceptance commands, from the project root with the Linux environment activated:

```bash
python -m pip check
python -B scripts/check_sam3.py
python -B scripts/check_sam3_wsl_environment.py --gpu-check --output ../outputs/linux-environment.json
python -B scripts/smoke_tracking.py --video videos/8.mp4 --output-dir ../outputs/gpu-smoke --seconds 12 --height 360 --stage sam3
python -B scripts/smoke_tracking.py --video videos/8.mp4 --output-dir ../outputs/gpu-smoke --seconds 12 --height 360 --chunk-seconds 4 --stage pipeline
streamlit run main_gui.py
```

## Compatibility and migration

GUI commands and public method names are preserved. `run_tracking` returns paths inside a unique run directory; the `output_dir` argument for tracking and report generation still refers to the base output directory. `analyze_behavior` requires a selected, validated context and the actual current run directory, as supplied by the GUI.

Legacy loose result files and version 1 manifests cannot be used as current results; the video must be reprocessed. Legacy RAG pickle caches are rebuilt. The huddle artifact retains the filename `huddle_simulation.html` for interface compatibility, but displays observations rather than a physical simulation. Personal tokens, weights, venvs and temporary results are excluded from the commit.

## Verified dependency sources

- [SAM 3 README at the pinned revision](https://github.com/facebookresearch/sam3/blob/847e1a3b15115a04c87c0760297f044f0555d970/README.md) and the local `pyproject.toml`.
- [SigLIP2 forward in Transformers 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/siglip2/modeling_siglip2.py): the returned normalized image embeddings are reused.
- [PyTorch 2.10.0](https://pypi.org/project/torch/2.10.0/), [Transformers 4.57.6](https://pypi.org/project/transformers/4.57.6/), [scikit-learn 1.8.0](https://pypi.org/project/scikit-learn/1.8.0/): release metadata for the selected direct dependencies.
- [actions/checkout](https://github.com/actions/checkout/commit/d23441a48e516b6c34aea4fa41551a30e30af803), [actions/setup-python](https://github.com/actions/setup-python/commit/ece7cb06caefa5fff74198d8649806c4678c61a1): verified pinned workflow commit SHAs.
