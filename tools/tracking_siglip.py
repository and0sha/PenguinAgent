import os
import json
import tempfile
import multiprocessing
from pathlib import Path
from tools.tracking_config import SIGLIP_CONFIDENCE_THRESH
from tools.tracking_utils import pad_to_square, require_nonempty_file
from tools.tracking_results import TrackingError

def _run_siglip_subprocess(crop_paths_and_ids, output_json, confidence_threshold=SIGLIP_CONFIDENCE_THRESH):
    try:
        for path, obj_id in crop_paths_and_ids:
            require_nonempty_file(path, f"SigLIP crop for object {obj_id}")
        results = _classify_siglip_crops(crop_paths_and_ids, confidence_threshold)
        Path(output_json).write_text(json.dumps({"status": "ok", "results": results}), encoding="utf-8")
    except Exception as exc:
        detail = f"SigLIP classification failed: {type(exc).__name__}: {exc}"
        try:
            Path(output_json).write_text(json.dumps({"status": "error", "error": detail}), encoding="utf-8")
        except OSError:
            pass
        if isinstance(exc, TrackingError):
            raise
        raise TrackingError(detail) from exc


def _species_decision(zero_label, zero_probability, probe_row, class_indices, class_names, threshold):
    """Reject non-penguins; classifier probability columns follow classifier.classes_."""
    import math

    if not math.isfinite(float(zero_probability)):
        raise TrackingError("SigLIP returned a nonfinite probability.")
    if zero_label != "penguin (Spheniscidae)" or zero_probability < threshold:
        return "NOT_PENGUIN"
    if len(probe_row) != len(class_indices) or not len(probe_row):
        raise TrackingError("Species classifier returned an invalid probability row.")
    probabilities = [float(probability) for probability in probe_row]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        raise TrackingError("Species classifier returned invalid probabilities.")
    best_column = max(range(len(probabilities)), key=probabilities.__getitem__)
    encoded_class = class_indices[best_column]
    class_index = int(encoded_class)
    if class_index != encoded_class or class_index < 0 or class_index >= len(class_names):
        raise TrackingError("Species classifier and label encoder classes disagree.")
    label = str(class_names[class_index])
    if label == "NOT_PENGUIN" or probabilities[best_column] < threshold:
        return "NOT_PENGUIN"
    return label.replace(" ", "_")


def _infer_siglip_batch(model, processor, images, candidate_labels, device):
    import torch

    inputs = processor(text=candidate_labels, images=images, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        zero_shot_probs = torch.sigmoid(outputs.logits_per_image)
        # SigLIP2 forward already provides normalized image embeddings. Reusing
        # them avoids running the entire vision encoder a second time per crop.
        image_features = outputs.image_embeds
    return zero_shot_probs.cpu().numpy(), image_features.cpu().numpy()


def _classify_siglip_crops(crop_paths_and_ids, confidence_threshold):
    import torch
    import joblib
    from transformers import AutoProcessor, AutoModel
    from PIL import Image
    from tools.tracking_config import SIGLIP_BATCH_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "google/siglip2-base-patch16-224"
    model_path = Path(__file__).resolve().parents[1] / "penguin_classifier.pkl"
    try:
        classifier_data = joblib.load(model_path)
        classifier, label_encoder = classifier_data['classifier'], classifier_data['label_encoder']
        class_indices, class_names = classifier.classes_, label_encoder.classes_
    except Exception as exc:
        raise TrackingError(f"Cannot load the penguin species classifier at {model_path}: {exc}") from exc

    print(f"Loading SigLIP on {device}; at most {SIGLIP_BATCH_SIZE} crops per inference batch...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    candidate_labels = [
    "penguin (Spheniscidae)",
    # Antarctica & Subantarctica Predators/Mammals
    "leopard seal (Hydrurga leptonyx)", "killer whale (Orcinus orca)", "southern elephant seal (Mirounga leonina)",
    "Weddell seal (Leptonychotes weddellii)", "crabeater seal (Lobodon carcinophaga)", "Ross seal (Ommatophoca rossii)",
    "Antarctic fur seal (Arctocephalus gazella)", "blue whale (Balaenoptera musculus)", "humpback whale (Megaptera novaeangliae)",
    "sperm whale (Physeter macrocephalus)", "Antarctic minke whale (Balaenoptera bonaerensis)", "southern right whale (Eubalaena australis)",
    # Antarctica Fish/Krill
    "Antarctic krill (Euphausia superba)", "Antarctic toothfish (Dissostichus mawsoni)", "Antarctic silverfish (Pleuragramma antarcticum)",
    # Antarctica Birds
    "wandering albatross (Diomedea exulans)", "black-browed albatross (Thalassarche melanophris)", "snow petrel (Pagodroma nivea)",
    "southern giant petrel (Macronectes giganteus)", "skua (Stercorarius)", "kelp gull (Larus dominicanus)",
    "snowy sheathbill (Chionis albus)", "imperial shag (Leucocarbo atriceps)",
    # South America
    "South American sea lion (Otaria flavescens)", "South American fur seal (Arctocephalus australis)", "marine otter (Lontra felina)",
    "guanaco (Lama guanicoe)", "culpeo fox (Lycalopex culpaeus)", "Peruvian pelican (Pelecanus thagus)",
    "Inca tern (Larosterna inca)", "Guanay cormorant (Phalacrocorax bougainvillii)", "green-backed firecrown hummingbird (Sephanoides sephaniodes)",
    # South Africa
    "Cape fur seal (Arctocephalus pusillus)", "great white shark (Carcharodon carcharias)", "caracal (Caracal caracal)",
    "Cape gull (Larus vetula)", "rock hyrax (Procavia capensis)",
    # Australia & New Zealand
    "New Zealand fur seal (Arctocephalus forsteri)", "New Zealand sea lion (Phocarctos hookeri)", "Australasian gannet (Morus serrator)",
    "weka (Gallirallus australis)", "short-tailed shearwater (Ardenna tenuirostris)", "stoat (Mustela erminea)",
    # Galapagos
    "Galapagos sea lion (Zalophus wollebaeki)", "marine iguana (Amblyrhynchus cristatus)", "flightless cormorant (Phalacrocorax harrisi)",
    "blue-footed booby (Sula nebouxii)", "Sally Lightfoot crab (Grapsus grapsus)", "Galapagos hawk (Buteo galapagoensis)",
    "green sea turtle (Chelonia mydas)",
    # Abiotic / Non-living Objects
    "iceberg", "sea ice (pack ice)", "snow", "meltwater pool", "pebbles and small stones", "rocky cliffs",
    "sand dunes", "volcanic rock (lava)", "large boulders", "guano", "mud and peat soil", "ocean water",
    "sea foam", "driftwood", "seashell fragments", "unknown object"
    ]
    results = {}
    for offset in range(0, len(crop_paths_and_ids), SIGLIP_BATCH_SIZE):
        batch = crop_paths_and_ids[offset:offset + SIGLIP_BATCH_SIZE]
        images = []
        try:
            for path, obj_id in batch:
                require_nonempty_file(path, f"SigLIP crop for object {obj_id}")
                with Image.open(path) as source:
                    images.append(pad_to_square(source.convert("RGB")))
            zero_probs, features = _infer_siglip_batch(model, processor, images, candidate_labels, device)
            probe_probs = classifier.predict_proba(features)
            if len(zero_probs) != len(batch) or len(probe_probs) != len(batch):
                raise TrackingError("SigLIP or species classifier returned an incomplete batch.")
            for index, (_, obj_id) in enumerate(batch):
                row = zero_probs[index]
                best_zero = max(range(len(row)), key=lambda position: float(row[position]))
                results[obj_id] = _species_decision(candidate_labels[best_zero], float(row[best_zero]),
                                                   probe_probs[index], class_indices, class_names,
                                                   confidence_threshold)
        finally:
            for image in images:
                image.close()
    return results

def get_penguin_species_batch(crop_paths_and_ids, confidence_threshold=SIGLIP_CONFIDENCE_THRESH):
    if not crop_paths_and_ids:
        return {}

    for path, obj_id in crop_paths_and_ids:
        require_nonempty_file(path, f"SigLIP crop for object {obj_id}")
    expected_ids = {str(obj_id) for _, obj_id in crop_paths_and_ids}
    if len(expected_ids) != len(crop_paths_and_ids):
        raise TrackingError("SigLIP batch contains duplicate object identifiers.")
    with tempfile.TemporaryDirectory(prefix="penguin-siglip-") as temporary:
        temp_json_path = os.path.join(temporary, "results.json")
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=_run_siglip_subprocess, args=(crop_paths_and_ids, temp_json_path, confidence_threshold))
        try:
            p.start()
            p.join()
            require_nonempty_file(temp_json_path, f"SigLIP worker result (process exit {p.exitcode})")
            try:
                with open(temp_json_path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError) as exc:
                raise TrackingError(f"SigLIP worker returned invalid JSON: {exc}") from exc
            if not isinstance(data, dict) or p.exitcode != 0 or data.get("status") != "ok":
                detail = data.get("error", "missing success status") if isinstance(data, dict) else "invalid result object"
                raise TrackingError(f"SigLIP worker failed (exit {p.exitcode}): {detail}")
            results = data.get("results")
            if not isinstance(results, dict) or set(results) != expected_ids or any(
                not isinstance(label, str) or not label for label in results.values()
            ):
                raise TrackingError("SigLIP worker returned incomplete or invalid species results.")
            return results
        except TrackingError:
            raise
        except Exception as exc:
            raise TrackingError(f"SigLIP worker failed: {exc}") from exc
        finally:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
