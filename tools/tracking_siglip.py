import os
import json
import tempfile
import multiprocessing
from PIL import Image
from tools.tracking_config import SIGLIP_CONFIDENCE_THRESH
from tools.tracking_utils import pad_to_square

def _run_siglip_subprocess(crop_paths_and_ids, output_json, confidence_threshold=SIGLIP_CONFIDENCE_THRESH):
    import torch
    import numpy as np
    import joblib
    import traceback
    from transformers import AutoProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "google/siglip2-base-patch16-224"
    
    try:
        print(f"Phase 3 Subprocess: Loading SigLIP model to {device}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        
        print("Phase 3 Subprocess: Loading linear probe classifier...")
        classifier = None
        label_encoder = None
        try:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "penguin_classifier.pkl"))
            if not os.path.exists(model_path):
                model_path = "penguin_classifier.pkl"
            data = joblib.load(model_path)
            classifier = data['classifier']
            label_encoder = data['label_encoder']
        except Exception as e:
            print(f"Phase 3 Subprocess: Error loading classifier: {e}")

        results = {}
        valid_inputs = []
        
        for path, obj_id in crop_paths_and_ids:
            if not os.path.exists(path):
                results[obj_id] = "NOT_PENGUIN"
                continue
            image = Image.open(path).convert("RGB")
            image = pad_to_square(image)
            valid_inputs.append((image, obj_id))
            
        if valid_inputs:
            images = [item[0] for item in valid_inputs]
            obj_ids = [item[1] for item in valid_inputs]
            
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
            
            inputs = processor(text=candidate_labels, images=images, padding="max_length", return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                zero_shot_probs = torch.sigmoid(outputs.logits_per_image)
                
                image_features = model.get_image_features(pixel_values=inputs.pixel_values)
                if hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
            features_np = image_features.cpu().numpy()
            
            if classifier is not None and label_encoder is not None:
                probe_probs = classifier.predict_proba(features_np)
            else:
                probe_probs = None
                
            for i, obj_id in enumerate(obj_ids):
                best_zero_idx = torch.argmax(zero_shot_probs[i]).item()
                best_zero_prob = zero_shot_probs[i][best_zero_idx].item()
                best_zero_label = candidate_labels[best_zero_idx]
                
                if "penguin" in best_zero_label and probe_probs is not None:
                    best_idx = np.argmax(probe_probs[i])
                    best_prob = probe_probs[i][best_idx]
                    best_label = label_encoder.classes_[best_idx]
                    
                    if best_label == "NOT_PENGUIN" or best_prob < confidence_threshold:
                        results[obj_id] = "NOT_PENGUIN"
                    else:
                        results[obj_id] = best_label.replace(" ", "_")
                else:
                    if best_zero_prob > confidence_threshold:
                        clean_name = best_zero_label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                        results[obj_id] = clean_name
                    else:
                        results[obj_id] = "NOT_PENGUIN"
                        
    except Exception as e:
        print(f"SigLIP Subprocess Error:\n{traceback.format_exc()}")
        results = {obj_id: "NOT_PENGUIN" for _, obj_id in crop_paths_and_ids}

    with open(output_json, 'w') as f:
        json.dump(results, f)
        
    print("Phase 3 Subprocess: Completed. VRAM will now be freed.")

def get_penguin_species_batch(crop_paths_and_ids, confidence_threshold=SIGLIP_CONFIDENCE_THRESH):
    if not crop_paths_and_ids:
        return {}

    fd, temp_json_path = tempfile.mkstemp(suffix=".json")
    os.close(fd) 
    
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_siglip_subprocess, args=(crop_paths_and_ids, temp_json_path, confidence_threshold))
    p.start()
    p.join()
    
    results = {}
    
    if p.exitcode != 0 or not os.path.exists(temp_json_path):
        print("Warning: The SigLIP process has failed or was terminated.")
        results = {obj_id: "NOT_PENGUIN" for _, obj_id in crop_paths_and_ids}
    else:
        try:
            with open(temp_json_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading SigLIP results: {e}")
            results = {obj_id: "NOT_PENGUIN" for _, obj_id in crop_paths_and_ids}
            
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)
        
    return results