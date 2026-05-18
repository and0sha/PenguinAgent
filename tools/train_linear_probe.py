import os
import argparse
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/siglip2-base-patch16-224"

def pad_to_square(image):
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (128, 128, 128))
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image

def extract_features(dataset_path, processor, model, batch_size=32):
    features = []
    labels = []
    
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    image_paths = []
    image_labels = []
    
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, img_name))
                image_labels.append(cls)
                
    if not image_paths:
        raise ValueError("No images found in the specified dataset directory.")
        
    print(f"Found {len(image_paths)} images across {len(classes)} classes.")
    
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = image_labels[i:i+batch_size]
        
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = pad_to_square(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            
        if not images:
            continue
            
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            
            outputs = model.get_image_features(**inputs)
            
        
            if hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            else:
                image_features = outputs 
            

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        features.extend(image_features.cpu().numpy())
        labels.extend(batch_labels)
        
    return np.array(features), np.array(labels)

def train_linear_probe(dataset_path, output_model_path):
    print(f"Loading SigLIP-2 model to {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    print("Extracting embeddings from your cropped dataset...")
    X, y_text = extract_features(dataset_path, processor, model)
    
    print("Encoding labels (Latin names)...")
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    print(f"Training Logistic Regression classifier (classes: {len(le.classes_)})...")
    
    clf = LogisticRegression(
        max_iter=2000, 
        C=1.0, 
        class_weight='balanced', 
        solver='lbfgs', 
        random_state=42
    )
    clf.fit(X, y)
    
    print("\n" + "="*30)
    print("FINAL TRAINING PERFORMANCE:")
    print("="*30)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, target_names=le.classes_))
    

    print(f"Saving probe to {output_model_path}...")
    joblib.dump({
        'classifier': clf, 
        'label_encoder': le,
        'model_id': model_id 
    }, output_model_path)
    print("Successfully finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Linear Probe on top of SigLIP-2.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to 'gbif_cropped_dataset'.")
    parser.add_argument("--output", type=str, default="penguin_classifier.pkl", help="Output path for the .pkl file.")
    args = parser.parse_args()
    
    train_linear_probe(args.dataset, args.output)
