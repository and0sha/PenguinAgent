import os
import argparse
model_id = "google/siglip2-base-patch16-224"

def pad_to_square(image):
    from PIL import Image
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (128, 128, 128))
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image

def extract_features(dataset_path, processor, model, batch_size=32):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    device = next(model.parameters()).device
    features = []
    labels = []
    
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    image_paths = []
    image_labels = []
    
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        for img_name in sorted(os.listdir(cls_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, img_name))
                image_labels.append(cls)
                
    if not image_paths:
        raise ValueError("No images found in the specified dataset directory.")
        
    print(f"Found {len(image_paths)} images across {len(classes)} classes.")
    
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = []
        
        images = []
        for path, label in zip(batch_paths, image_labels[i:i+batch_size]):
            try:
                with Image.open(path) as source:
                    img = source.convert("RGB")
                padded = pad_to_square(img)
                if padded is not img:
                    img.close()
                img = padded
                images.append(img)
                batch_labels.append(label)
            except (OSError, ValueError) as e:
                print(f"Error loading {path}: {e}")
                continue
            
        if not images:
            continue
            
        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
        finally:
            for image in images:
                image.close()
        
        with torch.no_grad():
            
            outputs = model.get_image_features(**inputs)
            
        
            if hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            else:
                image_features = outputs 
            

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        batch_features = image_features.cpu().numpy()
        if batch_features.ndim != 2 or len(batch_features) != len(batch_labels) or not np.isfinite(batch_features).all():
            raise ValueError("The model returned invalid or misaligned image features.")
        features.extend(batch_features)
        labels.extend(batch_labels)
        
    if not features:
        raise ValueError("No readable images remain in the dataset.")
    return np.array(features), np.array(labels)

def train_linear_probe(dataset_path, output_model_path):
    import torch
    import joblib
    from transformers import AutoProcessor, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SigLIP-2 model to {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    print("Extracting embeddings from your cropped dataset...")
    X, y_text = extract_features(dataset_path, processor, model)
    if len(set(y_text)) < 2:
        raise ValueError("Training requires readable images from at least two classes.")
    
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
    print("TRAINING-SET PERFORMANCE (not held-out validation):")
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
