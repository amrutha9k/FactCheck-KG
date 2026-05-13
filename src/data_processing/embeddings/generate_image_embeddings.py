"""
This script extracts image features using pre-trained CLIP and ResNet50 models.
It processes the PolitiFact data splits, validates image integrity, and 
generates high-dimensional embeddings. The output is saved as half-precision 
numpy arrays to optimize storage.
"""

import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms

# Determine project root and directory paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
splits_dir = os.path.join(project_root, "data", "splits")
emb_root = os.path.join(project_root, "data", "embeddings")

# Execution configuration
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard ResNet transformation pipeline
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_visual_models():
    """Initializes and returns the CLIP and ResNet50 feature extractors."""
    print(f"Initializing models on {DEVICE}...")
    
    # Setup CLIP
    clip_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(DEVICE).eval()
    
    # Setup ResNet50 (removing the final classification layer)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_features = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet_features = resnet_features.to(DEVICE).eval()
    
    return processor, clip_model, resnet_features

def run_embedding_generation():
    clip_proc, clip_model, resnet_model = load_visual_models()
    
    # Create subdirectories for model-specific embeddings
    clip_dir = os.path.join(emb_root, "clip")
    resnet_dir = os.path.join(emb_root, "resnet50")
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(resnet_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    labels = ["fake", "real"]

    for split in splits:
        for label in labels:
            id_file = os.path.join(emb_root, f"{split}_{label}_ids.npy")
            csv_file = os.path.join(splits_dir, split, f"{label}.csv")
            
            if not os.path.exists(id_file) or not os.path.exists(csv_file):
                continue
                
            master_ids = np.load(id_file, allow_pickle=True).astype(str).tolist()
            df = pd.read_csv(csv_file)
            df['id'] = df['id'].astype(str)
            
            # Create lookups for article metadata
            path_map = dict(zip(df['id'], df['image_path']))
            exist_map = dict(zip(df['id'], df['Image_exist']))
            
            num_samples = len(master_ids)
            print(f"Generating image embeddings for {split} {label} ({num_samples} samples)")

            # Initialize output buffers
            all_clip_feats = np.zeros((num_samples, 768), dtype=np.float32)
            all_resnet_feats = np.zeros((num_samples, 2048), dtype=np.float32)

            for start_idx in tqdm(range(0, num_samples, BATCH_SIZE), desc="Inference"):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                batch_ids = master_ids[start_idx:end_idx]
                
                resnet_batch_tensors = []
                clip_batch_images = []
                valid_mask = []

                for aid in batch_ids:
                    rel_path = str(path_map.get(aid, ""))
                    abs_path = os.path.join(project_root, rel_path)
                    is_available = str(exist_map.get(aid, "no")).lower() == "yes"
                    
                    success = False
                    if is_available and os.path.exists(abs_path):
                        try:
                            with Image.open(abs_path) as img:
                                rgb_image = img.convert("RGB")
                                resnet_batch_tensors.append(resnet_transform(rgb_image))
                                clip_batch_images.append(rgb_image)
                                success = True
                        except Exception:
                            pass
                    
                    if not success:
                        # Fallback for missing or corrupted images
                        resnet_batch_tensors.append(torch.zeros(3, 224, 224))
                        clip_batch_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                    
                    valid_mask.append(success)

                # Execute ResNet50 inference
                r_input = torch.stack(resnet_batch_tensors).to(DEVICE)
                with torch.no_grad():
                    r_out = resnet_model(r_input).squeeze().cpu().numpy()
                    if len(batch_ids) == 1: r_out = r_out.reshape(1, -1)

                # Execute CLIP inference
                c_input = clip_proc(images=clip_batch_images, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    c_out = clip_model.get_image_features(**c_input)
                    # Normalize visual vectors as per CLIP training objectives
                    c_out = c_out / c_out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    c_out = c_out.cpu().numpy()

                # Map results back to the master output arrays
                for batch_idx, is_valid in enumerate(valid_mask):
                    if is_valid:
                        all_resnet_feats[start_idx + batch_idx] = r_out[batch_idx]
                        all_clip_feats[start_idx + batch_idx] = c_out[batch_idx]

            # Save split embeddings in half-precision to conserve disk space
            np.save(os.path.join(resnet_dir, f"{split}_{label}_image.npy"), all_resnet_feats.astype(np.float16))
            np.save(os.path.join(clip_dir, f"{split}_{label}_image.npy"), all_clip_feats.astype(np.float16))

if __name__ == "__main__":
    run_embedding_generation()
