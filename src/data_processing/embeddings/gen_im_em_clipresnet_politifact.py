import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms

SPLITS = ["train", "val", "test"]
LABELS = ["fake", "real"]
BATCH_SIZE = 16
DEVICE = "cpu"

BASE_DIR = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/splits"
OUTPUT_DIR = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/embeddings"

# ResNet Preprocessing
resnet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_resnet_model():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    return resnet.to(DEVICE).eval()

def generate_image_embeddings():
    print("Loading models...")
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_proc = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(DEVICE).eval()
    
    resnet_model = get_resnet_model()
    
    clip_out_dir = os.path.join(OUTPUT_DIR, "clip")
    resnet_out_dir = os.path.join(OUTPUT_DIR, "resnet50")
    os.makedirs(clip_out_dir, exist_ok=True)
    os.makedirs(resnet_out_dir, exist_ok=True)

    for split in SPLITS:
        for label in LABELS:
            id_path = os.path.join(OUTPUT_DIR, f"{split}_{label}_ids.npy")
            csv_path = os.path.join(BASE_DIR, split, label, f"{label}_data.csv")
            
            if not os.path.exists(id_path) or not os.path.exists(csv_path):
                continue
                
            master_ids = np.load(id_path, allow_pickle=True).astype(str).tolist()
            df = pd.read_csv(csv_path)
            
            # Map ID to image path and exist flag
            df['id'] = df['id'].astype(str)
            path_lookup = dict(zip(df['id'], df['image_path']))
            exist_lookup = dict(zip(df['id'], df['Image_exist']))
            
            total_items = len(master_ids)
            print(f"\nProcessing Images: {split} {label} ({total_items} items)")

            clip_embeddings = np.zeros((total_items, 768), dtype=np.float32)
            resnet_embeddings = np.zeros((total_items, 2048), dtype=np.float32)

            for i in tqdm(range(0, total_items, BATCH_SIZE)):
                batch_ids = master_ids[i : i + BATCH_SIZE]
                
                resnet_tensors = []
                clip_images = []
                valid_mask = []

                for aid in batch_ids:
                    img_path = str(path_lookup.get(aid, ""))
                    exists = str(exist_lookup.get(aid, "no")).strip().lower() == "yes"
                    
                    is_valid = False
                    if exists and os.path.exists(img_path):
                        try:
                            with Image.open(img_path) as img:
                                rgb_img = img.convert("RGB")
                                resnet_tensors.append(resnet_preprocess(rgb_img))
                                clip_images.append(rgb_img)
                                is_valid = True
                        except:
                            pass
                    
                    if not is_valid:
                        # Append placeholders for missing images
                        resnet_tensors.append(torch.zeros(3, 224, 224))
                        # CLIP processor requires at least one real PIL image structure, 
                        # so we create a blank dummy image
                        clip_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                    
                    valid_mask.append(is_valid)

                # --- Run ResNet50 ---
                resnet_batch = torch.stack(resnet_tensors).to(DEVICE)
                with torch.no_grad():
                    r_feats = resnet_model(resnet_batch).squeeze(-1).squeeze(-1).cpu().numpy()
                if len(batch_ids) == 1:
                    r_feats = r_feats.reshape(1, -1)

                # --- Run CLIP ---
                clip_inputs = clip_proc(images=clip_images, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    c_feats = clip_model.get_image_features(**clip_inputs)
                    # Normalize CLIP features
                    c_feats = c_feats / c_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    c_feats = c_feats.cpu().numpy()

                # --- Store valid features, leaving missing as zeros ---
                for j, is_valid in enumerate(valid_mask):
                    if is_valid:
                        resnet_embeddings[i + j] = r_feats[j]
                        clip_embeddings[i + j] = c_feats[j]

            # Save arrays
            np.save(os.path.join(resnet_out_dir, f"{split}_{label}_image.npy"), resnet_embeddings.astype(np.float16))
            np.save(os.path.join(clip_out_dir, f"{split}_{label}_image.npy"), clip_embeddings.astype(np.float16))

if __name__ == "__main__":
    generate_image_embeddings()
