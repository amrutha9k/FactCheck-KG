import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

TEXT_MODELS = {
    "muril": "google/muril-base-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base"
}

SPLITS = ["train", "val", "test"]
LABELS = ["fake", "real"]
BATCH_SIZE = 16
DEVICE = "cpu"

BASE_DIR = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/splits"
OUTPUT_DIR = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/embeddings"

def process_text_model(model_key, model_name):
    print(f"\nModel: {model_key.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    
    model_output_dir = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(model_output_dir, exist_ok=True)

    for split in SPLITS:
        for label in LABELS:
            id_path = os.path.join(OUTPUT_DIR, f"{split}_{label}_ids.npy")
            csv_path = os.path.join(BASE_DIR, split, label, f"{label}_data.csv")
            
            if not os.path.exists(id_path) or not os.path.exists(csv_path):
                continue
                
            master_ids = np.load(id_path, allow_pickle=True).astype(str).tolist()
            df = pd.read_csv(csv_path)
            
            # Create a lookup dictionary mapping id -> title
            text_lookup = dict(zip(df["id"].astype(str), df["title"].fillna("").astype(str)))
            ordered_texts = [text_lookup.get(aid, "") for aid in master_ids]

            all_embeddings = []
            print(f"Processing {split} {label} ({len(ordered_texts)} items)...")

            for i in tqdm(range(0, len(ordered_texts), BATCH_SIZE)):
                batch_texts = ordered_texts[i : i + BATCH_SIZE]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Extract [CLS] token representation
                batch_feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(batch_feats)

            if all_embeddings:
                full_array = np.vstack(all_embeddings).astype(np.float16)
                save_path = os.path.join(model_output_dir, f"{split}_{label}_text.npy")
                np.save(save_path, full_array)
            
    del model
    del tokenizer

if __name__ == "__main__":
    for key, name in TEXT_MODELS.items():
        process_text_model(key, name)
