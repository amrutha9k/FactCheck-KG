"""
This script generates text embeddings for article titles using multiple transformer 
models: MuRIL, mBERT, and XLM-RoBERTa. It ensures data alignment by following 
the master ID sequences and saves the output in model-specific directories.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Project path configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
splits_dir = os.path.join(project_root, "data", "splits")
emb_root = os.path.join(project_root, "data", "embeddings")

# Define target models and execution parameters
TEXT_MODELS = {
    "muril": "google/muril-base-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base"
}

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_model_embeddings(model_key, model_path):
    print(f"Processing text embeddings using {model_key.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
    
    # Create directory for the specific model's output
    model_output_dir = os.path.join(emb_root, model_key)
    os.makedirs(model_output_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    labels = ["fake", "real"]

    for split in splits:
        for label in labels:
            id_path = os.path.join(emb_root, f"{split}_{label}_ids.npy")
            csv_path = os.path.join(splits_dir, split, f"{label}.csv")
            
            if not os.path.exists(id_path) or not os.path.exists(csv_path):
                continue
                
            master_ids = np.load(id_path, allow_pickle=True).astype(str).tolist()
            df = pd.read_csv(csv_path)
            
            # Align titles with the master ID sequence
            title_lookup = dict(zip(df["id"].astype(str), df["title"].fillna("").astype(str)))
            ordered_titles = [title_lookup.get(aid, "") for aid in master_ids]

            all_embeddings = []
            print(f"Generating features for {split} {label} split...")

            for i in tqdm(range(0, len(ordered_titles), BATCH_SIZE), desc=model_key):
                batch_texts = ordered_titles[i : i + BATCH_SIZE]
                
                # Tokenize and run forward pass
                inputs = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Extract the CLS token (first token) representation for the whole sentence
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

            if all_embeddings:
                final_matrix = np.vstack(all_embeddings).astype(np.float16)
                save_filename = f"{split}_{label}_text.npy"
                np.save(os.path.join(model_output_dir, save_filename), final_matrix)
    
    # Explicitly clear model from memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    for key, path in TEXT_MODELS.items():
        process_model_embeddings(key, path)
