import os
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

def extract_entities_from_text():
    # ==========================================
    # PATH SETUP
    # ==========================================
    base_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/splits"
    id_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/embeddings"
    output_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/entities" # To save entities
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # LOAD SPACY MODEL
    # ==========================================
    print("Loading spaCy English NLP model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[!] Model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        return

    # Using 'val' to match the directory structure we created earlier
    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']

    for split in splits:
        for label in labels:
            print(f"\n{'='*40}")
            print(f"Processing: {split}/{label}")
            print(f"{'='*40}")
            
            # --- INPUT PATHS ---
            id_file = os.path.join(id_dir, f"{split}_{label}_ids.npy")
            csv_file = os.path.join(base_dir, split, label, f"{label}_data.csv")
            
            # --- OUTPUT PATH ---
            out_file = os.path.join(output_dir, f"{split}_{label}_text_entities.pkl")

            if not os.path.exists(id_file):
                print(f"   Master ID file missing: {os.path.basename(id_file)}. Skipping.")
                continue
            if not os.path.exists(csv_file):
                print(f"   CSV data file missing: {os.path.basename(csv_file)}. Skipping.")
                continue

            # ==========================================
            # LOAD DATA
            # ==========================================
            master_ids = np.load(id_file, allow_pickle=True)
            
            try:
                df = pd.read_csv(csv_file)
                if 'id' not in df.columns or 'title' not in df.columns:
                    print(f"   Required columns ('id', 'title') missing in CSV. Skipping.")
                    continue
                
                # Map original 'id' to the article 'title'
                claim_map = dict(zip(df['id'].astype(str), df['title']))
            except Exception as e:
                print(f"   Error reading CSV: {e}")
                continue

            # ==========================================
            # 1. PREPARE ALIGNED TEXT LIST
            # ==========================================
            print(f"   Aligning {len(master_ids)} titles...")
            aligned_texts = []
            
            for mid in master_ids:
                mid = str(mid)
                text = claim_map.get(mid, "")
                
                # Handle NaNs or pure whitespace
                if pd.isna(text) or str(text).strip() == "":
                    aligned_texts.append("")
                else:
                    aligned_texts.append(str(text))

            # ==========================================
            # 2. FAST EXTRACTION USING nlp.pipe()
            # ==========================================
            print("   Extracting entities (Fast Batching)...")
            entities_to_save = []
            
            # nlp.pipe takes our list and processes it in batches of 256
            for doc in tqdm(nlp.pipe(aligned_texts, batch_size=256), total=len(aligned_texts), desc="NER"):
                if len(doc) == 0:
                    entities_to_save.append([])
                else:
                    # Extract unique entity strings, strip whitespace, ignore empty ones
                    extracted_ents = list(set([ent.text.strip() for ent in doc.ents if ent.text.strip()]))
                    entities_to_save.append(extracted_ents)

            # ==========================================
            # 3. SAVING
            # ==========================================
            pd.to_pickle(entities_to_save, out_file)
            print(f"   Saved Text Entities to: {os.path.basename(out_file)}")

if __name__ == "__main__":
    extract_entities_from_text()
    print("\nAll text entity extraction completed successfully!")
