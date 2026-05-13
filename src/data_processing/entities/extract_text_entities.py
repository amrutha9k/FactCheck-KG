"""
This script extracts named entities from article titles using the spaCy NLP model.
It aligns the extracted entities with the master IDs found in the embeddings 
directory and saves the results as pickle files for downstream tasks.
"""

import os
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
splits_dir = os.path.join(project_root, "data", "splits")
id_dir = os.path.join(project_root, "data", "embeddings")
output_dir = os.path.join(project_root, "data", "entities")

os.makedirs(output_dir, exist_ok=True)

def run_text_ner():
    print("Loading spaCy NLP model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Required spaCy model 'en_core_web_sm' not found. Please install it first.")
        return

    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']

    for split in splits:
        for label in labels:
            id_path = os.path.join(id_dir, f"{split}_{label}_ids.npy")
            csv_path = os.path.join(splits_dir, split, f"{label}.csv")
            output_path = os.path.join(output_dir, f"{split}_{label}_text_entities.pkl")

            if not os.path.exists(id_path) or not os.path.exists(csv_path):
                continue

            print(f"Extracting text entities for {split} {label}...")

            # Load master IDs and split data
            master_ids = np.load(id_path, allow_pickle=True)
            df = pd.read_csv(csv_path)
            
            # Map IDs to titles for quick alignment
            title_lookup = dict(zip(df['id'].astype(str), df['title']))

            # Align texts to the master ID order
            aligned_titles = []
            for mid in master_ids:
                title = title_lookup.get(str(mid), "")
                if pd.isna(title) or not str(title).strip():
                    aligned_titles.append("")
                else:
                    aligned_titles.append(str(title))

            # Batch process entities using nlp.pipe for efficiency
            extracted_entities = []
            for doc in tqdm(nlp.pipe(aligned_titles, batch_size=256), total=len(aligned_titles), desc="NER processing"):
                if not doc:
                    extracted_entities.append([])
                else:
                    # Capture unique, non-empty entities
                    entities = list(set([ent.text.strip() for ent in doc.ents if ent.text.strip()]))
                    extracted_entities.append(entities)

            pd.to_pickle(extracted_entities, output_path)

if __name__ == "__main__":
    run_text_ner()
