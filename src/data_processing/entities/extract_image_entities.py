"""
This script extracts named entities from images using the Qwen 8B model via vLLM.
It processes images from the defined data splits, identifies people, organizations, 
and locations, and saves the resulting entity lists as pickle files.
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(os.path.join(project_root, "src/models"))

try:
    from qwen_module_8B import QwenCaptioner
except ImportError:
    print("Error: Qwen module not found in src/models.")
    sys.exit(1)

class QwenParser:
    """Parses raw model output to extract a clean list of identified entities."""
    
    @staticmethod
    def parse_full(raw_text):
        if not isinstance(raw_text, str) or pd.isna(raw_text):
            return []

        # Regex to capture content between specific markers in the model output
        pattern = r"ENTITIES_IDENTIFIED:(.*?)(?=VISUAL_DESCRIPTION:|TEXT_TRANSCRIPTION:|SCENE_CONTEXT:|$)"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        raw_ents = match.group(1).strip() if match else ""

        clean_entities = []
        ignore_keywords = ["NO_ENTITIES_FOUND", "NONE", "NO TEXT", "NOT VISIBLE"]

        if raw_ents:
            for line in raw_ents.split('\n'):
                line = line.strip()
                if not line or any(k in line.upper() for k in ignore_keywords):
                    continue

                # Clean bullet points and parenthetical notes
                name = re.sub(r'\s*\([^)]*\)$', '', line).strip()
                name = re.sub(r'^[-*]\s*', '', name)

                if len(name) > 1:
                    clean_entities.append(name)

        return clean_entities

def main():
    splits_dir = os.path.join(project_root, "data", "splits")
    emb_dir = os.path.join(project_root, "data", "embeddings")
    ent_dir = os.path.join(project_root, "data", "entities")
    os.makedirs(ent_dir, exist_ok=True)

    print("Loading Qwen 8B via vLLM...")
    captioner = QwenCaptioner()

    batch_size = 16
    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']

    for split in splits:
        for label in labels:
            id_file = os.path.join(emb_dir, f"{split}_{label}_ids.npy")
            csv_file = os.path.join(splits_dir, split, f"{label}.csv")

            if not os.path.exists(id_file) or not os.path.exists(csv_file):
                continue

            print(f"Processing image entities for {split} {label}...")
            master_ids = np.load(id_file, allow_pickle=True).astype(str).tolist()
            df = pd.read_csv(csv_file)
            df['id'] = df['id'].astype(str)

            path_lookup = dict(zip(df['id'], df['image_path']))
            exist_lookup = dict(zip(df['id'], df['Image_exist']))

            entities_to_save = []
            batch_paths = []
            batch_indices = []

            for aid in tqdm(master_ids, desc="Extracting entities"):
                rel_path = str(path_lookup.get(aid, ""))
                abs_path = os.path.join(project_root, rel_path)
                exists = str(exist_lookup.get(aid, "no")).lower() == "yes"

                if not exists or not os.path.exists(abs_path):
                    entities_to_save.append([])
                    continue

                batch_paths.append(abs_path)
                batch_indices.append(len(entities_to_save))
                entities_to_save.append(None)

                if len(batch_paths) >= batch_size:
                    try:
                        results = captioner.generate_batch_captions(batch_paths)
                        for i, raw_output in enumerate(results):
                            entities_to_save[batch_indices[i]] = QwenParser.parse_full(raw_output)
                    except Exception as e:
                        print(f"Batch processing error: {e}")
                        for i in batch_indices:
                            entities_to_save[i] = []
                    
                    batch_paths = []
                    batch_indices = []

            # Process remaining items
            if batch_paths:
                try:
                    results = captioner.generate_batch_captions(batch_paths)
                    for i, raw_output in enumerate(results):
                        entities_to_save[batch_indices[i]] = QwenParser.parse_full(raw_output)
                except Exception as e:
                    print(f"Final batch error: {e}")
                    for i in batch_indices:
                        entities_to_save[i] = []

            # Cleanup any remaining placeholders
            entities_to_save = [e if e is not None else [] for e in entities_to_save]

            out_path = os.path.join(ent_dir, f"{split}_{label}_image_entities.pkl")
            pd.to_pickle(entities_to_save, out_path)

if __name__ == "__main__":
    main()
