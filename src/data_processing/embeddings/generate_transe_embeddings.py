 """
This script maps Wikidata identifiers (QIDs) to their pre-trained TransE 
knowledge graph embeddings. It follows the nested directory structure 
required for the Wikidata source files and generates aligned outputs for 
the PolitiFact dataset splits.
"""

import os
import numpy as np
import pandas as pd

# Define path constants relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_dir = os.path.join(project_root, "data")
qid_dir = os.path.join(data_dir, "qids")
out_dir = os.path.join(data_dir, "embeddings", "transe")

# Wikidata source files aligned with the specific nested structure
# path: data/wikidata/knowledge graphs/entity2id.txt
# path: data/wikidata/embeddings/dimension_100/transe/entity2vec.bin
wikidata_root = os.path.join(data_dir, "wikidata")
entity_mapping_path = os.path.join(wikidata_root, "knowledge graphs", "entity2id.txt")
entity_vector_path = os.path.join(wikidata_root, "embeddings", "dimension_100", "transe", "entity2vec.bin")

os.makedirs(out_dir, exist_ok=True)

def run_transe_mapping():
    print("Loading entity-to-ID mapping from 'knowledge graphs'...")
    entity_to_id = {}
    
    if not os.path.exists(entity_mapping_path):
        print(f"Error: Could not find {entity_mapping_path}")
        return

    with open(entity_mapping_path, "r") as f:
        # Skip the first line containing the total count
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                entity_to_id[parts[0]] = int(parts[1])

    print("Loading pre-trained TransE vectors from 'dimension_100/transe'...")
    if not os.path.exists(entity_vector_path):
        print(f"Error: Could not find {entity_vector_path}")
        return

    # Load 100-dimensional binary embedding vectors
    all_vectors = np.fromfile(entity_vector_path, dtype=np.float32).reshape(-1, 100)

    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']
    unique_dataset_qids = set()

    # Collect all QIDs across every data split
    for split in splits:
        for label in labels:
            qid_file = os.path.join(qid_dir, f"{split}_{label}_final_qids.pkl")
            if os.path.exists(qid_file):
                split_data = pd.read_pickle(qid_file)
                for row in split_data:
                    unique_dataset_qids.update(row)

    print(f"Aligning embeddings for {len(unique_dataset_qids)} identified entities...")
    
    zero_vector = np.zeros(100, dtype=np.float32)
    qid_lookup_table = {}

    for qid in unique_dataset_qids:
        if qid in entity_to_id:
            idx = entity_to_id[qid]
            qid_lookup_table[qid] = all_vectors[idx].astype(np.float32)
        else:
            # Provide zero-vector if entity is missing from the knowledge graph
            qid_lookup_table[qid] = zero_vector

    # Save the consolidated QID-to-vector lookup dictionary
    pd.to_pickle(qid_lookup_table, os.path.join(out_dir, "qid_to_transe_embedding.pkl"))

    # Generate split-specific aligned embedding files
    for split in splits:
        for label in labels:
            qid_file = os.path.join(qid_dir, f"{split}_{label}_final_qids.pkl")
            if not os.path.exists(qid_file):
                continue

            qids_list = pd.read_pickle(qid_file)
            split_embeddings = []

            for row_qids in qids_list:
                if not row_qids:
                    split_embeddings.append([])
                else:
                    # Append the specific 100D vector for each QID in the article
                    vectors = [qid_lookup_table[q] for q in row_qids]
                    split_embeddings.append(vectors)

            save_path = os.path.join(out_dir, f"{split}_{label}_qid_transe_embeddings.pkl")
            pd.to_pickle(split_embeddings, save_path)
            print(f"Successfully saved TransE embeddings for: {split} {label}")

if __name__ == "__main__":
    run_transe_mapping()
