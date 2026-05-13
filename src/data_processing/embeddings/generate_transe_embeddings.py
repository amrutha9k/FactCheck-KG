"""
This script maps Wikidata identifiers to their pre-trained TransE knowledge 
graph embeddings. It produces aligned embedding lists for each data split 
and a global QID-to-embedding lookup dictionary.
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

# Wikidata source files (assumed to be in data/wikidata within the repo)
wikidata_dir = os.path.join(data_dir, "wikidata")
entity_mapping_path = os.path.join(wikidata_dir, "entity2id.txt")
entity_vector_path = os.path.join(wikidata_dir, "entity2vec.bin")

os.makedirs(out_dir, exist_ok=True)

def run_transe_mapping():
    # Load the mapping of QID strings to integer indices
    print("Loading entity-to-ID mapping...")
    entity_to_id = {}
    if not os.path.exists(entity_mapping_path):
        print(f"Error: Mapping file not found at {entity_mapping_path}")
        return

    with open(entity_mapping_path, "r") as f:
        next(f) # Skip the header count line
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                entity_to_id[parts[0]] = int(parts[1])

    # Load the binary TransE embeddings
    print("Loading pre-trained TransE vectors...")
    if not os.path.exists(entity_vector_path):
        print(f"Error: Vector file not found at {entity_vector_path}")
        return

    all_vectors = np.fromfile(entity_vector_path, dtype=np.float32).reshape(-1, 100)

    # Collect all QIDs used across all dataset splits
    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']
    unique_dataset_qids = set()

    for split in splits:
        for label in labels:
            qid_file = os.path.join(qid_dir, f"{split}_{label}_final_qids.pkl")
            if os.path.exists(qid_file):
                split_qids = pd.read_pickle(qid_file)
                for row in split_qids:
                    unique_dataset_qids.update(row)

    print(f"Mapping embeddings for {len(unique_dataset_qids)} unique entities...")
    
    # Create the global lookup dictionary
    zero_vector = np.zeros(100, dtype=np.float32)
    qid_to_embedding = {}

    for qid in unique_dataset_qids:
        if qid in entity_to_id:
            idx = entity_to_id[qid]
            qid_to_embedding[qid] = all_vectors[idx].astype(np.float32)
        else:
            # Assign zero vector if QID is missing from the knowledge graph
            qid_to_embedding[qid] = zero_vector

    # Save the global lookup dictionary
    global_lookup_path = os.path.join(out_dir, "qid_to_transe_embedding.pkl")
    pd.to_pickle(qid_to_embedding, global_lookup_path)

    # Generate and save aligned embedding lists for each split
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
                    # Retrieve the pre-mapped vector for each QID in the row
                    vectors = [qid_to_embedding[q] for q in row_qids]
                    split_embeddings.append(vectors)

            save_path = os.path.join(out_dir, f"{split}_{label}_qid_transe_embeddings.pkl")
            pd.to_pickle(split_embeddings, save_path)
            print(f"Saved TransE embeddings for {split} {label}")

if __name__ == "__main__":
    run_transe_mapping()
