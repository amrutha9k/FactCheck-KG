"""
This script aggregates all Wikidata identifiers (QIDs) identified across the 
different data splits. It identifies a unique set of entities to facilitate 
subgraph extraction and embedding generation for the knowledge graph.
"""

import os
import pickle
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
qids_dir = os.path.join(project_root, "data", "qids")

def generate_unique_qid_list():
    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']

    unique_qids = set()
    
    print("Collecting unique entities from all data splits...")

    for split in splits:
        for label in labels:
            file_name = f"{split}_{label}_final_qids.pkl"
            file_path = os.path.join(qids_dir, file_name)
            
            if not os.path.exists(file_path):
                continue

            # Load the collection of QID lists for this split
            qids_data = pd.read_pickle(file_path)

            for entity_list in qids_data:
                for qid in entity_list:
                    # Validate that the string matches the Wikidata QID format
                    if isinstance(qid, str) and qid.startswith('Q') and qid[1:].isdigit():
                        unique_qids.add(qid)

            print(f"Processed split: {split} {label}")

    # Save the consolidated unique QID list
    output_path = os.path.join(qids_dir, "unique_qids.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(list(unique_qids), f)

    print(f"Extraction complete. Total unique entities identified: {len(unique_qids)}")
    print(f"Consolidated QID list saved to: {output_path}")

if __name__ == "__main__":
    generate_unique_qid_list()
