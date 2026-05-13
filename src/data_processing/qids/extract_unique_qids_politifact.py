import os
import pickle
import pandas as pd
from pathlib import Path

def get_unique_qids():
    # Setup Paths
    base_dir = Path("/scratch/sg/Amrutha_MTP2/data/FakeNewsNet")
    qids_dir = base_dir / "qids"
    
    # Create a new subfolder for Knowledge Graph / Subgraph preparation
    output_dir = base_dir / "subgraph_data"
    output_dir.mkdir(exist_ok=True)

    splits = ['train', 'val', 'test']
    labels = ['fake', 'real']

    unique_qids = set()
    total_articles = 0
    total_qids_extracted = 0

    print("Scanning QID files across all splits...")
    print("-" * 60)

    for split in splits:
        for label in labels:
            file_name = f"{split}_{label}_final_qids.pkl"
            file_path = qids_dir / file_name
            
            if not file_path.exists():
                print(f"Skipping missing file: {file_name}")
                continue

            # Load the list of lists of QIDs
            qids_list = pd.read_pickle(file_path)
            total_articles += len(qids_list)

            # Flatten the list and add to our set
            for row in qids_list:
                for qid in row:
                    # Basic validation to ensure it looks like a Wikidata QID (starts with 'Q' and a number)
                    if isinstance(qid, str) and qid.startswith('Q') and qid[1:].isdigit():
                        unique_qids.add(qid)
                        total_qids_extracted += 1

            print(f"Processed: {file_name}")

    print("-" * 60)
    print("QID Extraction Summary:")
    print("-" * 60)
    print(f"Total articles processed : {total_articles}")
    print(f"Total raw QIDs found     : {total_qids_extracted} (includes duplicates across articles)")
    print(f"Total UNIQUE QIDs        : {len(unique_qids)}")
    print("-" * 60)

    # Save the unique QIDs as a list in a pickle file
    out_file = output_dir / "unique_qids.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(list(unique_qids), f)

    print(f"Saved unique QIDs to: {out_file}")

if __name__ == "__main__":
    get_unique_qids()
