"""
This script extracts unique identifiers from the training, validation, and test 
CSV splits and saves them as numpy files in the embeddings directory.
"""

import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
splits_dir = os.path.join(project_root, "data", "splits")
output_dir = os.path.join(project_root, "data", "embeddings")

def create_master_ids():
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        for label in ["fake", "real"]:
            csv_path = os.path.join(splits_dir, split, f"{label}.csv")
            
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            
            if "id" not in df.columns:
                continue
                
            ids = df["id"].astype(str).tolist()
            save_path = os.path.join(output_dir, f"{split}_{label}_ids.npy")
            np.save(save_path, np.array(ids))

if __name__ == "__main__":
    create_master_ids()
