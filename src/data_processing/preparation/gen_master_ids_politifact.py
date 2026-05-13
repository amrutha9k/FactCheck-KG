import os
import pandas as pd
import numpy as np

def generate_master_ids():
    base_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/splits"
    output_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/embeddings"
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    labels = ["fake", "real"]
    
    print("Generating master ID files...")
    
    for split in splits:
        for label in labels:
            csv_path = os.path.join(base_dir, split, label, f"{label}_data.csv")
            
            if not os.path.exists(csv_path):
                print(f"Skipping {split}/{label} - CSV not found")
                continue
                
            df = pd.read_csv(csv_path)
            
            # Using the primary 'id' column
            if "id" not in df.columns:
                print(f"Error: 'id' column missing in {csv_path}")
                continue
                
            ids = df["id"].astype(str).tolist()
            
            # Save to npy
            save_path = os.path.join(output_dir, f"{split}_{label}_ids.npy")
            np.save(save_path, np.array(ids))
            
            print(f"Saved {len(ids)} IDs to {save_path}")

if __name__ == "__main__":
    generate_master_ids()
