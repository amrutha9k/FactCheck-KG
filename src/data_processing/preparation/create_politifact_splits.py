"""
This script partitions the PolitiFact dataset into training, validation, and testing sets.
Instead of copying images, it generates CSV files in the splits directory that contain
relative paths to the images stored in the central data/politifact/ directory.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_dir = os.path.join(project_root, "data")
splits_root = os.path.join(data_dir, "splits")

def save_split_csv(df, split_name, label):
    """
    Saves the split dataframe as a CSV and ensures image paths point to the central store.
    """
    split_dir = os.path.join(splits_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    updated_records = []
    
    for _, row in df.iterrows():
        new_row = row.copy()
        
        # Verify if image exists in the central politifact folder
        image_name = f"{row['image_id']}.jpg" # or derived from mapping
        if 'image_path' in row and pd.notna(row['image_path']) and row['image_path'] != "None":
            # Extract filename from existing path
            image_name = os.path.basename(row['image_path'])
        
        # Construct the central path: data/politifact/[label]/images/[filename]
        central_path = os.path.join("data", "politifact", label, "images", image_name)
        absolute_central_path = os.path.join(project_root, central_path)
        
        if os.path.exists(absolute_central_path):
            new_row['image_path'] = central_path
            new_row['Image_exist'] = 'yes'
        else:
            new_row['image_path'] = ""
            new_row['Image_exist'] = 'no'
            
        updated_records.append(new_row)
        
    final_df = pd.DataFrame(updated_records)
    output_path = os.path.join(split_dir, f"{label}.csv")
    final_df.to_csv(output_path, index=False)

def main():
    for label in ["fake", "real"]:
        csv_input_path = os.path.join(data_dir, "politifact", label, f"politifact_{label}.csv")
        
        if not os.path.exists(csv_input_path):
            continue
            
        df = pd.read_csv(csv_input_path)
        
        train_df, remainder_df = train_test_split(df, test_size=0.20, random_state=42)
        val_df, test_df = train_test_split(remainder_df, test_size=0.50, random_state=42)
        
        save_split_csv(train_df, "train", label)
        save_split_csv(val_df, "val", label)
        save_split_csv(test_df, "test", label)

if __name__ == "__main__":
    main()
