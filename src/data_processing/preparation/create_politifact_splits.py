"""
This script partitions the PolitiFact dataset into training, validation, and testing sets.
It migrates images into the split directory structure and updates CSV records to 
maintain correct data-to-image mappings.
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_dir = os.path.join(project_root, "data")
output_dir = os.path.join(data_dir, "splits")

def process_and_copy(df, split_name, label, base_out_dir):
    """
    Creates directory structures and migrates data for a specific split.
    """
    split_folder = os.path.join(base_out_dir, split_name, label)
    image_output_folder = os.path.join(split_folder, "images")
    os.makedirs(image_output_folder, exist_ok=True)
    
    updated_records = []
    
    for _, row in df.iterrows():
        new_row = row.copy()
        raw_path = row.get('image_path')
        
        if str(row.get('Image_exist')).lower() == 'yes' and pd.notna(raw_path):
            source_path = str(raw_path)
            
            if os.path.exists(source_path):
                filename = os.path.basename(source_path)
                destination_path = os.path.join(image_output_folder, filename)
                
                shutil.copy2(source_path, destination_path)
                new_row['image_path'] = os.path.relpath(destination_path, project_root)
            else:
                new_row['Image_exist'] = 'no'
                new_row['image_path'] = ""
        else:
            new_row['image_path'] = ""
            
        updated_records.append(new_row)
        
    final_df = pd.DataFrame(updated_records)
    csv_save_path = os.path.join(split_folder, f"{label}_data.csv")
    final_df.to_csv(csv_save_path, index=False)

def main():
    for label in ["fake", "real"]:
        csv_input_path = os.path.join(data_dir, "politifact", label, f"politifact_{label}.csv")
        
        if not os.path.exists(csv_input_path):
            continue
            
        df = pd.read_csv(csv_input_path)
        
        train_df, remainder_df = train_test_split(df, test_size=0.20, random_state=42)
        val_df, test_df = train_test_split(remainder_df, test_size=0.50, random_state=42)
        
        process_and_copy(train_df, "train", label, output_dir)
        process_and_copy(val_df, "val", label, output_dir)
        process_and_copy(test_df, "test", label, output_dir)

if __name__ == "__main__":
    main()
