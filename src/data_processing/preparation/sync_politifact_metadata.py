"""
This script synchronizes the original PolitiFact datasets with the metadata 
from the image crawler. It ensures IDs are aligned and that image paths 
are recorded using the new repository's relative structure.
"""

import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_root = os.path.join(project_root, "data", "politifact")

def synchronize_metadata():
    for label in ["fake", "real"]:
        label_dir = os.path.join(data_root, label)
        original_csv = os.path.join(label_dir, f"politifact_{label}.csv")
        mapping_csv = os.path.join(label_dir, "mapping.csv")

        if not os.path.exists(original_csv) or not os.path.exists(mapping_csv):
            continue

        df_original = pd.read_csv(original_csv)
        df_mapping = pd.read_csv(mapping_csv)

        # Map original politifact IDs into the crawler mapping file
        title_to_id = dict(zip(df_original['title'], df_original['id']))
        df_mapping['politifact_id'] = df_mapping['text'].map(title_to_id)

        # Reorder columns for better organization
        cols = list(df_mapping.columns)
        if 'politifact_id' in cols:
            cols.remove('politifact_id')
            cols.insert(1, 'politifact_id')
        df_mapping = df_mapping[cols]
        df_mapping.to_csv(mapping_csv, index=False)

        # Update the original dataset with image information
        title_to_img_id = dict(zip(df_mapping['text'], df_mapping['id']))
        
        # Helper to convert crawler paths to relative repo paths
        def normalize_path(title):
            img_id = title_to_img_id.get(title)
            if img_id is None:
                return "None"
            
            # Find the file in the images folder to determine the extension
            img_dir = os.path.join(label_dir, "images")
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                if os.path.exists(os.path.join(img_dir, f"{int(img_id)}{ext}")):
                    return os.path.join("data", "politifact", label, "images", f"{int(img_id)}{ext}")
            return "None"

        df_original['image_id'] = df_original['title'].map(title_to_img_id)
        df_original['image_path'] = df_original['title'].apply(normalize_path)

        # Set existence flag and finalize data types
        df_original['Image_exist'] = df_original['image_path'].apply(lambda x: 'yes' if x != "None" else 'no')
        df_original['image_id'] = df_original['image_id'].fillna(-1).astype(int)

        df_original.to_csv(original_csv, index=False)

if __name__ == "__main__":
    synchronize_metadata()
