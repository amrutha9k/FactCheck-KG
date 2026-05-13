"""
This script synchronizes the original PolitiFact CSV files with the metadata 
generated during image downloading. It ensures that both the original data 
and the mapping records share consistent IDs and image path information.
"""

import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_root = os.path.join(project_root, "data", "politifact")

def synchronize_metadata():
    labels = ["fake", "real"]

    for label in labels:
        label_dir = os.path.join(data_root, label)
        original_csv = os.path.join(label_dir, f"politifact_{label}.csv")
        mapping_csv = os.path.join(label_dir, "mapping.csv")

        if not os.path.exists(original_csv) or not os.path.exists(mapping_csv):
            continue

        df_original = pd.read_csv(original_csv)
        df_mapping = pd.read_csv(mapping_csv)

        # Update mapping file with IDs from the original dataset
        title_to_id_map = dict(zip(df_original['title'], df_original['id']))
        df_mapping['politifact_id'] = df_mapping['text'].map(title_to_id_map)

        # Ensure politifact_id is positioned early in the mapping file
        mapping_columns = list(df_mapping.columns)
        if 'politifact_id' in mapping_columns:
            mapping_columns.remove('politifact_id')
            mapping_columns.insert(1, 'politifact_id')
        df_mapping = df_mapping[mapping_columns]

        df_mapping.to_csv(mapping_csv, index=False)

        # Update original dataset with image information from mapping file
        title_to_img_id = dict(zip(df_mapping['text'], df_mapping['id']))
        title_to_img_path = dict(zip(df_mapping['text'], df_mapping['image_path']))

        df_original['image_id'] = df_original['title'].map(title_to_img_id)
        df_original['image_path'] = df_original['title'].map(title_to_img_path)

        # Set existence flag and clean up missing values
        df_original['Image_exist'] = df_original['image_id'].notna().map({True: 'yes', False: 'no'})
        df_original['image_id'] = df_original['image_id'].fillna(-1).astype(int)
        df_original['image_path'] = df_original['image_path'].fillna("None")

        df_original.to_csv(original_csv, index=False)

if __name__ == "__main__":
    synchronize_metadata()
