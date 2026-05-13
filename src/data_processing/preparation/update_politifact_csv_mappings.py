import pandas as pd
import os

def update_datasets():
    # Base directory based on your folder structure
    base_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet"
    
    # We are only focusing on politifact datasets per your request
    datasets = [
        ("politifact_fake", "politifact_fake.csv"),
        ("politifact_real", "politifact_real.csv")
    ]

    for folder_name, orig_filename in datasets:
        orig_path = os.path.join(base_dir, folder_name, orig_filename)
        mapping_path = os.path.join(base_dir, folder_name, "mapping.csv")

        # Check if files exist before processing
        if not os.path.exists(orig_path):
            print(f"Error: Original file not found at {orig_path}")
            continue
        if not os.path.exists(mapping_path):
            print(f"Error: Mapping file not found at {mapping_path}")
            continue

        print(f"Processing dataset: {folder_name}...")

        # Load both CSVs
        df_orig = pd.read_csv(orig_path)
        df_map = pd.read_csv(mapping_path)

        # ---------------------------------------------------------
        # STEP 1: Update mapping.csv with politifact_id
        # ---------------------------------------------------------
        
        # Create a dictionary mapping the original title to the original politifact id
        title_to_orig_id = dict(zip(df_orig['title'], df_orig['id']))
        
        # Map this to the 'text' column in mapping.csv (which contains the titles)
        df_map['politifact_id'] = df_map['text'].map(title_to_orig_id)

        # Reorder columns so 'politifact_id' is right next to 'id' for better readability
        cols = list(df_map.columns)
        if 'politifact_id' in cols:
            cols.remove('politifact_id')
            cols.insert(1, 'politifact_id') 
        df_map = df_map[cols]

        # Save the updated mapping.csv
        df_map.to_csv(mapping_path, index=False)
        print(f"  -> Updated {mapping_path}")

        # ---------------------------------------------------------
        # STEP 2: Update original CSVs with image metadata
        # ---------------------------------------------------------
        
        # Create dictionaries from mapping.csv linking text(title) to image details
        title_to_image_id = dict(zip(df_map['text'], df_map['id']))
        title_to_image_path = dict(zip(df_map['text'], df_map['image_path']))

        # Add the new columns to the original dataframe using the title
        df_orig['image_id'] = df_orig['title'].map(title_to_image_id)
        df_orig['image_path'] = df_orig['title'].map(title_to_image_path)

        # Create Image_exist column: 'yes' if image_id is not NaN, else 'no'
        df_orig['Image_exist'] = df_orig['image_id'].notna().map({True: 'yes', False: 'no'})

        # Fills missing IDs with -1 and paths with "None"
        df_orig['image_id'] = df_orig['image_id'].fillna(-1).astype(int)
        df_orig['image_path'] = df_orig['image_path'].fillna("None")
        # Save the updated original CSV
        df_orig.to_csv(orig_path, index=False)
        print(f"  -> Updated {orig_path}")
        print("-" * 60)

if __name__ == "__main__":
    print("Starting mapping update process...")
    print("-" * 60)
    update_datasets()
    print("Update complete.")
