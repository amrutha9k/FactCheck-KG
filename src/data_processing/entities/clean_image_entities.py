"""
This script post-processes identified image entities by removing category tags,
stripping unnecessary punctuation, and splitting merged strings. It overwrites 
the existing pickle files in the data/entities directory with cleaned, 
unique entity lists.
"""

import os
import re
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
ent_dir = os.path.join(project_root, "data", "entities")

def clean_raw_entity(raw_ent):
    """
    Splits entity strings and removes model-generated category labels.
    """
    parts = str(raw_ent).split(',')
    cleaned_parts = []
    
    # Regex pattern to identify common NER tags used by the model
    tag_pattern = r"(PERSON|ORG|LOCATION|GPE|LOC|FLAG|SYMBOL)"
    
    for part in parts:
        part = part.strip()
        
        # Remove prefixes like "PERSON: "
        part = re.sub(rf"^{tag_pattern}\s*:\s*", "", part, flags=re.IGNORECASE)
        
        # Remove parenthetical tags like "(ORG)"
        part = re.sub(rf"\(\s*{tag_pattern}\s*\)", "", part, flags=re.IGNORECASE)
        
        # Remove trailing tags
        part = re.sub(rf"\b{tag_pattern}$", "", part, flags=re.IGNORECASE)
        
        # Strip remaining quotes and whitespace
        part = re.sub(r"^['\"]|['\"]$", "", part).strip()
        
        if len(part) > 1:
            cleaned_parts.append(part)
            
    return cleaned_parts

def run_cleanup():
    splits = ["train", "val", "test"]
    labels = ["fake", "real"]

    print("Cleaning image entity data...")

    for split in splits:
        for label in labels:
            file_name = f"{split}_{label}_image_entities.pkl"
            file_path = os.path.join(ent_dir, file_name)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                raw_entities_data = pd.read_pickle(file_path)
            except Exception:
                continue

            cleaned_master_list = []
            
            for entity_list in raw_entities_data:
                processed_row = []
                for entity in entity_list:
                    cleaned_items = clean_raw_entity(entity)
                    processed_row.extend(cleaned_items)
                
                # Maintain order while removing duplicates within the row
                unique_row = list(dict.fromkeys(processed_row))
                cleaned_master_list.append(unique_row)

            # Overwrite the pickle file with cleaned data
            pd.to_pickle(cleaned_master_list, file_path)
            print(f"Successfully processed and updated: {file_name}")

if __name__ == "__main__":
    run_cleanup()
