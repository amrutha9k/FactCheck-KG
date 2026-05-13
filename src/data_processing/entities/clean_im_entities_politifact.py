import os
import pandas as pd
import re

def clean_raw_entity(raw_ent):
    """
    Cleans tags like 'PERSON:', '(ORG)', and splits by comma.
    """
    # Split by comma if Qwen merged them: "Bill Gates (PERSON), Donald Trump"
    parts = str(raw_ent).split(',')
    cleaned_parts = []
    
    for part in parts:
        part = part.strip()
        
        # Remove Prefixes ("PERSON: Villanueva" -> "Villanueva")
        part = re.sub(r"^(PERSON|ORG|LOCATION|GPE|LOC|FLAG|SYMBOL)\s*:\s*", "", part, flags=re.IGNORECASE)
        
        # Remove Parentheses tags ("Bill Gates (PERSON)" -> "Bill Gates")
        part = re.sub(r"\(\s*(PERSON|ORG|LOCATION|GPE|LOC|FLAG|SYMBOL)\s*\)", "", part, flags=re.IGNORECASE)
        
        # Remove trailing tags ("Donald Trump PERSON" -> "Donald Trump")
        part = re.sub(r"\b(PERSON|ORG|LOCATION|GPE|LOC|FLAG|SYMBOL)$", "", part, flags=re.IGNORECASE)
        
        # Clean quotes/punctuation
        part = re.sub(r"^['\"]|['\"]$", "", part)
        part = part.strip()
        
        if len(part) > 1:
            cleaned_parts.append(part)
            
    return cleaned_parts

def process_files():
    # Only target the entities folder
    ent_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet/entities"
    
    splits = ["train", "val", "test"]
    labels = ["fake", "real"]

    print("Starting Image Entity Cleanup...")
    print("-" * 60)

    for split in splits:
        for label in labels:
            # ONLY target the image entities
            file_name = f"{split}_{label}_image_entities.pkl"
            file_path = os.path.join(ent_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"Skipping missing file: {file_name}")
                continue
                
            try:
                dirty_entities = pd.read_pickle(file_path)
            except:
                continue

            clean_entities_master = []
            changes = 0
            
            for ent_list in dirty_entities:
                new_row_ents = []
                for ent in ent_list:
                    cleaned = clean_raw_entity(ent)
                    new_row_ents.extend(cleaned)
                
                # Remove duplicates in the row
                unique_row_ents = list(dict.fromkeys(new_row_ents))
                clean_entities_master.append(unique_row_ents)
                
                if ent_list != unique_row_ents:
                    changes += 1

            # OVERWRITE the exact same file so mGENRE can find it easily
            pd.to_pickle(clean_entities_master, file_path)
            
            print(f"Cleaned and overwritten: {file_name} ({changes} rows modified)")

    print("-" * 60)
    print("Done! Image entities are now clean and ready for mGENRE.")

if __name__ == "__main__":
    process_files()
