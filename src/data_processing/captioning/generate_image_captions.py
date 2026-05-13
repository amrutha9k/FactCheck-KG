"""
This script generates descriptive captions for images in the PolitiFact dataset.
It utilizes the Qwen 8B model via vLLM for high-throughput batch inference.
The resulting captions are saved in the data/captions directory, categorized
by dataset split and label.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Setup paths to import the model module from the models directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(os.path.join(project_root, "src", "models"))

try:
    from qwen_module_8B import QwenCaptioner
except ImportError:
    print("Error: Required model module not found in src/models.")
    sys.exit(1)

def run_captioning():
    # Define directory structure
    splits_dir = os.path.join(project_root, "data", "splits")
    captions_dir = os.path.join(project_root, "data", "captions")
    os.makedirs(captions_dir, exist_ok=True)

    batch_size = 32
    splits = ["train", "val", "test"]
    labels = ["fake", "real"]

    print("Initializing Qwen 8B model via vLLM...")
    captioner = QwenCaptioner()

    for split in splits:
        for label in labels:
            csv_path = os.path.join(splits_dir, split, f"{label}.csv")
            output_csv = os.path.join(captions_dir, f"{split}_{label}_captions.csv")

            if not os.path.exists(csv_path):
                continue

            print(f"Processing captions for {split} {label} dataset...")
            df = pd.read_csv(csv_path)
            
            # Implementation of resume logic
            processed_ids = set()
            if os.path.exists(output_csv):
                try:
                    existing_data = pd.read_csv(output_csv)
                    processed_ids = set(existing_data["id"].astype(str).tolist())
                    print(f"Resuming process: {len(processed_ids)} items already completed.")
                except Exception:
                    print("Existing output file found but could not be read. Starting fresh.")

            batch_paths = []
            batch_meta = []
            is_new_file = not os.path.exists(output_csv)

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}/{label}"):
                article_id = str(row["id"])
                title = str(row.get("title", ""))
                
                if article_id in processed_ids:
                    continue

                # Construct absolute path for image verification
                rel_img_path = str(row.get("image_path", ""))
                abs_img_path = os.path.join(project_root, rel_img_path)
                image_exists = str(row.get("Image_exist", "no")).strip().lower() == "yes"

                if not image_exists or not os.path.exists(abs_img_path):
                    results = [{"id": article_id, "title": title, "caption": "Image unavailable"}]
                    pd.DataFrame(results).to_csv(output_csv, mode="a", header=is_new_file, index=False)
                    is_new_file = False
                    continue

                try:
                    with Image.open(abs_img_path) as img:
                        img.verify()
                except Exception:
                    results = [{"id": article_id, "title": title, "caption": "Corrupt image"}]
                    pd.DataFrame(results).to_csv(output_csv, mode="a", header=is_new_file, index=False)
                    is_new_file = False
                    continue

                batch_paths.append(abs_img_path)
                batch_meta.append({"id": article_id, "title": title})

                if len(batch_paths) >= batch_size:
                    try:
                        captions = captioner.generate_batch_captions(batch_paths)
                        results = []
                        for meta, cap in zip(batch_meta, captions):
                            results.append({
                                "id": meta["id"],
                                "title": meta["title"],
                                "caption": cap,
                            })
                        pd.DataFrame(results).to_csv(output_csv, mode="a", header=is_new_file, index=False)
                        is_new_file = False
                    except Exception as e:
                        print(f"Batch generation failed: {e}")
                    
                    batch_paths = []
                    batch_meta = []

            # Handle remaining items in the final batch
            if batch_paths:
                try:
                    captions = captioner.generate_batch_captions(batch_paths)
                    results = [{"id": m["id"], "title": m["title"], "caption": c} for m, c in zip(batch_meta, captions)]
                    pd.DataFrame(results).to_csv(output_csv, mode="a", header=is_new_file, index=False)
                except Exception as e:
                    print(f"Final batch generation failed: {e}")

if __name__ == "__main__":
    run_captioning()
