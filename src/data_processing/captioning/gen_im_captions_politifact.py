import os
import sys
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Add models directory to path so we can import qwen_module_8B
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(current_dir, "../models"))
sys.path.append(models_dir)

from qwen_module_8B import QwenCaptioner

def run_vllm_captioning():
    # Setup Paths
    base_dir = "/scratch/sg/Amrutha_MTP2/data/FakeNewsNet"
    splits_dir = os.path.join(base_dir, "splits")
    captions_dir = os.path.join(base_dir, "captions")
    
    os.makedirs(captions_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    labels = ["fake", "real"]
    BATCH_SIZE = 32

    print("=" * 60)
    print("LOADING QWEN 8B MODEL VIA VLLM (ONLY ONCE)")
    print("=" * 60)
    
    captioner = QwenCaptioner()
    print("Model ready. Running jobs...")

    for split in splits:
        for label in labels:
            print("\n" + "=" * 60)
            print(f"Running Captioning: {split} -> {label}")
            print("=" * 60)

            csv_path = os.path.join(splits_dir, split, label, f"{label}_data.csv")
            output_csv = os.path.join(captions_dir, f"{split}_{label}_captions.csv")

            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}. Skipping.")
                continue

            # Load Data
            df = pd.read_csv(csv_path)
            
            # Resume Logic
            processed_ids = set()
            if os.path.exists(output_csv):
                try:
                    existing_df = pd.read_csv(output_csv)
                    processed_ids = set(existing_df["id"].astype(str).tolist())
                    print(f"Resuming... Found {len(processed_ids)} already processed items.")
                except Exception as e:
                    print(f"Could not read existing CSV. Starting fresh.")

            batch_img_paths = []
            batch_metadata = []
            write_header = not os.path.exists(output_csv)

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
                article_id = str(row["id"])
                title = str(row["title"])
                image_path = str(row.get("image_path", ""))
                image_exists = str(row.get("Image_exist", "no")).strip().lower() == "yes"

                # Skip if already processed
                if article_id in processed_ids:
                    continue

                # We only caption if the image exists and the path is valid
                if not image_exists or not os.path.exists(image_path):
                    results = [{
                        "id": article_id,
                        "title": title,
                        "caption": "Image content unavailable"
                    }]
                    pd.DataFrame(results).to_csv(output_csv, mode="a", header=write_header, index=False)
                    write_header = False
                    continue

                # Verify image is not corrupted
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception:
                    print(f"Corrupt file skipped: {image_path}")
                    results = [{"id": article_id, "title": title, "caption": "Image corrupted"}]
                    pd.DataFrame(results).to_csv(output_csv, mode="a", header=write_header, index=False)
                    write_header = False
                    continue

                batch_img_paths.append(image_path)
                batch_metadata.append({"id": article_id, "title": title})

                # Run Batch
                if len(batch_img_paths) >= BATCH_SIZE:
                    try:
                        # Call vLLM generation
                        captions = captioner.generate_batch_captions(batch_img_paths)
                        
                        results = []
                        for meta, cap in zip(batch_metadata, captions):
                            results.append({
                                "id": meta["id"],
                                "title": meta["title"],
                                "caption": cap,
                            })

                        pd.DataFrame(results).to_csv(output_csv, mode="a", header=write_header, index=False)
                        write_header = False

                    except Exception as e:
                        print(f"Batch failed: {e}")

                    batch_img_paths = []
                    batch_metadata = []

            # Final Batch
            if batch_img_paths:
                print("Processing final batch...")
                try:
                    captions = captioner.generate_batch_captions(batch_img_paths)
                    
                    results = []
                    for meta, cap in zip(batch_metadata, captions):
                        results.append({
                            "id": meta["id"],
                            "title": meta["title"],
                            "caption": cap,
                        })

                    pd.DataFrame(results).to_csv(output_csv, mode="a", header=write_header, index=False)

                except Exception as e:
                    print(f"Final batch failed: {e}")

            print(f"Completed! Saved to: {output_csv}")

if __name__ == "__main__":
    run_vllm_captioning()
