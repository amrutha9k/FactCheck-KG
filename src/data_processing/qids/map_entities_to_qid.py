"""
This script links extracted entities to Wikidata identifiers (QIDs). 
It uses the mGENRE model for entity disambiguation against Wikipedia 
and subsequently retrieves unique QIDs via the Wikidata API.
"""

import os
import re
import time
import torch
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote
from genre.fairseq_model import mGENRE

# Configuration parameters for API interaction and batching
BATCH_SIZE = 32
REQUEST_DELAY = 3.0
MAX_RETRIES = 3
CHECKPOINT_EVERY = 200
TIMEOUT = 20
HEADERS = {"User-Agent": "FakeNewsResearch/1.0 (Entity Mapping)"}

# Initialize path constants relative to project structure
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_dir = os.path.join(project_root, "data")
output_root = os.path.join(project_root, "outputs")

# Setup logging
log_dir = os.path.join(output_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "qid_mapping.log"),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

QID_CACHE = {}

def clean_entity_text(text):
    """Removes common category tags from the entity string."""
    text = str(text).strip()
    return re.sub(r",?\s*(PERSON|ORG|LOCATION|GPE|LOC|FLAG|SYMBOL)$", "", text, flags=re.IGNORECASE)

def insert_markers(entity, context):
    """Wraps the target entity in specific markers within its context."""
    if not context or context == "Image unavailable":
        return f"[START] {entity} [END]"
    try:
        pattern = re.compile(rf'\b({re.escape(entity)})\b', re.IGNORECASE)
        marked, count = pattern.subn(rf'[START] \1 [END]', context, count=1)
        return marked if count > 0 else f"{context} [START] {entity} [END]"
    except Exception:
        return f"{context} [START] {entity} [END]"

def truncate_input(text, max_tokens=900):
    """Shortens input text while preserving the entity and its immediate surroundings."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    
    match = re.search(r'\[START\](.*?)\[END\]', text)
    if not match:
        return " ".join(tokens[:max_tokens])

    # Allocate token budget around the entity markers
    entity_part = match.group(0).split()
    remaining_budget = max_tokens - len(entity_part)
    left_side = text[:match.start()].split()[-int(remaining_budget * 0.6):]
    right_side = text[match.end():].split()[:int(remaining_budget * 0.4):]
    
    return " ".join(left_side + entity_part + right_side).strip()

def fetch_wikidata_qids(titles):
    """Queries the Wikidata API to convert Wikipedia titles into QIDs."""
    unique_titles = [t for t in list(set(titles)) if t not in QID_CACHE]
    if not unique_titles:
        return {t: QID_CACHE.get(t) for t in titles}

    results = {}
    chunks = [unique_titles[i:i+20] for i in range(0, len(unique_titles), 20)]
    
    for chunk in chunks:
        encoded = "|".join(quote(t, safe='') for t in chunk)
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles={encoded}&props=sitelinks&sitefilter=enwiki&format=json"
        
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                if response.status_code == 200:
                    data = response.json().get("entities", {})
                    mapping = {}
                    for qid, entry in data.items():
                        if qid.startswith("Q"):
                            title = entry.get("sitelinks", {}).get("enwiki", {}).get("title", "").replace(" ", "_")
                            if title: mapping[title] = qid
                    
                    for t in chunk:
                        qid = mapping.get(t)
                        results[t] = qid
                        if qid: QID_CACHE[t] = qid
                    break
            except Exception as e:
                logging.error(f"API Error: {e}")
                time.sleep(5)
    return results

def run_mapping_pipeline():
    # Model and data directory setup
    mgenre_path = os.path.join(data_dir, "models", "mgenre")
    splits_dir = os.path.join(data_dir, "splits")
    emb_dir = os.path.join(data_dir, "embeddings")
    ent_dir = os.path.join(data_dir, "entities")
    cap_dir = os.path.join(data_dir, "captions")
    qid_dir = os.path.join(data_dir, "qids")
    os.makedirs(qid_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading mGENRE disambiguation model on {device}...")
    
    try:
        model = mGENRE.from_pretrained(mgenre_path).eval()
        if device == "cuda":
            model = model.cuda().half()
    except Exception as e:
        print(f"Failed to load mGENRE from {mgenre_path}: {e}")
        return

    for split in ['train', 'val', 'test']:
        for label in ['fake', 'real']:
            print(f"Mapping entities for {split} {label}...")
            
            output_file = os.path.join(qid_dir, f"{split}_{label}_final_qids.pkl")
            if os.path.exists(output_file):
                continue

            # Load necessary data split components
            try:
                master_ids = np.load(os.path.join(emb_dir, f"{split}_{label}_ids.npy"), allow_pickle=True)
                text_ents = pd.read_pickle(os.path.join(ent_dir, f"{split}_{label}_text_entities.pkl"))
                img_ents = pd.read_pickle(os.path.join(ent_dir, f"{split}_{label}_image_entities.pkl"))
                df_split = pd.read_csv(os.path.join(splits_dir, split, f"{label}.csv"))
                df_cap = pd.read_csv(os.path.join(cap_dir, f"{split}_{label}_captions.csv"))
            except FileNotFoundError:
                continue

            text_context = dict(zip(df_split['id'].astype(str), df_split['title'].fillna("").astype(str)))
            img_context = dict(zip(df_cap['id'].astype(str), df_cap['caption'].fillna("").astype(str)))

            all_inputs, mapping_tracker = [], []
            for i, mid in enumerate(master_ids):
                mid = str(mid)
                t_ctx, i_ctx = text_context.get(mid, ""), img_context.get(mid, "")

                # Process text entities with title context
                if i < len(text_ents):
                    for ent in text_ents[i]:
                        processed = truncate_input(insert_markers(clean_entity_text(ent), t_ctx))
                        all_inputs.append(processed)
                        mapping_tracker.append(i)

                # Process image entities with caption context
                if i < len(img_ents):
                    for ent in img_ents[i]:
                        processed = truncate_input(insert_markers(clean_entity_text(ent), i_ctx))
                        all_inputs.append(processed)
                        mapping_tracker.append(i)

            # Generate Wikipedia titles using mGENRE inference
            wiki_titles = []
            if all_inputs:
                for i in tqdm(range(0, len(all_inputs), BATCH_SIZE), desc="Disambiguating"):
                    batch = all_inputs[i:i + BATCH_SIZE]
                    with torch.no_grad():
                        outputs = model.sample(batch, skip_invalid_size_inputs=True)
                    for out in outputs:
                        wiki_titles.append(out[0]['text'].split(" >> ")[0].replace(" ", "_"))

            # Fetch QIDs for disambiguated titles
            qid_map = fetch_wikidata_qids(wiki_titles) if wiki_titles else {}
            
            final_qid_results = [set() for _ in range(len(master_ids))]
            for idx, title in enumerate(wiki_titles):
                qid = qid_map.get(title)
                if qid:
                    final_qid_results[mapping_tracker[idx]].add(qid)

            # Save the final aligned QID lists
            pd.to_pickle([list(q) for q in final_qid_results], output_file)

if __name__ == "__main__":
    run_mapping_pipeline()
