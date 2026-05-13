"""
This script implements a hybrid downloading strategy for PolitiFact news images.
It first attempts a standard HTTP request and falls back to cloudscraper if 
a 403 error is encountered. It includes domain tracking to skip dead links 
and uses thread pooling for parallel processing to optimize download speed.
"""

import os
import csv
import time
import threading
import warnings
import urllib3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from urllib.parse import urlparse, urljoin

# Conditional import for cloudscraper
try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

urllib3.disable_warnings()
warnings.filterwarnings('ignore')

# Configuration and path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
BASE_DATA_PATH = os.path.join(project_root, "data", "politifact")

CONNECT_TIMEOUT = 3
READ_TIMEOUT = 6
MAX_WORKERS = 6
LOG_FILE = "download_errors.log"

# Global tracking for domain health
failed_domains = defaultdict(int)
dead_domains = set()
protected_domains = set() 
domain_locks = defaultdict(threading.Lock)
thread_local = threading.local()

def log_error(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"

def is_domain_dead(domain):
    return domain in dead_domains or failed_domains[domain] >= 10

def get_sessions():
    """Provides thread-safe access to standard and cloudscraper sessions."""
    if not hasattr(thread_local, "fast_session"):
        thread_local.fast_session = requests.Session()
        thread_local.fast_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        if HAS_CLOUDSCRAPER:
            thread_local.cloud_session = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
            )
        else:
            thread_local.cloud_session = None
    return thread_local.fast_session, thread_local.cloud_session

def extract_image_from_html(soup, base_url):
    """Searches HTML metadata and tags for the most relevant article image."""
    # Check OpenGraph and Twitter metadata
    for attr in [{"property": "og:image"}, {"name": "twitter:image"}, {"property": "twitter:image"}]:
        meta = soup.find("meta", attrs=attr)
        if meta and meta.get("content"):
            img_url = meta["content"]
            return urljoin(base_url, img_url) if img_url.startswith(("/", "//")) else img_url

    # Fallback to general image tags, excluding common UI elements
    for img in soup.find_all("img", limit=15):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src or any(x in src.lower() for x in ['logo', 'icon', 'avatar', 'sprite', 'pixel']):
            continue
        return urljoin(base_url, src) if src.startswith(("/", "//")) else src

    return None

def extract_image_url(url):
    """Executes the request and handles 403 retries via cloudscraper."""
    domain = get_domain(url)
    if is_domain_dead(domain):
        return None
    
    fast_session, cloud_session = get_sessions()
    use_cloud_first = domain in protected_domains
    
    try:
        if use_cloud_first and cloud_session:
            with domain_locks[domain]:
                time.sleep(1.0)
                response = cloud_session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        else:
            response = fast_session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), verify=False)
            if response.status_code == 403 and cloud_session:
                protected_domains.add(domain)
                with domain_locks[domain]:
                    time.sleep(1.0)
                    response = cloud_session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        
        if response.status_code != 200:
            failed_domains[domain] += 1
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        return extract_image_from_html(soup, url)

    except Exception:
        failed_domains[domain] += 1
        return None

def download_image(img_url, save_path):
    """Downloads and validates image content, converting to RGB if necessary."""
    fast_session, _ = get_sessions()
    try:
        response = fast_session.get(img_url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), verify=False)
        if response.status_code != 200 or len(response.content) < 1000:
            return False

        img = Image.open(BytesIO(response.content))
        img.verify() # Basic corruption check
        
        if img.size[0] < 100 or img.size[1] < 100:
            return False
        
        img = Image.open(BytesIO(response.content))
        if save_path.lower().endswith(('.jpg', '.jpeg')) and img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            img = img.convert('RGBA') if img.mode == 'P' else img
            bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = bg
        
        img.save(save_path, quality=85, optimize=True)
        return os.path.exists(save_path)
    except Exception:
        return False

def process_article(row, images_dir, count, label):
    title = row.get("title")
    url = row.get("news_url") or row.get("url")
    if not isinstance(title, str) or not isinstance(url, str):
        return None

    img_url = extract_image_url(url)
    if not img_url:
        return None

    # Handle file extensions and define local path
    ext = os.path.splitext(img_url.split('?')[0])[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]:
        ext = ".jpg"
    
    img_name = f"{count}{ext}"
    local_path = os.path.join(images_dir, img_name)

    if download_image(img_url, local_path):
        return [count, title, local_path, label]
    return None

def run_scraper(label):
    """Processes the specific PolitiFact label (fake or real)."""
    print(f"Starting download for PolitiFact {label} data...")
    
    label_path = os.path.join(BASE_DATA_PATH, label)
    images_path = os.path.join(label_path, "images")
    os.makedirs(images_path, exist_ok=True)

    csv_path = os.path.join(label_path, f"politifact_{label}.csv")
    mapping_path = os.path.join(label_path, "mapping.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    processed_titles = set()
    current_id = 0

    if os.path.exists(mapping_path):
        existing_df = pd.read_csv(mapping_path)
        if not existing_df.empty:
            current_id = existing_df["id"].max() + 1
            processed_titles = set(existing_df["text"].tolist())
        write_mode = "a"
    else:
        write_mode = "w"

    todo_df = df[~df["title"].isin(processed_titles)].copy()
    
    with open(mapping_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_mode == "w":
            writer.writerow(["id", "text", "image_path", "label"])

        tasks = []
        for i, (idx, row) in enumerate(todo_df.iterrows()):
            tasks.append((row, images_path, current_id + i, label))
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_article, *t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Downloading {label}"):
                result = future.result()
                if result:
                    writer.writerow(result)
                    f.flush()

if __name__ == "__main__":
    if not HAS_CLOUDSCRAPER:
        print("Warning: Cloudscraper is not installed. Some downloads may fail with 403 errors.")
    
    for label in ["fake", "real"]:
        try:
            run_scraper(label)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress.")
            break
        except Exception as e:
            print(f"An error occurred while processing {label}: {e}")
