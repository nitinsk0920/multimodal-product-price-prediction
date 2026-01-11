#DOWNLOAD TRAIN SET IMAGES FROM THE URL

import os
import pandas as pd
import requests
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import time

train_csv_path = "/kaggle/input/mlhack/train.csv"
train_image_folder = "/kaggle/working/train_images"
IMAGE_COLUMN = "image_link"
MAX_WORKERS = min(64, cpu_count())  # Adjust concurrency

# Create directories
os.makedirs(train_image_folder, exist_ok=True)

# Load CSVs
train_df = pd.read_csv(train_csv_path)

print("✅ Columns in train:", train_df.columns.tolist())

# Extract image links
train_links = train_df[IMAGE_COLUMN].dropna().unique().tolist()

print(f"Train images to download: {len(train_links)}")



def download_image(image_url, save_folder):
    try:
        filename = Path(image_url).name
        save_path = os.path.join(save_folder, filename)

        # Skip if already downloaded
        if os.path.exists(save_path):
            return "exists"

        headers = {"User-Agent": "Mozilla/5.0"}  # Prevent blocking
        response = requests.get(image_url, headers=headers, timeout=10)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return "ok"
        else:
            return f"error_{response.status_code}"

    except Exception as e:
        return f"fail_{str(e)[:40]}"



def parallel_download(image_links, folder):
    print(f"📥 Starting downloads → {folder}")
    start = time.time()
    os.makedirs(folder, exist_ok=True)
    
    download_fn = partial(download_image, save_folder=folder)
    
    results = []
    with Pool(MAX_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(download_fn, image_links), total=len(image_links)):
            results.append(result)
    
    ok = sum(1 for r in results if r == "ok" or r == "exists")
    fail = len(results) - ok
    print(f"✅ Completed: {ok} |⚠️ Failed: {fail} | ⏱️ Time: {round(time.time() - start, 2)}s")
    return ok, fail



ok_train, fail_train = parallel_download(train_links, train_image_folder)

print("\n📊 FINAL STATS:")
print(f"Train → Downloaded: {ok_train}, Failed: {fail_train}")

# Optional sanity check
print(f"\n🖼️ Train images in folder: {len(os.listdir(train_image_folder))}")




#Mapping downloaded train images to their corresponding text(catalog_content) and url
import pandas as pd
import os

# Load your dataset
train_df = pd.read_csv("/kaggle/input/mlhack/train.csv")

# Extract only the useful columns
train_df = train_df[["image_link", "catalog_content"]]

# Derive image filenames from URLs (if needed)
train_df["image_name"] = train_df["image_link"].apply(lambda x: os.path.basename(x))

# Keep only rows where the image actually exists
image_folder = "/kaggle/working/train_images"
train_df["exists"] = train_df["image_name"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
train_df = train_df[train_df["exists"] == True].reset_index(drop=True)

print("✅ Final usable samples:", len(train_df))


#DOWNLOAD TEST SET IMAGES FROM THE URL:
import os
import pandas as pd
import requests
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import time

train_csv_path = "/kaggle/input/mlhack/test.csv"
train_image_folder = "/kaggle/working/test_images"
IMAGE_COLUMN = "image_link"
MAX_WORKERS = min(64, cpu_count())  # Adjust concurrency

# Create directories
os.makedirs(train_image_folder, exist_ok=True)

# Load CSVs
train_df = pd.read_csv(train_csv_path)

print("✅ Columns in train:", train_df.columns.tolist())

# Extract image links
train_links = train_df[IMAGE_COLUMN].dropna().unique().tolist()

print(f"Train images to download: {len(train_links)}")



def download_image(image_url, save_folder):
    try:
        filename = Path(image_url).name
        save_path = os.path.join(save_folder, filename)

        # Skip if already downloaded
        if os.path.exists(save_path):
            return "exists"

        headers = {"User-Agent": "Mozilla/5.0"}  # Prevent blocking
        response = requests.get(image_url, headers=headers, timeout=10)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return "ok"
        else:
            return f"error_{response.status_code}"

    except Exception as e:
        return f"fail_{str(e)[:40]}"

def parallel_download(image_links, folder):
    print(f"📥 Starting downloads → {folder}")
    start = time.time()
    os.makedirs(folder, exist_ok=True)
    
    download_fn = partial(download_image, save_folder=folder)
    
    results = []
    with Pool(MAX_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(download_fn, image_links), total=len(image_links)):
            results.append(result)
    
    ok = sum(1 for r in results if r == "ok" or r == "exists")
    fail = len(results) - ok
    print(f"✅ Completed: {ok} | ⚠️ Failed: {fail} | ⏱️ Time: {round(time.time() - start, 2)}s")
    return ok, fail



ok_train, fail_train = parallel_download(train_links, train_image_folder)

print("\n📊 FINAL STATS:")
print(f"Train → Downloaded: {ok_train}, Failed: {fail_train}")

# Optional sanity check
print(f"\n🖼️ Train images in folder: {len(os.listdir(train_image_folder))}")


#Mapping downloaded test images to their corresponding text(catalog_content) and url

import pandas as pd
import os

# Load your dataset
train_df = pd.read_csv("/kaggle/input/mlhack/test.csv")

# Extract only the useful columns
train_df = train_df[["image_link", "catalog_content"]]

# Derive image filenames from URLs (if needed)
train_df["image_name"] = train_df["image_link"].apply(lambda x: os.path.basename(x))

# Keep only rows where the image actually exists
image_folder = "/kaggle/working/test_images"
train_df["exists"] = train_df["image_name"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
train_df = train_df[train_df["exists"] == True].reset_index(drop=True)
print("✅ Final usable samples:", len(train_df))

