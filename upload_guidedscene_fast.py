import os
import sys
import time
from huggingface_hub import HfApi

# Enable HF Transfer (Rust-accelerated multi-part transfers)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

LOCAL_DIR = '/Volumes/ExternalSSD/current_works'
REPO_ID = 'rabbitKabbit/GuidedScene'
REPO_TYPE = 'dataset'

if not os.path.exists(LOCAL_DIR):
    print(f"❌ Error: Local directory '{LOCAL_DIR}' does not exist.")
    sys.exit(1)

api = HfApi()

# High-priority new folders generated after the previous upload
NEW_FOLDERS = [
    "work_41",
]

# Discover any other subfolders on SSD
all_subfolders = sorted([
    f for f in os.listdir(LOCAL_DIR) 
    if not f.startswith(".") and os.path.isdir(os.path.join(LOCAL_DIR, f))
])

# Ensure new folders come first, followed by others
queue = [f for f in NEW_FOLDERS if f in all_subfolders]
for f in all_subfolders:
    if f not in queue:
        queue.append(f)

print(f"🚀 Starting accelerated per-folder sync to '{REPO_ID}' ({len(queue)} folders in queue)...")
print(f"⚡ HF_HUB_ENABLE_HF_TRANSFER is ACTIVE.\n")

ignore_patterns = [
    "*.DS_Store",
    ".*",
    "._*",
    "*/.*",
    "*/._*",
    "*$RECYCLE.BIN*",
    "*.Spotlight*",
    "*.TemporaryItems*",
    "*.Trashes*",
    "*.fseventsd*"
]

total_start = time.time()

for idx, folder in enumerate(queue, 1):
    folder_path = os.path.join(LOCAL_DIR, folder)
    local_files = [f for dp, _, fns in os.walk(folder_path) for f in fns if not f.startswith(".")]
    
    if len(local_files) == 0:
        print(f"[{idx}/{len(queue)}] ⏭️  Skipping empty folder: '{folder}'")
        continue

    total_sz_mb = sum(
        os.path.getsize(os.path.join(dp, f)) 
        for dp, _, fns in os.walk(folder_path) 
        for f in fns if not f.startswith(".")
    ) / (1024 * 1024)

    print(f"\n📤 [{idx}/{len(queue)}] Uploading '{folder}' ({len(local_files)} files, {total_sz_mb:.1f} MB)...")
    start_time = time.time()
    
    try:
        commit_info = api.upload_folder(
            folder_path=folder_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=f"current_works/{folder}",
            commit_message=f"Upload {folder} ({len(local_files)} files, {total_sz_mb:.1f} MB)",
            ignore_patterns=ignore_patterns
        )
        elapsed = time.time() - start_time
        speed_mbps = (total_sz_mb / elapsed) if elapsed > 0 else 0
        print(f"  ✅ '{folder}' uploaded successfully in {elapsed:.1f}s (~{speed_mbps:.2f} MB/s)!")
    except Exception as e:
        print(f"  ❌ Error uploading '{folder}': {e}")
        continue

total_elapsed = time.time() - total_start
print("\n" + "=" * 70)
print(f"🎉 Complete sync finished in {total_elapsed / 60:.1f} minutes!")
