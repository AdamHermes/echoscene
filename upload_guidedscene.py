import sys
import os
from huggingface_hub import HfApi

LOCAL_DIR = '/Volumes/ExternalSSD/current_works'
REPO_ID = 'rabbitKabbit/GuidedScene'
REPO_TYPE = 'dataset'

if not os.path.exists(LOCAL_DIR):
    print(f"❌ Error: Local directory '{LOCAL_DIR}' does not exist. Please make sure the external SSD is connected.")
    sys.exit(1)

print(f"🚀 Starting upload of '{LOCAL_DIR}' to Hugging Face repository '{REPO_ID}' (repo_type='{REPO_TYPE}')...")

api = HfApi()

try:
    commit_info = api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="current_works",
        commit_message="Update current_works dataset with new works from external SSD",
        ignore_patterns=[
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
    )
    print("\n✅ Upload Completed Successfully!")
    print(f"Commit URL: {commit_info.commit_url if hasattr(commit_info, 'commit_url') else commit_info}")
except Exception as e:
    print(f"\n❌ Error during upload: {e}")
    sys.exit(1)
