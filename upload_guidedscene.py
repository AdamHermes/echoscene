import sys, os
from huggingface_hub import HfApi

LOCAL_DIR = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works'
REPO_ID = 'rabbitKabbit/GuidedScene'
REPO_TYPE = 'dataset'

print(f"Starting upload of '{LOCAL_DIR}' to Hugging Face repository '{REPO_ID}'...")

api = HfApi()

try:
    commit_info = api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="current_works",
        commit_message="Upload all current_works benchmark datasets and model outputs",
        ignore_patterns=["*.DS_Store", ".*"]
    )
    print("\n✅ Upload Completed Successfully!")
    print(f"Commit URL: {commit_info.commit_url if hasattr(commit_info, 'commit_url') else commit_info}")
except Exception as e:
    print(f"\n❌ Error during upload: {e}")
    sys.exit(1)
