"""Upload fine-tuned GPT-SoVITS test samples to Google Drive"""
import os
import sys
import subprocess
import json

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")

# Install google API client if needed
subprocess.run([PYTHON_EXE, "-m", "pip", "install", "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib", "-q"],
               capture_output=True)

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Try using existing credentials
# The previous session used OAuth - let's try to find the token
token_path = os.path.join(GPT_SOVITS_DIR, "gdrive_token.json")
creds_path = os.path.join(GPT_SOVITS_DIR, "credentials.json")

# Check what auth files exist
for p in [token_path, creds_path]:
    print(f"  {os.path.basename(p)}: {'EXISTS' if os.path.exists(p) else 'NOT FOUND'}")

# Try to use the token from previous session
if os.path.exists(token_path):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    creds = Credentials.from_authorized_user_file(token_path)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_path, 'w') as f:
            f.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    # Target folder: "시온 데뷔곡 > GPT-SoVITS 음성 샘플 (JFla)"
    folder_id = "1y1H0K1nJV6CpVWcQYak20RGvjEU4-YDh"

    # Create subfolder for fine-tuned samples
    subfolder_metadata = {
        'name': 'GPT-SoVITS 파인튜닝 샘플',
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [folder_id]
    }
    subfolder = service.files().create(body=subfolder_metadata, fields='id').execute()
    subfolder_id = subfolder.get('id')
    print(f"Created folder: GPT-SoVITS 파인튜닝 샘플 (ID: {subfolder_id})")

    # Upload files
    test_dir = os.path.join(GPT_SOVITS_DIR, "test_output_finetuned")
    files_to_upload = [f for f in sorted(os.listdir(test_dir))
                       if f.endswith('.wav') and ('v3' in f or 'pretrained' in f)]

    for fname in files_to_upload:
        fpath = os.path.join(test_dir, fname)
        metadata = {
            'name': fname,
            'parents': [subfolder_id]
        }
        media = MediaFileUpload(fpath, mimetype='audio/wav')
        file = service.files().create(body=metadata, media_body=media, fields='id,name').execute()
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  Uploaded: {fname} ({size_kb:.0f} KB) -> {file.get('id')}")

    print(f"\nAll files uploaded to Google Drive!")
    print(f"Folder: https://drive.google.com/drive/folders/{subfolder_id}")
else:
    print("\nNo Google Drive token found. Skipping upload.")
    print("Files are available locally at:")
    test_dir = os.path.join(GPT_SOVITS_DIR, "test_output_finetuned")
    for f in sorted(os.listdir(test_dir)):
        if f.endswith('.wav') and ('v3' in f or 'pretrained' in f):
            print(f"  {os.path.join(test_dir, f)}")
