"""Separate vocals from JFla songs using Demucs"""
import subprocess
import os
import sys
import glob
import shutil

input_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_audio"
output_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_vocals"
os.makedirs(output_dir, exist_ok=True)

# Get all wav files
wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
print(f"Found {len(wav_files)} files to process")

python_exe = r"C:\Users\thtgg\workspace2\GPT-SoVITS\.venv\Scripts\python.exe"

for i, wav_file in enumerate(wav_files, 1):
    fname = os.path.basename(wav_file)
    print(f"\n[{i}/{len(wav_files)}] Processing: {fname}")

    # Run demucs with htdemucs model (best quality)
    cmd = [
        python_exe, "-m", "demucs",
        "--two-stems", "vocals",  # Only separate vocals vs other
        "-n", "htdemucs",         # Best model
        "--out", os.path.join(input_dir, "separated"),
        wav_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        continue

    # Find the vocals output
    # Demucs outputs to: separated/htdemucs/<filename_without_ext>/vocals.wav
    stem = os.path.splitext(fname)[0]
    vocals_path = os.path.join(input_dir, "separated", "htdemucs", stem, "vocals.wav")

    if os.path.exists(vocals_path):
        # Copy to output dir with original name
        dest = os.path.join(output_dir, fname)
        shutil.copy2(vocals_path, dest)
        size_mb = os.path.getsize(dest) / (1024*1024)
        print(f"  Vocals saved: {size_mb:.1f} MB")
    else:
        print(f"  ERROR: Vocals file not found at {vocals_path}")
        # List what's in separated dir
        sep_dir = os.path.join(input_dir, "separated")
        if os.path.exists(sep_dir):
            for root, dirs, files in os.walk(sep_dir):
                for f in files:
                    print(f"    Found: {os.path.join(root, f)}")

# Summary
print(f"\n=== Summary ===")
vocal_files = glob.glob(os.path.join(output_dir, "*.wav"))
total_size = 0
for f in sorted(vocal_files):
    size_mb = os.path.getsize(f) / (1024*1024)
    total_size += size_mb
    print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")
print(f"Total: {len(vocal_files)} files, {total_size:.1f} MB")
