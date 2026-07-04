"""Separate vocals from COOMO audio using Demucs"""
import os
import subprocess
import shutil

INPUT_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS\coomo_audio"
OUTPUT_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS\coomo_vocals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')])
print(f"Processing {len(files)} files with Demucs (htdemucs)...\n")

for i, fname in enumerate(files, 1):
    fpath = os.path.join(INPUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    print(f"[{i}/{len(files)}] {fname} ({size_mb:.1f} MB)")

    cmd = [
        PYTHON_EXE, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", os.path.join(GPT_SOVITS_DIR, "demucs_out"),
        fpath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode == 0:
        # Move vocal file to output dir
        stem = os.path.splitext(fname)[0]
        vocal_path = os.path.join(GPT_SOVITS_DIR, "demucs_out", "htdemucs", stem, "vocals.wav")
        if os.path.exists(vocal_path):
            dest = os.path.join(OUTPUT_DIR, f"{stem}_vocals.wav")
            shutil.copy2(vocal_path, dest)
            vsize = os.path.getsize(dest) / (1024*1024)
            print(f"  OK -> {dest} ({vsize:.1f} MB)")
        else:
            print(f"  Vocal file not found at {vocal_path}")
    else:
        print(f"  FAILED: {result.stderr[-300:]}")

print(f"\nDone! Vocals in {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024*1024)
    print(f"  {f} ({sz:.1f} MB)")
