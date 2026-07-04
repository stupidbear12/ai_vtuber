"""Demucs vocal separation for JFla v3 (13 originals)"""
import os
import subprocess
import shutil

input_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_audio_v3"
output_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_vocals_v3"
os.makedirs(output_dir, exist_ok=True)

wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])
print(f"Found {len(wav_files)} WAV files to separate")

for i, f in enumerate(wav_files, 1):
    out_name = f.replace('.wav', '')
    expected_output = os.path.join(output_dir, f)
    if os.path.exists(expected_output):
        print(f"[{i}/{len(wav_files)}] SKIP {f} (already exists)")
        continue

    print(f"\n[{i}/{len(wav_files)}] Separating {f}...")
    input_path = os.path.join(input_dir, f)

    cmd = [
        r"C:\Users\thtgg\workspace2\GPT-SoVITS\.venv\Scripts\python.exe",
        "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", os.path.join(output_dir, "_temp"),
        input_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}")
        continue

    # Move vocals file to output dir
    vocals_path = os.path.join(output_dir, "_temp", "htdemucs", out_name, "vocals.wav")
    if os.path.exists(vocals_path):
        shutil.move(vocals_path, expected_output)
        print(f"  OK: {os.path.getsize(expected_output)/(1024*1024):.1f} MB")
    else:
        print(f"  FAILED: vocals.wav not found")

# Cleanup temp
temp_dir = os.path.join(output_dir, "_temp")
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)

# Summary
print(f"\n{'='*60}")
vocals = sorted([f for f in os.listdir(output_dir) if f.endswith('.wav')])
total = 0
for f in vocals:
    sz = os.path.getsize(os.path.join(output_dir, f)) / (1024*1024)
    total += sz
    print(f"  {f} ({sz:.1f} MB)")
print(f"\nTotal: {len(vocals)} files, {total:.1f} MB")
