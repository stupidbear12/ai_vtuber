"""Run ASR on mixed sliced vocals"""
import sys, os

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS"))
os.chdir(GPT_SOVITS_DIR)

from tools.asr.fasterwhisper_asr import execute_asr, get_models

SLICED_DIR = os.path.join(GPT_SOVITS_DIR, "mixed_sliced")
ASR_DIR = os.path.join(GPT_SOVITS_DIR, "mixed_asr")
os.makedirs(ASR_DIR, exist_ok=True)

n_files = len([f for f in os.listdir(SLICED_DIR) if f.endswith('.wav')])
print(f"Running ASR on {n_files} files...")

# Use medium model for better accuracy with mixed Korean/English
execute_asr(SLICED_DIR, ASR_DIR, "medium", "auto", "float16")

# Check results
list_file = os.path.join(ASR_DIR, "mixed_sliced.list")
if os.path.exists(list_file):
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    valid = [l for l in lines if l.strip() and len(l.strip().split('|')) >= 4 and l.strip().split('|')[3].strip()]
    print(f"Total: {len(lines)}, with text: {len(valid)}")

    # Save filtered
    filtered = os.path.join(ASR_DIR, "mixed_sliced_filtered.list")
    with open(filtered, 'w', encoding='utf-8') as f:
        f.writelines(valid)
    print(f"Filtered list: {filtered}")
else:
    print("ASR list not found!")

print("ASR done!")
