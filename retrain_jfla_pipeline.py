"""Full retrain pipeline for JFla extended (14 songs).
Steps: slice → ASR → 1a/1b/1c preprocessing
"""
import os
import sys
import shutil

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, gpt_sovits_pkg)
os.chdir(GPT_SOVITS_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

EXP_NAME = "sion_jfla_v2"
VOCALS_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_vocals_extended")
SLICED_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_sliced_v2")
ASR_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_asr_v2")
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)

os.makedirs(SLICED_DIR, exist_ok=True)
os.makedirs(ASR_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

# ============================================================
# STEP 1: Slice audio
# ============================================================
print("=" * 60)
print("STEP 1: Slicing audio...")
print("=" * 60)

import subprocess

wav_files = sorted([f for f in os.listdir(VOCALS_DIR) if f.endswith('.wav')])
print(f"Found {len(wav_files)} vocal files to slice")

for f in wav_files:
    input_path = os.path.join(VOCALS_DIR, f)
    print(f"  Slicing {f}...")
    try:
        slice_script = os.path.join(GPT_SOVITS_DIR, "tools", "slice_audio.py")
        cmd = [
            sys.executable, slice_script,
            input_path, SLICED_DIR,
            "-34", "4000", "300", "10", "500", "0.9", "0.3", "0", "1"
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(GPT_SOVITS_DIR, "tools") + os.pathsep + GPT_SOVITS_DIR
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, cwd=GPT_SOVITS_DIR, env=env)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:300]}")
        else:
            print(f"    OK")
    except Exception as e:
        print(f"    ERROR: {e}")

sliced_count = len([f for f in os.listdir(SLICED_DIR) if f.endswith('.wav')])
print(f"\nSliced segments: {sliced_count}")

# ============================================================
# STEP 2: ASR
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: ASR (faster-whisper)...")
print("=" * 60)

from tools.asr.fasterwhisper_asr import execute_asr

asr_output = os.path.join(ASR_DIR, "jfla_sliced_v2.list")
execute_asr(
    input_folder=SLICED_DIR,
    output_folder=ASR_DIR,
    model_path="medium",
    language="auto",
    precision="float16",
)

# Find the generated list file
list_files = [f for f in os.listdir(ASR_DIR) if f.endswith('.list')]
if list_files:
    asr_file = os.path.join(ASR_DIR, list_files[0])
    with open(asr_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filter valid entries (duration 1-20s, has text)
    valid = []
    import soundfile as sf
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 4:
            continue
        wav_path = parts[0].strip()
        text = parts[3].strip()
        if not text or len(text) < 2:
            continue
        if os.path.exists(wav_path):
            data, sr = sf.read(wav_path)
            dur = len(data) / sr
            if 1 <= dur <= 20:
                valid.append(line)

    filtered_file = os.path.join(ASR_DIR, "jfla_sliced_v2_filtered.list")
    with open(filtered_file, 'w', encoding='utf-8') as f:
        f.writelines(valid)
    print(f"ASR: {len(lines)} total → {len(valid)} valid entries")
else:
    print("ERROR: No .list file found!")
    sys.exit(1)

# ============================================================
# STEP 3: 1a - Phonemes & BERT
# ============================================================
print("\n" + "=" * 60)
print("STEP 3a: Phonemes & BERT features...")
print("=" * 60)

prep_env = os.environ.copy()
prep_env["inp_text"] = filtered_file
prep_env["inp_wav_dir"] = SLICED_DIR
prep_env["exp_name"] = EXP_NAME
prep_env["opt_dir"] = EXP_DIR
prep_env["bert_pretrained_dir"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "chinese-roberta-wwm-ext-large")
prep_env["cnhubert_base_dir"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "chinese-hubert-base")
prep_env["pretrained_s2G"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
prep_env["s2config_path"] = os.path.join(gpt_sovits_pkg, "configs", "s2.json")
prep_env["is_half"] = "True"
prep_env["i_part"] = "0"
prep_env["all_parts"] = "1"
prep_env["_CUDA_VISIBLE_DEVICES"] = "0"
prep_env["PYTHONPATH"] = GPT_SOVITS_DIR + os.pathsep + gpt_sovits_pkg
prep_env["PYTHONIOENCODING"] = "utf-8"

# Step 1a: get-text (phonemes & BERT)
print("Running 1-get-text.py...")
result = subprocess.run(
    [sys.executable, os.path.join(gpt_sovits_pkg, "prepare_datasets", "1-get-text.py")],
    env=prep_env, cwd=GPT_SOVITS_DIR,
    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600
)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print(f"1a ERROR: {result.stderr[-500:]}")

# Rename files with "-0" suffix
for f in os.listdir(EXP_DIR):
    if f.endswith("-0.txt") or f.endswith("-0.tsv"):
        new_name = f.replace("-0.txt", ".txt").replace("-0.tsv", ".tsv")
        src = os.path.join(EXP_DIR, f)
        dst = os.path.join(EXP_DIR, new_name)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"  Renamed: {f} → {new_name}")

# ============================================================
# STEP 3b: HuBERT & wav32k
# ============================================================
print("\n" + "=" * 60)
print("STEP 3b: HuBERT & wav32k features...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, os.path.join(gpt_sovits_pkg, "prepare_datasets", "2-get-hubert-wav32k.py")],
    env=prep_env, cwd=GPT_SOVITS_DIR,
    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600
)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print(f"1b ERROR: {result.stderr[-500:]}")

# ============================================================
# STEP 3c: Semantic tokens
# ============================================================
print("\n" + "=" * 60)
print("STEP 3c: Semantic tokens...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, os.path.join(gpt_sovits_pkg, "prepare_datasets", "3-get-semantic.py")],
    env=prep_env, cwd=GPT_SOVITS_DIR,
    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600
)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print(f"1c ERROR: {result.stderr[-500:]}")

# Rename semantic file
for f in os.listdir(EXP_DIR):
    if "name2semantic" in f and f.endswith("-0.tsv"):
        new_name = f.replace("-0.tsv", ".tsv")
        src = os.path.join(EXP_DIR, f)
        dst = os.path.join(EXP_DIR, new_name)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"  Renamed: {f} → {new_name}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)

for subdir in ["2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k", "6-name2semantic.tsv"]:
    path = os.path.join(EXP_DIR, subdir)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"  {subdir}: {count} entries")
    elif os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"  {subdir}/: {count} files")
    else:
        print(f"  {subdir}: NOT FOUND")

print("\nREADY FOR TRAINING")
print(f"Experiment: {EXP_NAME}")
print(f"Directory: {EXP_DIR}")
