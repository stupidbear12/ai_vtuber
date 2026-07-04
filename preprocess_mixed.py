"""Preprocess mixed JFla + COOMO vocals for GPT-SoVITS training"""
import os
import subprocess
import shutil
import sys

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")

# Directories
JFLA_VOCALS = os.path.join(GPT_SOVITS_DIR, "jfla_vocals")
COOMO_VOCALS = os.path.join(GPT_SOVITS_DIR, "coomo_vocals")
MIXED_VOCALS = os.path.join(GPT_SOVITS_DIR, "mixed_vocals")
MIXED_SLICED = os.path.join(GPT_SOVITS_DIR, "mixed_sliced")
MIXED_ASR = os.path.join(GPT_SOVITS_DIR, "mixed_asr")
EXP_NAME = "sion_mixed"
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)

# Step 1: Combine JFla + COOMO vocals into one directory
print("=" * 60)
print("Step 1: Combining JFla + COOMO vocals")
os.makedirs(MIXED_VOCALS, exist_ok=True)

count = 0
for src_dir, prefix in [(JFLA_VOCALS, "jfla"), (COOMO_VOCALS, "coomo")]:
    if not os.path.exists(src_dir):
        print(f"  WARNING: {src_dir} not found!")
        continue
    for f in sorted(os.listdir(src_dir)):
        if f.endswith('.wav'):
            src = os.path.join(src_dir, f)
            dst = os.path.join(MIXED_VOCALS, f"{prefix}_{f}" if not f.startswith(prefix) else f)
            shutil.copy2(src, dst)
            sz = os.path.getsize(dst) / (1024*1024)
            print(f"  {os.path.basename(dst)} ({sz:.1f} MB)")
            count += 1
print(f"  Total: {count} files")

# Step 2: Slice audio
print("\n" + "=" * 60)
print("Step 2: Slicing audio")
os.makedirs(MIXED_SLICED, exist_ok=True)

gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
env = os.environ.copy()
env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"

slice_cmd = [
    PYTHON_EXE, "-c",
    f"""
import sys
sys.path.insert(0, r'{GPT_SOVITS_DIR}')
sys.path.insert(0, r'{gpt_sovits_pkg}')
from tools.slice_audio import slice_audio
slice_audio(
    inp=r'{MIXED_VOCALS}',
    opt_root=r'{MIXED_SLICED}',
    threshold=-34,
    min_length=4000,
    min_interval=300,
    hop_size=10,
    max_sil_kept=500,
    _max=0.9,
    alpha=0.25,
    n_parts=1,
    i_part=0
)
print('Slicing done!')
"""
]
result = subprocess.run(slice_cmd, capture_output=True, text=True, env=env, timeout=300)
if result.returncode != 0:
    print(f"  FAILED: {result.stderr[-500:]}")
    sys.exit(1)

sliced_files = [f for f in os.listdir(MIXED_SLICED) if f.endswith('.wav')]
print(f"  Sliced into {len(sliced_files)} segments")

# Step 3: ASR with faster-whisper
print("\n" + "=" * 60)
print("Step 3: ASR transcription")
os.makedirs(MIXED_ASR, exist_ok=True)

asr_cmd = [
    PYTHON_EXE, "-c",
    f"""
import sys, os
sys.path.insert(0, r'{GPT_SOVITS_DIR}')
sys.path.insert(0, r'{gpt_sovits_pkg}')
os.chdir(r'{GPT_SOVITS_DIR}')

from tools.asr.fasterwhisper import execute
# Detect language - mixed English (JFla) and Korean (COOMO)
# Use auto-detect since we have mixed languages
execute(
    r'{MIXED_SLICED}',
    'medium',
    'auto',
    0,
    r'{MIXED_ASR}',
    'all',
    1,
    0
)
print('ASR done!')
"""
]
result = subprocess.run(asr_cmd, capture_output=True, text=True, env=env, timeout=600)
if result.returncode != 0:
    print(f"  FAILED: {result.stderr[-500:]}")
    # Try with base model if medium fails
    print("  Retrying with base model...")
    asr_cmd_base = [
        PYTHON_EXE, "-c",
        f"""
import sys, os
sys.path.insert(0, r'{GPT_SOVITS_DIR}')
sys.path.insert(0, r'{gpt_sovits_pkg}')
os.chdir(r'{GPT_SOVITS_DIR}')

from tools.asr.fasterwhisper import execute
execute(
    r'{MIXED_SLICED}',
    'base',
    'auto',
    0,
    r'{MIXED_ASR}',
    'all',
    1,
    0
)
print('ASR done!')
"""
    ]
    result = subprocess.run(asr_cmd_base, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED again: {result.stderr[-500:]}")
        sys.exit(1)

asr_list = os.path.join(MIXED_ASR, "mixed_sliced.list")
if os.path.exists(asr_list):
    with open(asr_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    valid = [l for l in lines if l.strip() and len(l.strip().split('|')) >= 4 and l.strip().split('|')[3].strip()]
    print(f"  Total entries: {len(lines)}, with text: {len(valid)}")

    # Filter out empty transcriptions
    filtered_list = os.path.join(MIXED_ASR, "mixed_sliced_filtered.list")
    with open(filtered_list, 'w', encoding='utf-8') as f:
        f.writelines(valid)
    print(f"  Filtered list saved: {filtered_list}")
else:
    print(f"  ASR list not found!")
    sys.exit(1)

print("\n" + "=" * 60)
print("Preprocessing Steps 1-3 complete!")
print(f"Next: Run 1a/1b/1c preprocessing with exp_name={EXP_NAME}")
