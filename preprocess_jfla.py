"""GPT-SoVITS preprocessing: slice audio + ASR transcription"""
import os
import sys
import subprocess

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")
ENV = os.environ.copy()
ENV["PYTHONPATH"] = GPT_SOVITS_DIR
VOCALS_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_vocals")
SLICED_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_sliced")
ASR_OUTPUT_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_asr")

os.makedirs(SLICED_DIR, exist_ok=True)
os.makedirs(ASR_OUTPUT_DIR, exist_ok=True)

# Step 1: Slice audio
print("=" * 60)
print("STEP 1: Slicing audio into segments")
print("=" * 60)

# slice_audio.py args: inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, all_part
cmd = [
    PYTHON_EXE,
    os.path.join(GPT_SOVITS_DIR, "tools", "slice_audio.py"),
    VOCALS_DIR,     # input directory
    SLICED_DIR,     # output directory
    "-34",          # threshold (dB)
    "4000",         # min_length (ms)
    "300",          # min_interval (ms)
    "10",           # hop_size (ms)
    "500",          # max_sil_kept (ms)
    "0.9",          # _max (normalize max)
    "0.25",         # alpha
    "0",            # i_part (partition index)
    "1",            # all_part (total partitions)
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd=GPT_SOVITS_DIR, env=ENV)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:1000])

# Count sliced files
sliced_files = [f for f in os.listdir(SLICED_DIR) if f.endswith('.wav')]
print(f"\nSliced into {len(sliced_files)} segments")

# Step 2: ASR transcription with faster-whisper
print("\n" + "=" * 60)
print("STEP 2: ASR transcription (faster-whisper, English)")
print("=" * 60)

cmd = [
    PYTHON_EXE,
    os.path.join(GPT_SOVITS_DIR, "tools", "asr", "fasterwhisper_asr.py"),
    "-i", SLICED_DIR,
    "-o", ASR_OUTPUT_DIR,
    "-s", "large-v3",
    "-l", "en",        # JFla sings in English
    "-p", "float16",
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd=GPT_SOVITS_DIR, env=ENV)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:1000])

# Check output
list_files = [f for f in os.listdir(ASR_OUTPUT_DIR) if f.endswith('.list')]
print(f"\nASR output files: {list_files}")
for lf in list_files:
    path = os.path.join(ASR_OUTPUT_DIR, lf)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  {lf}: {len(lines)} transcribed segments")
    for line in lines[:5]:
        print(f"    {line.strip()}")
    if len(lines) > 5:
        print(f"    ... ({len(lines) - 5} more)")
