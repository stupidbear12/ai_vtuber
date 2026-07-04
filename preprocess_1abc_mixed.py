"""Run GPT-SoVITS preprocessing steps 1a, 1b, 1c for mixed dataset"""
import os
import subprocess
import sys

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")

EXP_NAME = "sion_mixed"
ASR_LIST = os.path.join(GPT_SOVITS_DIR, "mixed_asr", "mixed_sliced_filtered.list")
SLICED_DIR = os.path.join(GPT_SOVITS_DIR, "mixed_sliced")
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)

# Pretrained model paths
BERT_DIR = os.path.join(gpt_sovits_pkg, "pretrained_models", "chinese-roberta-wwm-ext-large")
CNHUBERT_DIR = os.path.join(gpt_sovits_pkg, "pretrained_models", "chinese-hubert-base")
PRETRAINED_S2G = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
S2_CONFIG = os.path.join(gpt_sovits_pkg, "configs", "s2.json")

os.makedirs(EXP_DIR, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"
env["PYTHONIOENCODING"] = "utf-8"

def run_step(name, script, extra_env=None):
    print(f"\n{'='*60}")
    print(f"Step {name}")
    print(f"{'='*60}")
    e = env.copy()
    if extra_env:
        e.update(extra_env)
    result = subprocess.run(
        [PYTHON_EXE, os.path.join(gpt_sovits_pkg, script)],
        cwd=GPT_SOVITS_DIR, env=e,
        capture_output=True, text=True, timeout=600
    )
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        for l in lines[-10:]:
            print(f"  {l}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        return False
    return True

# Common env vars
common_env = {
    "inp_text": ASR_LIST,
    "inp_wav_dir": SLICED_DIR,
    "exp_name": EXP_NAME,
    "opt_dir": EXP_DIR,
    "bert_pretrained_dir": BERT_DIR,
    "cnhubert_base_dir": CNHUBERT_DIR,
    "pretrained_s2G": PRETRAINED_S2G,
    "s2config_path": S2_CONFIG,
    "is_half": "True",
    "i_part": "0",
    "all_parts": "1",
    "_CUDA_VISIBLE_DEVICES": "0",
}

# Step 1a: Phonemes + BERT
if not run_step("1a (Phonemes + BERT)", "prepare_datasets/1-get-text.py", common_env):
    print("Step 1a failed!")
    sys.exit(1)

# Check output
name2text = os.path.join(EXP_DIR, "2-name2text.txt")
if os.path.exists(name2text):
    with open(name2text, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  name2text entries: {len(lines)}")

# Step 1b: HuBERT + wav32k
if not run_step("1b (HuBERT + wav32k)", "prepare_datasets/2-get-hubert-wav32k.py", common_env):
    print("Step 1b failed!")
    sys.exit(1)

# Check outputs
for subdir in ["4-cnhubert", "5-wav32k"]:
    d = os.path.join(EXP_DIR, subdir)
    if os.path.exists(d):
        n = len(os.listdir(d))
        print(f"  {subdir}: {n} files")

# Step 1c: Semantic tokens
if not run_step("1c (Semantic tokens)", "prepare_datasets/3-get-semantic.py", common_env):
    print("Step 1c failed!")
    sys.exit(1)

# Check output
sem_file = os.path.join(EXP_DIR, "6-name2semantic.tsv")
if os.path.exists(sem_file):
    with open(sem_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  semantic entries: {len(lines)}")

print(f"\n{'='*60}")
print("All preprocessing steps complete!")
print(f"Experiment dir: {EXP_DIR}")
