"""GPT-SoVITS fine-tuning pipeline for JFla voice"""
import os
import sys
import subprocess
import json
import shutil

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")

# Experiment config
EXP_NAME = "jfla"
ASR_LIST = os.path.join(GPT_SOVITS_DIR, "jfla_asr", "jfla_sliced.list")
WAV_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_sliced")
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)
BERT_DIR = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large")
HUBERT_DIR = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
PRETRAINED_S2G = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
PRETRAINED_S1 = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
PRETRAINED_S2D = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2D2333k.pth")

os.makedirs(EXP_DIR, exist_ok=True)

def run_step(step_name, script, env_vars, timeout=600):
    """Run a preprocessing/training step"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")

    env = os.environ.copy()
    gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
    env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"
    env.update(env_vars)

    cmd = [PYTHON_EXE, "-s", script]
    result = subprocess.run(cmd, cwd=GPT_SOVITS_DIR, env=env,
                          capture_output=True, text=True, timeout=timeout)

    if result.stdout:
        # Print last 20 lines of stdout
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(f"  {line}")

    if result.returncode != 0 and result.stderr:
        print(f"  STDERR: {result.stderr[-500:]}")
        return False

    print(f"  -> Done!")
    return True

def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Filter ASR list - remove entries with empty transcriptions
    print("Filtering ASR list...")
    with open(ASR_LIST, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 4 and parts[3].strip():
            filtered.append(line.strip())

    filtered_list = os.path.join(EXP_DIR, "asr_filtered.list")
    with open(filtered_list, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered) + '\n')
    print(f"  Filtered: {len(filtered)}/{len(lines)} segments with text")

    base_env = {
        "inp_text": filtered_list,
        "inp_wav_dir": WAV_DIR,
        "exp_name": EXP_NAME,
        "opt_dir": EXP_DIR,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
    }

    if step in ["all", "1a"]:
        # Step 1a: Get text (phonemes + BERT features)
        env = {**base_env, "bert_pretrained_dir": BERT_DIR}
        ok = run_step("1a - Get Text/Phonemes",
                      "GPT_SoVITS/prepare_datasets/1-get-text.py", env, timeout=300)
        if not ok and step != "all":
            return

        # Merge outputs
        txt_path = os.path.join(EXP_DIR, "2-name2text-0.txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf8') as f:
                content = f.read()
            merged = os.path.join(EXP_DIR, "2-name2text.txt")
            with open(merged, 'w', encoding='utf8') as f:
                f.write(content)
            os.remove(txt_path)
            lines = content.strip().split('\n')
            print(f"  Generated {len(lines)} phoneme entries")

    if step in ["all", "1b"]:
        # Step 1b: Get HuBERT features + resample to 32k
        env = {**base_env,
               "cnhubert_base_dir": HUBERT_DIR,
               "sv_path": "",
              }
        ok = run_step("1b - Get HuBERT + Resample",
                      "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", env, timeout=600)
        if not ok and step != "all":
            return

    if step in ["all", "1c"]:
        # Step 1c: Get semantic tokens
        env = {**base_env,
               "pretrained_s2G": PRETRAINED_S2G,
               "s2config_path": "GPT_SoVITS/configs/s2.json",
              }
        ok = run_step("1c - Get Semantic Tokens",
                      "GPT_SoVITS/prepare_datasets/3-get-semantic.py", env, timeout=600)
        if not ok and step != "all":
            return

        # Merge outputs
        sem_path = os.path.join(EXP_DIR, "6-name2semantic-0.tsv")
        if os.path.exists(sem_path):
            with open(sem_path, 'r', encoding='utf8') as f:
                content = f.read()
            merged = os.path.join(EXP_DIR, "6-name2semantic.tsv")
            with open(merged, 'w', encoding='utf8') as f:
                f.write("item_name\tsemantic_audio\n" + content)
            os.remove(sem_path)

    print(f"\n{'='*60}")
    print("Preprocessing complete! Ready for training.")
    print(f"Experiment directory: {EXP_DIR}")
    print(f"{'='*60}")

    # List what we have
    for item in sorted(os.listdir(EXP_DIR)):
        path = os.path.join(EXP_DIR, item)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  {item} ({size:,} bytes)")
        else:
            count = len(os.listdir(path)) if os.path.isdir(path) else 0
            print(f"  {item}/ ({count} files)")

if __name__ == "__main__":
    main()
