"""Run GPT-SoVITS fine-tuning (SoVITS + GPT training)"""
import os
import sys
import json
import yaml
import subprocess

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")
EXP_NAME = "jfla"
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)
VERSION = "v2"

# Pretrained models
PRETRAINED_S2G = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
PRETRAINED_S2D = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2D2333k.pth")
PRETRAINED_S1 = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")

step = sys.argv[1] if len(sys.argv) > 1 else "sovits"

if step in ["sovits", "both"]:
    print("=" * 60)
    print("TRAINING: SoVITS (s2_train.py)")
    print("=" * 60)

    # Load and modify s2.json config
    config_path = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "configs", "s2.json")
    with open(config_path) as f:
        data = json.load(f)

    s2_dir = EXP_DIR
    os.makedirs(os.path.join(s2_dir, f"logs_s2_{VERSION}"), exist_ok=True)

    # Training params - conservative for small dataset
    data["train"]["batch_size"] = 8       # Small batch for 66 samples
    data["train"]["epochs"] = 20          # Enough epochs for fine-tuning
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = PRETRAINED_S2G
    data["train"]["pretrained_s2D"] = PRETRAINED_S2D
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = 5
    data["train"]["gpu_numbers"] = "0"
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 0
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = s2_dir
    data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = "SoVITS_weights_v2"
    data["name"] = EXP_NAME
    data["version"] = VERSION

    tmp_config = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s2.json")
    os.makedirs(os.path.dirname(tmp_config), exist_ok=True)
    with open(tmp_config, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Config saved to: {tmp_config}")
    print(f"Batch size: {data['train']['batch_size']}")
    print(f"Epochs: {data['train']['epochs']}")
    print(f"Starting SoVITS training...")

    env = os.environ.copy()
    gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
    env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"

    result = subprocess.run(
        [PYTHON_EXE, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_config],
        cwd=GPT_SOVITS_DIR, env=env, timeout=3600
    )
    print(f"SoVITS training exit code: {result.returncode}")

if step in ["gpt", "both"]:
    print("\n" + "=" * 60)
    print("TRAINING: GPT (s1_train.py)")
    print("=" * 60)

    # Load and modify s1longer-v2.yaml config
    config_path = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "configs", "s1longer-v2.yaml")
    with open(config_path) as f:
        data = yaml.safe_load(f)

    s1_dir = EXP_DIR
    os.makedirs(os.path.join(s1_dir, f"logs_s1_{VERSION}"), exist_ok=True)

    # Training params
    data["train"]["batch_size"] = 8
    data["train"]["epochs"] = 20
    data["train"]["save_every_n_epoch"] = 5
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = "GPT_weights_v2"
    data["train"]["exp_name"] = EXP_NAME
    data["pretrained_s1"] = PRETRAINED_S1
    data["train_semantic_path"] = os.path.join(s1_dir, "6-name2semantic.tsv")
    data["train_phoneme_path"] = os.path.join(s1_dir, "2-name2text.txt")
    data["output_dir"] = os.path.join(s1_dir, f"logs_s1_{VERSION}")

    tmp_config = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s1.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Config saved to: {tmp_config}")
    print(f"Starting GPT training...")

    env = os.environ.copy()
    gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
    env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["hz"] = "25hz"

    result = subprocess.run(
        [PYTHON_EXE, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_config],
        cwd=GPT_SOVITS_DIR, env=env, timeout=3600
    )
    print(f"GPT training exit code: {result.returncode}")

print("\n" + "=" * 60)
print("DONE! Checking for output weights...")
print("=" * 60)

# Check for output weights
for wdir in ["SoVITS_weights_v2", "GPT_weights_v2"]:
    full_path = os.path.join(GPT_SOVITS_DIR, wdir)
    if os.path.exists(full_path):
        files = os.listdir(full_path)
        jfla_files = [f for f in files if "jfla" in f.lower()]
        print(f"\n{wdir}/:")
        for f in jfla_files:
            size_mb = os.path.getsize(os.path.join(full_path, f)) / (1024*1024)
            print(f"  {f} ({size_mb:.1f} MB)")
    else:
        print(f"\n{wdir}/: not found")
