"""Train SoVITS + GPT on mixed JFla+COOMO dataset"""
import os
import subprocess
import json
import sys
import yaml

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")

EXP_NAME = "sion_mixed"
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)

env = os.environ.copy()
env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"
env["PYTHONIOENCODING"] = "utf-8"

# ===================== SoVITS Training =====================
print("=" * 60)
print("SoVITS Training")
print("=" * 60)

s2_config_path = os.path.join(gpt_sovits_pkg, "configs", "s2.json")
with open(s2_config_path, 'r') as f:
    s2_config = json.load(f)

# Update config for our experiment
s2_config["train"]["batch_size"] = 8
s2_config["train"]["epochs"] = 20
s2_config["train"]["save_every_epoch"] = 5
s2_config["train"]["if_save_latest"] = True
s2_config["train"]["if_save_every_weights"] = True
s2_config["train"]["half_weights_save_dir"] = os.path.join(GPT_SOVITS_DIR, "SoVITS_weights_v2")
s2_config["train"]["exp_dir"] = EXP_DIR
s2_config["data"]["exp_dir"] = EXP_DIR
s2_config["model"]["version"] = "v2"
s2_config["s2_ckpt_dir"] = os.path.join(EXP_DIR, "logs_s2_v2")
s2_config["name"] = EXP_NAME
s2_config["train"]["pretrained_s2G"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
s2_config["train"]["pretrained_s2D"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s2D2333k.pth")
s2_config["train"]["text_low_lr_rate"] = 0.4
s2_config["train"]["lora_rank"] = 0
s2_config["train"]["gpu_numbers"] = "0"

tmp_s2 = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s2_mixed.json")
os.makedirs(os.path.dirname(tmp_s2), exist_ok=True)
with open(tmp_s2, 'w') as f:
    json.dump(s2_config, f, indent=2)

print(f"Config: {tmp_s2}")
print(f"Batch size: 8, Epochs: 20")

s2_cmd = [PYTHON_EXE, os.path.join(gpt_sovits_pkg, "s2_train.py"), "--config", tmp_s2]
result = subprocess.run(s2_cmd, cwd=GPT_SOVITS_DIR, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600)
if result.returncode != 0:
    print(f"SoVITS FAILED: {(result.stderr or '')[-500:]}")
    sys.exit(1)

# Check SoVITS weights
sovits_dir = os.path.join(GPT_SOVITS_DIR, "SoVITS_weights_v2")
sovits_files = [f for f in os.listdir(sovits_dir) if f.startswith(EXP_NAME)]
print(f"SoVITS weights: {sovits_files}")

# ===================== GPT Training =====================
print("\n" + "=" * 60)
print("GPT Training")
print("=" * 60)

s1_config_path = os.path.join(gpt_sovits_pkg, "configs", "s1longer-v2.yaml")
with open(s1_config_path, 'r') as f:
    s1_config = yaml.safe_load(f)

s1_config["train_semantic_path"] = os.path.join(EXP_DIR, "6-name2semantic.tsv")
s1_config["train_phoneme_path"] = os.path.join(EXP_DIR, "2-name2text.txt")
s1_config["output_dir"] = os.path.join(EXP_DIR, "logs_s1_v2")
s1_config["pretrained_s1"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
s1_config["data"]["max_eval_sample"] = 8
s1_config["data"]["max_sec"] = 54
s1_config["data"]["batch_size"] = 8
s1_config["data"]["num_workers"] = 1
s1_config["train"]["epochs"] = 20
s1_config["train"]["save_every_n_epoch"] = 5
s1_config["train"]["half_weights_save_dir"] = os.path.join(GPT_SOVITS_DIR, "GPT_weights_v2")
s1_config["train"]["exp_name"] = EXP_NAME
s1_config["train"]["precision"] = "16-mixed"

tmp_s1 = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s1_mixed.yaml")
with open(tmp_s1, 'w') as f:
    yaml.dump(s1_config, f, default_flow_style=False)

print(f"Config: {tmp_s1}")
print(f"Batch size: 8, Epochs: 20")

s1_cmd = [PYTHON_EXE, os.path.join(gpt_sovits_pkg, "s1_train.py"), "--config_file", tmp_s1]
result = subprocess.run(s1_cmd, cwd=GPT_SOVITS_DIR, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600)
if result.returncode != 0:
    print(f"GPT FAILED: {(result.stderr or '')[-500:]}")
    # Even if GPT training fails to save weights automatically, check for checkpoints
    print("Checking for GPT checkpoints...")

# Check GPT weights
gpt_dir = os.path.join(GPT_SOVITS_DIR, "GPT_weights_v2")
gpt_files = [f for f in os.listdir(gpt_dir) if f.startswith(EXP_NAME)]
print(f"GPT weights: {gpt_files}")

# If no GPT weights, try to extract from checkpoint
if not gpt_files:
    ckpt_dir = os.path.join(EXP_DIR, "logs_s1_v2", "ckpt")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if ckpts:
            print(f"Found checkpoints: {ckpts}")
            print("Extracting weights manually...")
            import torch
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", {})
            config = yaml.safe_load(open(tmp_s1, 'r'))
            to_save = {"weight": {}, "config": config, "info": f"GPT-{EXP_NAME}"}
            for key in state_dict:
                to_save["weight"][key] = state_dict[key].half()
            output_path = os.path.join(gpt_dir, f"{EXP_NAME}-e20.ckpt")
            torch.save(to_save, output_path)
            sz = os.path.getsize(output_path) / (1024*1024)
            print(f"Saved: {output_path} ({sz:.1f} MB)")

print("\n" + "=" * 60)
print("Training complete!")
