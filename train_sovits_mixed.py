"""Run SoVITS training directly for sion_mixed"""
import os
import sys
import json

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, gpt_sovits_pkg)
os.chdir(GPT_SOVITS_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

EXP_NAME = "sion_mixed"
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)

# Create config
s2_config_path = os.path.join(gpt_sovits_pkg, "configs", "s2.json")
with open(s2_config_path, 'r') as f:
    s2_config = json.load(f)

s2_config["train"]["batch_size"] = 8
s2_config["train"]["epochs"] = 20
s2_config["train"]["save_every_epoch"] = 5
s2_config["train"]["if_save_latest"] = True
s2_config["train"]["if_save_every_weights"] = True
s2_config["train"]["half_weights_save_dir"] = os.path.join(GPT_SOVITS_DIR, "SoVITS_weights_v2")
s2_config["train"]["exp_dir"] = EXP_DIR
s2_config["train"]["gpu_numbers"] = "0"
s2_config["train"]["pretrained_s2G"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
s2_config["train"]["pretrained_s2D"] = os.path.join(gpt_sovits_pkg, "pretrained_models", "gsv-v2final-pretrained", "s2D2333k.pth")
s2_config["train"]["text_low_lr_rate"] = 0.4
s2_config["train"]["lora_rank"] = 0
s2_config["data"]["exp_dir"] = EXP_DIR
s2_config["model"]["version"] = "v2"
s2_config["s2_ckpt_dir"] = os.path.join(EXP_DIR, "logs_s2_v2")
s2_config["name"] = EXP_NAME

tmp_config = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s2_mixed.json")
os.makedirs(os.path.dirname(tmp_config), exist_ok=True)
with open(tmp_config, 'w') as f:
    json.dump(s2_config, f, indent=2)

print(f"Config saved: {tmp_config}")
print(f"Starting SoVITS training: batch_size=8, epochs=20")

# Set sys.argv for s2_train.py
sys.argv = ["s2_train.py", "--config", tmp_config]

# Import and run training
from GPT_SoVITS.s2_train import run as s2_run
# Note: s2_train.py reads sys.argv and calls run() at module level
