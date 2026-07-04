"""Extract half-precision GPT weights from PyTorch Lightning checkpoint"""
import os
import sys
import torch
import yaml
from collections import OrderedDict

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS"))

ckpt_path = os.path.join(GPT_SOVITS_DIR, "logs", "jfla", "logs_s1_v2", "ckpt", "epoch=4-step=15.ckpt")
config_path = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s1.yaml")
output_dir = os.path.join(GPT_SOVITS_DIR, "GPT_weights_v2")
os.makedirs(output_dir, exist_ok=True)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

with open(config_path) as f:
    config = yaml.safe_load(f)

to_save = OrderedDict()
to_save["weight"] = OrderedDict()

# Extract model state dict
state_dict = ckpt.get("state_dict", {})
for key in state_dict:
    to_save["weight"][key] = state_dict[key].half()

to_save["config"] = config
to_save["info"] = "GPT-e5"

output_path = os.path.join(output_dir, "jfla-e5.ckpt")
print(f"Saving to: {output_path}")
torch.save(to_save, output_path)

size_mb = os.path.getsize(output_path) / (1024*1024)
print(f"Saved: {size_mb:.1f} MB")
print("Done!")
