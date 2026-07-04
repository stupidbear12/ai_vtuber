import os, sys, torch, shutil
from collections import OrderedDict
from time import time as ttime

G_PATH = r"C:\Users\thtgg\workspace2\GPT-SoVITS\logs\sion_jfla_v3\logs_s2_v2\G_233333333333.pth"
CONFIG_PATH = r"C:\Users\thtgg\workspace2\GPT-SoVITS\logs\sion_jfla_v3\logs_s2_v2\config.json"
SAVE_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS\SoVITS_weights_v2"

import json
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

print("Loading full checkpoint...")
ckpt = torch.load(G_PATH, map_location="cpu")
model_state = ckpt.get("model", ckpt)

opt = OrderedDict()
opt["weight"] = {}
for key in model_state.keys():
    if "enc_q" in key:
        continue
    opt["weight"][key] = model_state[key].half()
opt["config"] = config
opt["info"] = "20epoch"

save_path = os.path.join(SAVE_DIR, "sion_jfla_v3_e20.pth")
tmp_path = "%s.pth" % ttime()
torch.save(opt, tmp_path)
shutil.move(tmp_path, save_path)

size_mb = os.path.getsize(save_path) / (1024*1024)
print(f"Saved: {save_path} ({size_mb:.1f} MB)")
print(f"Keys: {len(opt['weight'])}")
