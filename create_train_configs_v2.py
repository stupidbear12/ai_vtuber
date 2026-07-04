"""Create SoVITS and GPT training configs for sion_jfla_v2"""
import os
import json
import yaml

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
EXP_NAME = "sion_jfla_v2"
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)
os.makedirs(os.path.join(GPT_SOVITS_DIR, "TEMP"), exist_ok=True)

# === SoVITS config ===
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

tmp_s2 = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s2_jfla_v2.json")
with open(tmp_s2, 'w') as f:
    json.dump(s2_config, f, indent=2)
print(f"SoVITS config: {tmp_s2}")

# === GPT config ===
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
s1_config["train"]["if_save_latest"] = True
s1_config["train"]["if_save_every_weights"] = True

tmp_s1 = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s1_jfla_v2.yaml")
with open(tmp_s1, 'w') as f:
    yaml.dump(s1_config, f, default_flow_style=False)
print(f"GPT config: {tmp_s1}")
print("Configs ready!")
