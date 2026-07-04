"""Create GPT training config for sion_mixed"""
import os
import yaml

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
EXP_NAME = "sion_mixed"
EXP_DIR = os.path.join(GPT_SOVITS_DIR, "logs", EXP_NAME)

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

tmp_s1 = os.path.join(GPT_SOVITS_DIR, "TEMP", "tmp_s1_mixed.yaml")
with open(tmp_s1, 'w') as f:
    yaml.dump(s1_config, f, default_flow_style=False)

print(f"GPT config saved: {tmp_s1}")
