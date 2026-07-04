"""Run faster-whisper ASR on sliced JFla vocals"""
import os
import sys

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
sys.path.insert(0, GPT_SOVITS_DIR)
os.chdir(GPT_SOVITS_DIR)

SLICED_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_sliced")
ASR_OUTPUT_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_asr")
os.makedirs(ASR_OUTPUT_DIR, exist_ok=True)

from tools.asr.fasterwhisper_asr import execute_asr, download_model

model_size = "base"
model_path = download_model(model_size)
print(f"Model path: {model_path}")

output_file = execute_asr(
    input_folder=SLICED_DIR,
    output_folder=ASR_OUTPUT_DIR,
    model_path=model_path,
    language="en",
    precision="float16",
)
print(f"ASR output: {output_file}")

# Show sample
with open(output_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f"\n{len(lines)} transcribed segments")
for line in lines[:10]:
    print(f"  {line.strip()}")
