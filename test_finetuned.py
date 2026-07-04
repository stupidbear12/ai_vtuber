"""Test fine-tuned GPT-SoVITS model with Korean TTS"""
import os
import sys
import time
import subprocess
import requests
import json

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_DIR, ".venv", "Scripts", "python.exe")

# Fine-tuned model paths
GPT_MODEL = os.path.join(GPT_SOVITS_DIR, "GPT_weights_v2", "jfla-e5.ckpt")
SOVITS_MODEL = os.path.join(GPT_SOVITS_DIR, "SoVITS_weights_v2", "jfla_e20_s420.pth")
REF_AUDIO = os.path.join(GPT_SOVITS_DIR, "jfla.wav")

# Verify files exist
for path, name in [(GPT_MODEL, "GPT"), (SOVITS_MODEL, "SoVITS"), (REF_AUDIO, "Reference")]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  {name}: {os.path.basename(path)} ({size_mb:.1f} MB)")
    else:
        print(f"  {name}: NOT FOUND - {path}")
        sys.exit(1)

# Update tts_infer.yaml to use fine-tuned models
config_path = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "configs", "tts_infer.yaml")
import yaml

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config['custom']['t2s_weights_path'] = GPT_MODEL
config['custom']['vits_weights_path'] = SOVITS_MODEL
config['custom']['version'] = 'v2'
config['custom']['device'] = 'cuda'
config['custom']['is_half'] = True

with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"\nUpdated {config_path}")
print(f"  GPT: {GPT_MODEL}")
print(f"  SoVITS: {SOVITS_MODEL}")

# Start the API server
print("\nStarting GPT-SoVITS API server...")
env = os.environ.copy()
gpt_sovits_pkg = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS")
env["PYTHONPATH"] = f"{GPT_SOVITS_DIR};{gpt_sovits_pkg}"

server_proc = subprocess.Popen(
    [PYTHON_EXE, "api_v2.py", "-a", "127.0.0.1", "-p", "9880"],
    cwd=GPT_SOVITS_DIR,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
print(f"  Server PID: {server_proc.pid}")

# Wait for server to be ready
print("  Waiting for server to start...")
for i in range(60):
    time.sleep(2)
    try:
        r = requests.get("http://127.0.0.1:9880/", timeout=2)
        if r.status_code == 200:
            print(f"  Server ready after {(i+1)*2}s")
            break
    except:
        pass
else:
    print("  Server failed to start within 120s!")
    server_proc.kill()
    stderr = server_proc.stderr.read().decode()
    print(f"  STDERR: {stderr[-1000:]}")
    sys.exit(1)

# Test sentences
test_cases = [
    # Korean
    ("안녕하세요, 저는 시온이에요. 반가워요!", "ko", "sion_kr_01"),
    ("오늘 방송 재미있게 할게요, 많이 응원해주세요!", "ko", "sion_kr_02"),
    ("와 진짜요? 대박이다!", "ko", "sion_kr_03"),
    ("음, 잠깐만요. 생각 좀 해볼게요.", "ko", "sion_kr_04"),
    ("여러분 안녕하세요! 오늘도 즐거운 하루 보내세요!", "ko", "sion_kr_05"),
    # English
    ("Hello everyone! I'm Sion, nice to meet you!", "en", "sion_en_01"),
    ("Today's stream is going to be really fun, please support me!", "en", "sion_en_02"),
]

output_dir = os.path.join(GPT_SOVITS_DIR, "test_output_finetuned")
os.makedirs(output_dir, exist_ok=True)

for text, lang, filename in test_cases:
    print(f"\nGenerating: [{lang}] {text}")
    params = {
        "text": text,
        "text_lang": lang,
        "ref_audio_path": REF_AUDIO,
        "prompt_lang": "en",
        "prompt_text": "I'm in love with the shape of you",
    }

    try:
        r = requests.get("http://127.0.0.1:9880/tts", params=params, timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            output_path = os.path.join(output_dir, f"{filename}.wav")
            with open(output_path, 'wb') as f:
                f.write(r.content)
            size_kb = len(r.content) / 1024
            print(f"  Saved: {filename}.wav ({size_kb:.0f} KB)")
        else:
            print(f"  Error: status={r.status_code}, size={len(r.content)}")
            if r.status_code != 200:
                print(f"  Response: {r.text[:500]}")
    except Exception as e:
        print(f"  Error: {e}")

# Summary
print(f"\n{'='*60}")
print(f"Test results in: {output_dir}")
files = os.listdir(output_dir)
total_size = 0
for f in sorted(files):
    fpath = os.path.join(output_dir, f)
    size_kb = os.path.getsize(fpath) / 1024
    total_size += size_kb
    print(f"  {f} ({size_kb:.0f} KB)")
print(f"Total: {len(files)} files, {total_size:.0f} KB")

# Kill server
server_proc.terminate()
print("\nServer stopped.")
