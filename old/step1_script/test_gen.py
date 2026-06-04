import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_NAME, SCRIPT_DIR, PROMPT_TEMPLATE, TARGET_LENGTH
import ollama
from datetime import datetime

topic = "시편 23편과 함께하는 새벽 lofi"
prompt = PROMPT_TEMPLATE.format(topic=topic, target_length=TARGET_LENGTH)

print("=" * 50)
print(f"  emeth 대본 생성 테스트")
print(f"  주제: {topic}")
print("=" * 50)
print()

full = ""
for chunk in ollama.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], stream=True):
    c = chunk["message"]["content"]
    print(c, end="", flush=True)
    full += c

# 저장
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fpath = os.path.join(SCRIPT_DIR, f"emeth_test_{ts}.txt")
with open(fpath, "w", encoding="utf-8") as f:
    f.write(f"# emeth 대본\n# 주제: {topic}\n\n{full}")
print(f"\n\n[저장 완료] {fpath}")
