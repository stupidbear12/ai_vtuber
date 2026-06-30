# -*- coding: utf-8 -*-
"""
sion_train.jsonl + koalpaca_sion.jsonl → sion_combined.jsonl
랜덤 셔플 포함
"""
import json, os, random

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training")

files = [
    ("sion_train.jsonl",    "기존 시온 학습"),
    ("koalpaca_sion.jsonl", "KoAlpaca 변환"),
]
output = os.path.join(TRAINING_DIR, "sion_combined.jsonl")

all_data = []
for fname, label in files:
    path = os.path.join(TRAINING_DIR, fname)
    before = len(all_data)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append(json.loads(line))
    print(f"  {label}: {len(all_data) - before}건 로드")

random.seed(42)
random.shuffle(all_data)
print(f"\n총 {len(all_data)}건 → 셔플 완료")

with open(output, "w", encoding="utf-8") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"저장: {output}")
