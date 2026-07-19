# -*- coding: utf-8 -*-
"""
v3 데이터 + 시온 멀티턴 데이터 병합 → v4
멀티턴 데이터를 3배 반복해서 가중치 부여
"""
import json
import os
import random

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    v3_train = os.path.join(TRAINING_DIR, "sion_combined_v3.jsonl")
    v3_eval = os.path.join(TRAINING_DIR, "sion_eval_v3.jsonl")
    multiturn = os.path.join(TRAINING_DIR, "sion_multiturn.jsonl")

    train_data = load_jsonl(v3_train)
    eval_data = load_jsonl(v3_eval)
    mt_data = load_jsonl(multiturn)

    print(f"v3 학습: {len(train_data)}건")
    print(f"v3 평가: {len(eval_data)}건")
    print(f"멀티턴: {len(mt_data)}건")

    # 멀티턴 데이터를 5배 반복 (강한 가중치)
    mt_repeated = mt_data * 5
    print(f"멀티턴 5x 반복: {len(mt_repeated)}건")

    # 병합
    combined = train_data + mt_repeated
    random.seed(42)
    random.shuffle(combined)

    # 멀티턴에서 일부를 eval에도 추가
    mt_eval = mt_data[:10]  # 10건
    eval_combined = eval_data + mt_eval

    # 저장
    out_train = os.path.join(TRAINING_DIR, "sion_combined_v4.jsonl")
    out_eval = os.path.join(TRAINING_DIR, "sion_eval_v4.jsonl")

    with open(out_train, "w", encoding="utf-8") as f:
        for entry in combined:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(out_eval, "w", encoding="utf-8") as f:
        for entry in eval_combined:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n최종:")
    print(f"  학습: {len(combined)}건 → {out_train}")
    print(f"  평가: {len(eval_combined)}건 → {out_eval}")

if __name__ == "__main__":
    main()
