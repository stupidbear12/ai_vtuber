# -*- coding: utf-8 -*-
"""
기존 시온 학습 데이터 + AI Hub 전처리 데이터 병합

Usage:
  python merge_training_data.py
  python merge_training_data.py --base training/sion_combined_clean.jsonl

Output:
  training/sion_combined_train.jsonl  — 학습용
  training/sion_combined_eval.jsonl   — 평가용
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42
TRAINING_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_TRAIN = TRAINING_DIR / "sion_combined_clean.jsonl"
DEFAULT_BASE_EVAL = TRAINING_DIR / "sion_eval.jsonl"
DEFAULT_AIHUB_TRAIN = TRAINING_DIR / "sion_aihub_emotion.jsonl"
DEFAULT_AIHUB_EVAL = TRAINING_DIR / "sion_aihub_emotion_eval.jsonl"
OUT_TRAIN = TRAINING_DIR / "sion_combined_train.jsonl"
OUT_EVAL = TRAINING_DIR / "sion_combined_eval.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 오류: {e}") from e
    return rows


def validate_example(ex: dict, source: str, index: int) -> None:
    if "messages" not in ex or not isinstance(ex["messages"], list):
        raise ValueError(f"{source}[{index}] messages 필드가 없습니다.")
    roles = {m.get("role") for m in ex["messages"]}
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"{source}[{index}] user/assistant 턴이 없습니다.")


def dedupe_key(ex: dict) -> str:
    parts = []
    for m in ex.get("messages", []):
        if m.get("role") == "system":
            continue
        parts.append(f"{m.get('role')}:{m.get('content', '')}")
    return "\n".join(parts)


def merge_splits(
    base_train: Path,
    base_eval: Path,
    aihub_train: Path,
    aihub_eval: Path,
    out_train: Path,
    out_eval: Path,
    shuffle: bool = True,
    dedupe: bool = True,
) -> dict:
    random.seed(SEED)

    train_parts: list[tuple[str, list[dict]]] = []
    eval_parts: list[tuple[str, list[dict]]] = []

    if base_train.exists():
        train_parts.append(("base_train", load_jsonl(base_train)))
    if aihub_train.exists():
        train_parts.append(("aihub_train", load_jsonl(aihub_train)))
    if base_eval.exists():
        eval_parts.append(("base_eval", load_jsonl(base_eval)))
    if aihub_eval.exists():
        eval_parts.append(("aihub_eval", load_jsonl(aihub_eval)))

    if not train_parts:
        raise FileNotFoundError(
            "학습 데이터가 없습니다. "
            f"{base_train.name} 또는 {aihub_train.name} 중 하나는 필요합니다."
        )

    train_data: list[dict] = []
    eval_data: list[dict] = []
    counts: dict[str, int] = {}

    for name, rows in train_parts:
        for i, ex in enumerate(rows):
            validate_example(ex, name, i)
        train_data.extend(rows)
        counts[name] = len(rows)

    for name, rows in eval_parts:
        for i, ex in enumerate(rows):
            validate_example(ex, name, i)
        eval_data.extend(rows)
        counts[name + "_eval"] = len(rows)

    if dedupe:
        seen: set[str] = set()
        unique_train: list[dict] = []
        dup_train = 0
        for ex in train_data:
            key = dedupe_key(ex)
            if key in seen:
                dup_train += 1
                continue
            seen.add(key)
            unique_train.append(ex)
        train_data = unique_train

        unique_eval: list[dict] = []
        dup_eval = 0
        for ex in eval_data:
            key = dedupe_key(ex)
            if key in seen:
                dup_eval += 1
                continue
            seen.add(key)
            unique_eval.append(ex)
        eval_data = unique_eval
        counts["deduped_train"] = dup_train
        counts["deduped_eval"] = dup_eval

    if shuffle:
        random.shuffle(train_data)
        random.shuffle(eval_data)

    with out_train.open("w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with out_eval.open("w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts["train_total"] = len(train_data)
    counts["eval_total"] = len(eval_data)
    counts["out_train"] = str(out_train)
    counts["out_eval"] = str(out_eval)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="시온 학습 데이터 병합")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_TRAIN,
                        help="기존 학습 JSONL")
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL,
                        help="기존 평가 JSONL")
    parser.add_argument("--aihub", type=Path, default=DEFAULT_AIHUB_TRAIN,
                        help="AI Hub 학습 JSONL")
    parser.add_argument("--aihub-eval", type=Path, default=DEFAULT_AIHUB_EVAL,
                        help="AI Hub 평가 JSONL")
    parser.add_argument("--out-train", type=Path, default=OUT_TRAIN)
    parser.add_argument("--out-eval", type=Path, default=OUT_EVAL)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("시온 학습 데이터 병합 (기존 + AI Hub)")
    print("=" * 60)

    try:
        stats = merge_splits(
            base_train=args.base,
            base_eval=args.base_eval,
            aihub_train=args.aihub,
            aihub_eval=args.aihub_eval,
            out_train=args.out_train,
            out_eval=args.out_eval,
            shuffle=not args.no_shuffle,
            dedupe=not args.no_dedupe,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for key in ("base_train", "aihub_train", "base_eval", "aihub_eval"):
        if key in stats:
            print(f"  {key}: {stats[key]:,}건")
    if stats.get("deduped_train"):
        print(f"  중복 제거 (train): {stats['deduped_train']:,}건")
    if stats.get("deduped_eval"):
        print(f"  중복 제거 (eval): {stats['deduped_eval']:,}건")

    print()
    print(f"  학습 합계: {stats['train_total']:,}건 -> {stats['out_train']}")
    print(f"  평가 합계: {stats['eval_total']:,}건 -> {stats['out_eval']}")
    print()
    print("다음: python finetune_sion.py")


if __name__ == "__main__":
    main()
