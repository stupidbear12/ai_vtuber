# -*- coding: utf-8 -*-
"""
prepare_data.py — KIT-19 데이터를 시온 LoRA 학습 포맷으로 변환

데이터 소스: snunlp/KIT-19-ToolKit-100000 (HuggingFace, Apache 2.0)
출력 포맷: Llama 3 chat template (instruction → response)

사용법:
    python prepare_data.py
    python prepare_data.py --output_dir ./data --max_samples 50000
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


# KIT-19 태스크 중 한국어 생성 능력에 도움이 되는 태스크만 선별
# (혐오표현 감지 같은 분류 태스크는 구어체 학습에 덜 유용)
PRIORITY_TASKS = {
    # 높은 우선순위: 한국어 생성/이해에 직접 도움
    "text_generation",
    "summarization",
    "question_answering",
    "paraphrase_detection",
    "sentence_completion",
    "dialogue",
    "translation",
    "grammar_correction",
    # 중간 우선순위: 간접적 도움
    "natural_language_inference",
    "sentiment_analysis",
    "topic_classification",
}

# 제외할 태스크 (구어체 학습에 비효과적)
EXCLUDE_TASKS = {
    "hatespeech_detection",  # 혐오표현 분류는 시온 학습에 부적합
}

# 시온 캐릭터 시스템 프롬프트
SION_SYSTEM_PROMPT = (
    '너는 "시온(sion)"이라는 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.\n'
    "\n"
    "캐릭터 설명\n"
    "- 20대 초반 여성, 항상 반말. 존댓말 절대 금지\n"
    '- 밝고 에너지 넘치며, 음악을 좋아하는 DJ\n'
    '- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐", "대박")\n'
    "\n"
    "규칙\n"
    "- 응답 맨 앞에 반드시 [감정:태그] 붙여. "
    "태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy\n"
    "- 1~2문장으로 짧게 답해\n"
    '- 모르는 건 절대 지어내지 마. "잘 모르겠는데?" 라고 솔직하게 답해\n'
    "- 실제로 하지 않은 행동을 말하지 마"
)


def download_kit19(cache_dir: str = "./data/raw") -> list[dict]:
    """KIT-19 데이터셋을 HuggingFace에서 다운로드한다.

    snunlp/KIT-19-ToolKit-100000 데이터셋의 CSV 파일을 직접 다운로드하여 파싱.
    datasets 라이브러리의 자동 파싱에 타입 오류가 있어 수동 처리.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "kit19_raw.json")

    # 캐시가 있으면 재사용
    if os.path.exists(cache_path):
        print(f"[캐시 사용] {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[다운로드] snunlp/KIT-19-ToolKit-100000 ...")

    # CSV 파일 목록 조회 및 다운로드
    from huggingface_hub import list_repo_files

    files = list_repo_files("snunlp/KIT-19-ToolKit-100000", repo_type="dataset")
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        # CSV가 없으면 datasets 라이브러리로 시도 (타입 에러 가능)
        print("[대체] datasets 라이브러리로 로드 시도...")
        ds = load_dataset(
            "snunlp/KIT-19-ToolKit-100000",
            split="train",
            trust_remote_code=True,
        )
        samples = [dict(row) for row in ds]
    else:
        import csv

        samples = []
        for csv_file in csv_files:
            local = hf_hub_download(
                "snunlp/KIT-19-ToolKit-100000",
                csv_file,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            with open(local, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(dict(row))

    print(f"[완료] 총 {len(samples):,}개 샘플 다운로드")

    # 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return samples


def filter_samples(samples: list[dict]) -> list[dict]:
    """학습에 유용한 샘플만 필터링한다."""
    filtered = []
    task_counts = {}

    for s in samples:
        task = s.get("task", "unknown")

        # 제외 태스크 스킵
        if task in EXCLUDE_TASKS:
            continue

        # 입력/출력이 비어있으면 스킵
        inp = s.get("input", "").strip()
        out = s.get("output", "").strip()
        if not inp or not out:
            continue

        # 너무 짧은 출력 스킵 (3자 미만)
        if len(out) < 3:
            continue

        filtered.append(s)
        task_counts[task] = task_counts.get(task, 0) + 1

    print(f"\n[필터링] {len(samples):,} → {len(filtered):,} 샘플")
    print("[태스크별 분포]")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        priority = "★" if task in PRIORITY_TASKS else " "
        print(f"  {priority} {task}: {count:,}")

    return filtered


def convert_to_chat_format(
    samples: list[dict],
    include_system: bool = False,
    max_samples: int = 0,
) -> list[dict]:
    """KIT-19 샘플을 Llama 3 chat 학습 포맷으로 변환한다.

    출력 포맷 (ShareGPT 스타일):
    {
        "conversations": [
            {"from": "system", "value": "..."},  # include_system=True 시
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    converted = []

    for s in samples:
        task = s.get("task", "")
        instruction = s.get("instruction", "").strip()
        inp = s.get("input", "").strip()
        out = s.get("output", "").strip()

        # 사용자 메시지 구성
        if instruction and inp:
            user_msg = f"{instruction}\n\n{inp}"
        elif inp:
            user_msg = inp
        elif instruction:
            user_msg = instruction
        else:
            continue

        # 대화 구성
        conversations = []
        if include_system:
            conversations.append({"from": "system", "value": SION_SYSTEM_PROMPT})
        conversations.append({"from": "human", "value": user_msg})
        conversations.append({"from": "gpt", "value": out})

        converted.append({"conversations": conversations, "task": task})

    # 셔플 후 제한
    random.shuffle(converted)
    if max_samples > 0:
        converted = converted[:max_samples]

    print(f"\n[변환 완료] {len(converted):,}개 학습 샘플 생성")
    return converted


def save_dataset(data: list[dict], output_path: str) -> None:
    """학습 데이터를 JSON 파일로 저장한다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[저장] {output_path} ({size_mb:.1f} MB, {len(data):,} 샘플)")


def main():
    parser = argparse.ArgumentParser(description="KIT-19 → 시온 LoRA 학습 데이터 변환")
    parser.add_argument(
        "--output_dir", default="./data", help="출력 디렉토리 (기본: ./data)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="최대 샘플 수 (0=전체, 기본: 0)",
    )
    parser.add_argument(
        "--include_system",
        action="store_true",
        help="시온 시스템 프롬프트 포함 (기본: False)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="랜덤 시드 (기본: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 다운로드
    raw_samples = download_kit19(os.path.join(args.output_dir, "raw"))

    # 2. 필터링
    filtered = filter_samples(raw_samples)

    # 3. 학습 포맷으로 변환
    train_data = convert_to_chat_format(
        filtered,
        include_system=args.include_system,
        max_samples=args.max_samples,
    )

    # 4. Train/Eval 분할 (95:5)
    split_idx = int(len(train_data) * 0.95)
    train_split = train_data[:split_idx]
    eval_split = train_data[split_idx:]

    # 5. 저장
    save_dataset(train_split, os.path.join(args.output_dir, "train.json"))
    save_dataset(eval_split, os.path.join(args.output_dir, "eval.json"))

    print(f"\n학습 데이터 준비 완료!")
    print(f"  Train: {len(train_split):,}")
    print(f"  Eval:  {len(eval_split):,}")


if __name__ == "__main__":
    main()
