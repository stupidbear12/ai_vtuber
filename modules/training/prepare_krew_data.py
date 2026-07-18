# -*- coding: utf-8 -*-
"""
prepare_krew_data.py — KREW korean-role-playing 데이터를 시온 LoRA 학습 포맷으로 변환

데이터 소스: huggingface-KREW/korean-role-playing (HuggingFace, Apache 2.0)
서브셋: general-roleplay-data (32.4K), exa-data (890), gf-persona-data (1.92K), youtube-couple-data (125)
출력 포맷: ShareGPT 형식 (시온 system prompt 포함)

사용법:
    python prepare_krew_data.py
    python prepare_krew_data.py --output_dir ./data --max_samples 20000
    python prepare_krew_data.py --subsets general-roleplay-data exa-data
"""

import argparse
import json
import os
import random
import re
from typing import Optional

from datasets import load_dataset


# ── 시온 캐릭터 시스템 프롬프트 (롤플레이 학습용) ─────────────────
SION_SYSTEM_PROMPT = (
    '너는 "시온(sion)"이라는 AI VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.\n'
    "\n"
    "캐릭터 설명\n"
    "- 20살 한국 여성. 항상 반말 사용\n"
    "- 활발하고 귀엽고, 애교가 많음. 가끔 츤데레\n"
    "- 시청자와 대화하며 노래도 부르는 방송인\n"
    '- 감탄사/의성어 자연스럽게 사용 (예: "헐~", "오오!", "ㅠㅠ", "흐흐", "대박", "엥?")\n'
    "\n"
    "말투 규칙\n"
    "- 반말로 친근하게 대화해\n"
    "- 짧고 자연스러운 구어체로 답해\n"
    '- 모르는 건 "잘 모르겠는데?" 라고 솔직하게 답해\n'
    "- 실제로 하지 않은 행동을 말하지 마"
)

# KREW 서브셋 목록
AVAILABLE_SUBSETS = [
    "general-roleplay-data",  # 32.4K — 일반 상황 기반 롤플레이
    "exa-data",               # 890   — 엑사 페르소나 대화 (시온으로 변환)
    "gf-persona-data",        # 1.92K — 연인 페르소나 대화 (친근한 대화 학습용)
    "youtube-couple-data",    # 125   — 유튜브 커플 실제 대화
]


def load_krew_subset(subset_name: str, cache_dir: str) -> list[dict]:
    """KREW 데이터셋의 특정 서브셋을 다운로드한다."""
    print(f"  [{subset_name}] 다운로드 중...")
    try:
        ds = load_dataset(
            "huggingface-KREW/korean-role-playing",
            subset_name,
            split="train",
            cache_dir=cache_dir,
        )
        samples = [dict(row) for row in ds]
        print(f"  [{subset_name}] {len(samples):,}개 샘플 로드 완료")
        return samples
    except Exception as e:
        print(f"  [{subset_name}] 로드 실패: {e}")
        return []


def clean_emotes(text: str) -> str:
    """*액션 표현* 을 제거한다. 시온은 액션 이모트를 사용하지 않음."""
    # *반갑게 웃으며* 같은 패턴 제거
    text = re.sub(r'\*[^*]+\*\s*', '', text)
    # 앞뒤 공백 정리
    text = text.strip()
    # 큰따옴표로 감싸진 대사만 남은 경우 따옴표 제거
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def replace_character_names(text: str) -> str:
    """엑사 캐릭터 이름을 시온으로 변환하고 {유저}를 시청자로 변환한다."""
    text = text.replace("엑사", "시온")
    text = text.replace("EXA", "시온")
    text = text.replace("{유저}", "시청자님")
    text = text.replace("{유저}", "시청자님")  # 다른 인코딩 대비
    return text


def convert_conversation(
    messages: list[dict],
    subset_name: str,
    clean_emote: bool = True,
) -> Optional[list[dict]]:
    """KREW 대화를 ShareGPT 형식으로 변환한다.

    입력: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    출력: [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
    """
    if not messages or len(messages) < 2:
        return None

    conversations = [{"from": "system", "value": SION_SYSTEM_PROMPT}]

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()

        if not content:
            continue

        # 캐릭터 이름 변환 (exa-data)
        if subset_name == "exa-data":
            content = replace_character_names(content)

        # 엑사 데이터의 emote 제거 (시온 스타일에 맞게)
        if clean_emote and role == "assistant":
            content = clean_emotes(content)
            if not content:
                continue

        # {유저} 플레이스홀더 정리 (모든 서브셋)
        content = content.replace("{유저}", "시청자님")

        if role == "user":
            conversations.append({"from": "human", "value": content})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": content})

    # 최소 system + human + gpt = 3턴 필요
    non_system = [c for c in conversations if c["from"] != "system"]
    if len(non_system) < 2:
        return None

    # human으로 시작하고 gpt로 끝나는지 확인
    if non_system[0]["from"] != "human":
        return None

    return conversations


def filter_quality(conversations: list[dict]) -> bool:
    """품질 필터: 너무 짧거나 문제 있는 대화 제거."""
    gpt_turns = [c for c in conversations if c["from"] == "gpt"]
    if not gpt_turns:
        return False

    # 모든 gpt 응답이 5자 미만이면 제외
    if all(len(c["value"]) < 5 for c in gpt_turns):
        return False

    # 총 텍스트 길이가 너무 짧으면 제외
    total_len = sum(len(c["value"]) for c in conversations if c["from"] != "system")
    if total_len < 20:
        return False

    return True


def process_krew_data(
    subsets: list[str],
    cache_dir: str,
    max_samples: int = 0,
    clean_emote: bool = True,
    seed: int = 42,
) -> list[dict]:
    """KREW 데이터를 다운로드하고 시온 학습용으로 변환한다."""
    random.seed(seed)
    all_data = []

    print("\n[1/3] KREW 데이터셋 다운로드")
    print("=" * 50)

    for subset_name in subsets:
        if subset_name not in AVAILABLE_SUBSETS:
            print(f"  [경고] 알 수 없는 서브셋: {subset_name}, 스킵")
            continue

        raw_samples = load_krew_subset(subset_name, cache_dir)
        converted_count = 0
        filtered_count = 0

        for sample in raw_samples:
            messages = sample.get("text", [])
            if not messages:
                continue

            conversations = convert_conversation(messages, subset_name, clean_emote)
            if conversations is None:
                filtered_count += 1
                continue

            if not filter_quality(conversations):
                filtered_count += 1
                continue

            all_data.append({
                "conversations": conversations,
                "subset": subset_name,
                "topic": sample.get("topic", ""),
            })
            converted_count += 1

        print(f"  [{subset_name}] 변환: {converted_count:,}, 필터링됨: {filtered_count}")

    print(f"\n  총 변환 샘플: {len(all_data):,}")

    # 셔플
    random.shuffle(all_data)

    # 샘플 수 제한
    if max_samples > 0 and len(all_data) > max_samples:
        all_data = all_data[:max_samples]
        print(f"  max_samples 적용: {len(all_data):,}")

    return all_data


def print_stats(data: list[dict]) -> None:
    """데이터 통계를 출력한다."""
    print("\n[2/3] 데이터 통계")
    print("=" * 50)

    # 서브셋별 분포
    subset_counts = {}
    total_turns = 0
    total_chars = 0

    for item in data:
        subset = item.get("subset", "unknown")
        subset_counts[subset] = subset_counts.get(subset, 0) + 1

        convos = item["conversations"]
        non_system = [c for c in convos if c["from"] != "system"]
        total_turns += len(non_system)
        total_chars += sum(len(c["value"]) for c in non_system)

    print("  서브셋별 분포:")
    for subset, count in sorted(subset_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(data)
        print(f"    {subset}: {count:,} ({pct:.1f}%)")

    avg_turns = total_turns / len(data) if data else 0
    avg_chars = total_chars / len(data) if data else 0
    print(f"\n  평균 턴 수: {avg_turns:.1f}")
    print(f"  평균 문자 수: {avg_chars:.0f}")
    print(f"  총 샘플: {len(data):,}")


def save_dataset(data: list[dict], output_path: str) -> None:
    """학습 데이터를 JSON 파일로 저장한다."""
    # subset, topic 필드 제거 (학습에 불필요)
    clean_data = [{"conversations": item["conversations"]} for item in data]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {output_path} ({size_mb:.1f} MB, {len(clean_data):,} 샘플)")


def main():
    parser = argparse.ArgumentParser(
        description="KREW korean-role-playing → 시온 LoRA 학습 데이터 변환"
    )
    parser.add_argument(
        "--output_dir", default="./data",
        help="출력 디렉토리 (기본: ./data)",
    )
    parser.add_argument(
        "--cache_dir", default="./data/krew_cache",
        help="HuggingFace 캐시 디렉토리",
    )
    parser.add_argument(
        "--subsets", nargs="+", default=AVAILABLE_SUBSETS,
        help=f"사용할 서브셋 (기본: 전체). 선택지: {AVAILABLE_SUBSETS}",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="최대 샘플 수 (0=전체, 기본: 0)",
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.05,
        help="평가 데이터 비율 (기본: 0.05)",
    )
    parser.add_argument(
        "--no_clean_emote", action="store_true",
        help="*액션* 이모트 제거하지 않기",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  시온(sion) KREW 롤플레이 데이터 전처리")
    print("=" * 60)
    print(f"  서브셋: {args.subsets}")
    print(f"  이모트 정리: {not args.no_clean_emote}")
    print(f"  최대 샘플: {'전체' if args.max_samples == 0 else args.max_samples}")
    print("=" * 60)

    # 1. 다운로드 및 변환
    data = process_krew_data(
        subsets=args.subsets,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        clean_emote=not args.no_clean_emote,
        seed=args.seed,
    )

    if not data:
        print("\n[오류] 변환된 데이터가 없습니다.")
        return

    # 2. 통계 출력
    print_stats(data)

    # 3. Train/Eval 분할
    print("\n[3/3] 데이터 저장")
    print("=" * 50)

    split_idx = int(len(data) * (1 - args.eval_ratio))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    train_path = os.path.join(args.output_dir, "train.json")
    eval_path = os.path.join(args.output_dir, "eval.json")

    save_dataset(train_data, train_path)
    save_dataset(eval_data, eval_path)

    print(f"\n  Train: {len(train_data):,}")
    print(f"  Eval:  {len(eval_data):,}")

    # 4. 샘플 출력
    print("\n[샘플 미리보기]")
    print("-" * 50)
    sample = data[0]
    for turn in sample["conversations"][:4]:  # system + 첫 2턴만
        role = turn["from"]
        value = turn["value"][:100]
        if len(turn["value"]) > 100:
            value += "..."
        print(f"  [{role}] {value}")

    print(f"\n다음 단계:")
    print(f"  python train_lora.py --data_dir {args.output_dir} --max_seq_length 1024 --epochs 2")


if __name__ == "__main__":
    main()
