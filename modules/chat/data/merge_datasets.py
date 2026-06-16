"""
시온(sion) AI VTuber 파인튜닝 데이터 강화 스크립트

HuggingFace 데이터셋 3개를 다운로드하여 시온 학습 포맷(ShareGPT + 감정태그)으로 변환하고,
기존 sion_dataset.jsonl과 합쳐 sion_dataset_v2.jsonl을 생성합니다.

사용법:
    pip install datasets --break-system-packages
    python merge_datasets.py
"""

import json
import re
import os
import ast
from pathlib import Path
from collections import Counter

# ─── 감정 분류기 (키워드 기반) ───────────────────────────────────────

EMOTION_KEYWORDS = {
    "excited": [
        "대박", "최고", "짱", "완전", "진짜", "미쳤", "소름", "갈거야", "가자",
        "해보자", "재밌", "신나", "기대", "좋겠다", "꼭", "드디어", "우와", "오오",
        "해볼래", "같이", "가고 싶", "하고 싶", "해보고 싶", "배우고 싶",
    ],
    "happy": [
        "좋아", "좋지", "행복", "웃기", "ㅋㅋ", "ㅎㅎ", "재밌", "맛있",
        "즐거", "기뻐", "반가", "축하", "고마", "감사", "사랑해", "멋있",
        "멋지", "좋겠", "다행", "잘됐", "귀여", "이쁘", "예쁘", "좋은",
        "괜찮", "화이팅", "응원", "행운", "잘 지냈",
    ],
    "sad": [
        "슬퍼", "슬프", "우울", "눈물", "울었", "힘들", "외로", "그리",
        "아쉬", "안타깝", "불쌍", "죽고", "싫어", "지겨", "지쳤", "포기",
        "힘내", "괴로", "서글", "쓸쓸", "허전", "미안", "죄송", "후회",
    ],
    "angry": [
        "짜증", "화나", "열받", "미친", "싫어", "나쁜", "최악", "별로",
        "어이없", "황당", "분노", "화가", "빡치", "욕", "못참", "억울",
        "진상", "짜증나", "환멸",
    ],
    "surprised": [
        "헐", "어머", "깜짝", "세상에", "진짜?", "정말?", "오!", "와!",
        "놀라", "충격", "신기", "독특", "처음", "몰랐", "설마", "갑자기",
        "대단", "불가능", "상상도",
    ],
    "worried": [
        "걱정", "불안", "무서", "두려", "긴장", "조심", "위험", "겁나",
        "어떡", "어떻게", "큰일", "심각", "아프", "다치", "사고", "병원",
        "아플", "조심해", "무섭",
    ],
    "thinking": [
        "생각", "고민", "궁금", "왜", "어떻게", "의미", "이유", "철학",
        "모르겠", "글쎄", "아마", "혹시", "그런가", "맞나", "인생",
        "사실", "원래", "관점", "차이",
    ],
    "love": [
        "사랑", "좋아해", "보고 싶", "그리워", "설레", "두근", "애정",
        "소중", "함께", "영원", "약속", "운명", "가슴이", "행복해",
        "예쁘", "안아", "포옹",
    ],
    "shy": [
        "부끄", "창피", "쑥스", "민망", "칭찬", "과찬", "에이",
        "별거 아니", "아니야", "그런거 아니",
    ],
    "calm": [
        "편안", "조용", "평화", "차분", "여유", "쉬", "릴렉스", "힐링",
        "산책", "음악", "커피", "책", "날씨", "바람", "하늘",
        "감성", "밤", "새벽", "고요",
    ],
}


def classify_emotion(text: str) -> str:
    """텍스트 내용을 기반으로 적절한 감정 태그를 반환합니다."""
    text_lower = text.lower()
    scores = Counter()

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1

    if not scores:
        return "happy"  # 기본값

    return scores.most_common(1)[0][0]


# ─── 필터링 유틸 ─────────────────────────────────────────────────────

def is_korean(text: str) -> bool:
    """텍스트에 한글이 30% 이상 포함되어 있는지 확인합니다."""
    if not text.strip():
        return False
    korean_chars = len(re.findall(r"[가-힣]", text))
    total_chars = len(re.findall(r"\S", text))  # 공백 제외
    if total_chars == 0:
        return False
    return korean_chars / total_chars > 0.3


def is_long_enough(text: str) -> bool:
    """응답이 최소 1문장 이상인지 확인합니다."""
    text = text.strip()
    if len(text) < 10:
        return False
    # 문장 종결 패턴 확인
    sentence_endings = re.findall(r"[.!?~다요해야지거든걸뭐까]", text)
    return len(sentence_endings) >= 1 or len(text) >= 20


def already_has_emotion_tag(text: str) -> bool:
    """이미 감정 태그가 있는지 확인합니다."""
    return bool(re.match(r"\[감정:\w+\]", text))


def add_emotion_tag(text: str) -> str:
    """텍스트에 감정 태그를 추가합니다. 이미 있으면 그대로 반환."""
    if already_has_emotion_tag(text):
        return text
    emotion = classify_emotion(text)
    return f"[감정:{emotion}] {text}"


# ─── 데이터셋 변환 함수들 ────────────────────────────────────────────

def convert_korean_roleplay(dataset) -> list[dict]:
    """
    huggingface-KREW/korean-role-playing 변환
    구조: text = [{content, role}, ...], optional topic
    role: "user" / "assistant" / "system"
    """
    results = []

    for item in dataset:
        text_list = item["text"]
        if not text_list or len(text_list) < 2:
            continue

        conversations = []
        for turn in text_list:
            content = turn["content"].strip()
            role = turn["role"].strip().lower()

            if not content:
                continue

            if role in ("user", "human"):
                conversations.append({"from": "human", "value": content})
            elif role in ("assistant", "gpt", "bot"):
                if not is_korean(content) or not is_long_enough(content):
                    continue
                tagged = add_emotion_tag(content)
                conversations.append({"from": "gpt", "value": tagged})
            elif role == "system":
                # 시스템 프롬프트는 제외 (파인튜닝 시 별도 추가)
                continue

        # 최소 1턴(human+gpt) 확인
        has_human = any(c["from"] == "human" for c in conversations)
        has_gpt = any(c["from"] == "gpt" for c in conversations)
        if has_human and has_gpt:
            # human-gpt 순서 정리: human으로 시작하도록
            cleaned = []
            for c in conversations:
                if not cleaned and c["from"] == "gpt":
                    continue  # gpt로 시작하면 스킵
                # 같은 role이 연속으로 오면 합치기
                if cleaned and cleaned[-1]["from"] == c["from"]:
                    cleaned[-1]["value"] += " " + c["value"]
                else:
                    cleaned.append(c)

            if len(cleaned) >= 2:
                results.append({"conversations": cleaned})

    return results


def convert_persona_chat(csv_data: list[dict]) -> list[dict]:
    """
    NLPBada/korean-persona-chat-dataset 변환
    구조: session_dialog (stringified list), session_persona (stringified list)
    dialog는 교대 발화 (speaker1, speaker2 번갈아)
    """
    results = []

    for item in csv_data:
        try:
            dialog_str = item.get("session_dialog", "")
            dialog = ast.literal_eval(dialog_str)
        except (ValueError, SyntaxError):
            continue

        if not dialog or len(dialog) < 2:
            continue

        conversations = []
        for i, utterance in enumerate(dialog):
            utterance = utterance.strip()
            if not utterance:
                continue

            if i % 2 == 0:
                # 짝수 인덱스 = 첫 번째 화자 → human
                conversations.append({"from": "human", "value": utterance})
            else:
                # 홀수 인덱스 = 두 번째 화자 → gpt
                if not is_korean(utterance) or not is_long_enough(utterance):
                    continue
                tagged = add_emotion_tag(utterance)
                conversations.append({"from": "gpt", "value": tagged})

        has_human = any(c["from"] == "human" for c in conversations)
        has_gpt = any(c["from"] == "gpt" for c in conversations)
        if has_human and has_gpt and len(conversations) >= 2:
            results.append({"conversations": conversations})

    return results


def convert_sharegpt_ko(data: list[dict]) -> list[dict]:
    """
    dbdu/ShareGPT-74k-ko 변환
    구조: ShareGPT 원본 포맷 - conversations [{from, value}, ...]
    from: "human", "gpt", "system"
    """
    results = []

    for item in data:
        convs = item.get("conversations", [])
        if not convs:
            continue

        conversations = []
        for turn in convs:
            role = turn.get("from", "").strip().lower()
            content = turn.get("value", "").strip()

            if not content:
                continue

            if role == "human":
                if not is_korean(content):
                    continue
                conversations.append({"from": "human", "value": content})
            elif role == "gpt":
                if not is_korean(content) or not is_long_enough(content):
                    continue
                tagged = add_emotion_tag(content)
                conversations.append({"from": "gpt", "value": tagged})
            # system은 제외

        # 연속 같은 role 합치기
        cleaned = []
        for c in conversations:
            if not cleaned and c["from"] == "gpt":
                continue
            if cleaned and cleaned[-1]["from"] == c["from"]:
                cleaned[-1]["value"] += " " + c["value"]
            else:
                cleaned.append(c)

        has_human = any(c["from"] == "human" for c in cleaned)
        has_gpt = any(c["from"] == "gpt" for c in cleaned)
        if has_human and has_gpt and len(cleaned) >= 2:
            results.append({"conversations": cleaned})

    return results


# ─── 메인 파이프라인 ─────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    existing_path = script_dir / "sion_dataset.jsonl"
    output_path = script_dir / "sion_dataset_v2.jsonl"

    print("=" * 60)
    print("시온(sion) 학습 데이터 강화 스크립트")
    print("=" * 60)

    # ── 1. 기존 데이터 로드 ──
    print("\n[1/5] 기존 sion_dataset.jsonl 로드 중...")
    existing_data = []
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_data.append(json.loads(line))
    print(f"  → 기존 데이터: {len(existing_data)}건")

    # ── 2. HuggingFace 데이터셋 다운로드 ──
    print("\n[2/5] HuggingFace 데이터셋 다운로드 중...")
    from datasets import load_dataset
    import csv
    import io

    # 2-1) korean-role-playing (4개 config)
    print("  → huggingface-KREW/korean-role-playing 다운로드 중...")
    roleplay_data = []
    configs = ["exa-data", "general-roleplay-data", "gf-persona-data", "youtube-couple-data"]
    for config in configs:
        try:
            ds = load_dataset("huggingface-KREW/korean-role-playing", config, split="train")
            roleplay_data.extend(ds)
            print(f"    - {config}: {len(ds)}건")
        except Exception as e:
            print(f"    - {config}: 실패 ({e})")
    print(f"  → 합계: {len(roleplay_data)}건")

    # 2-2) korean-persona-chat-dataset
    print("  → NLPBada/korean-persona-chat-dataset 다운로드 중...")
    persona_data = []
    try:
        ds_persona = load_dataset("NLPBada/korean-persona-chat-dataset", split="train")
        persona_data = list(ds_persona)
        print(f"    - train: {len(persona_data)}건")
    except Exception as e:
        print(f"    - 실패 ({e})")
    # validation set도 시도
    try:
        ds_persona_val = load_dataset("NLPBada/korean-persona-chat-dataset", split="validation")
        persona_data.extend(list(ds_persona_val))
        print(f"    - validation: {len(ds_persona_val)}건")
    except Exception:
        pass
    print(f"  → 합계: {len(persona_data)}건")

    # 2-3) ShareGPT-74k-ko
    print("  → dbdu/ShareGPT-74k-ko 다운로드 중...")
    sharegpt_data = []
    try:
        ds_sharegpt = load_dataset("dbdu/ShareGPT-74k-ko", split="train")
        sharegpt_data = list(ds_sharegpt)
        print(f"    - {len(sharegpt_data)}건")
    except Exception:
        # JSON 파일 직접 다운로드 시도
        try:
            from huggingface_hub import hf_hub_download
            for fname in ["part1_ko_cleaned.json", "part2_ko_cleaned.json"]:
                try:
                    fpath = hf_hub_download(
                        repo_id="dbdu/ShareGPT-74k-ko",
                        filename=fname,
                        repo_type="dataset",
                    )
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sharegpt_data.extend(data)
                    print(f"    - {fname}: {len(data)}건")
                except Exception as e2:
                    print(f"    - {fname}: 실패 ({e2})")
        except ImportError:
            print("    - huggingface_hub 없음. pip install huggingface_hub 필요")
    print(f"  → 합계: {len(sharegpt_data)}건")

    # ── 3. 포맷 변환 ──
    print("\n[3/5] 시온 포맷으로 변환 중...")

    converted_roleplay = convert_korean_roleplay(roleplay_data)
    print(f"  → korean-role-playing: {len(roleplay_data)}건 → {len(converted_roleplay)}건")

    converted_persona = convert_persona_chat(persona_data)
    print(f"  → korean-persona-chat: {len(persona_data)}건 → {len(converted_persona)}건")

    converted_sharegpt = convert_sharegpt_ko(sharegpt_data)
    print(f"  → ShareGPT-74k-ko: {len(sharegpt_data)}건 → {len(converted_sharegpt)}건")

    # ── 4. 병합 ──
    print("\n[4/5] 데이터 병합 중...")
    all_data = existing_data + converted_roleplay + converted_persona + converted_sharegpt
    print(f"  → 총 데이터: {len(all_data)}건")

    # ── 5. 저장 및 통계 ──
    print("\n[5/5] sion_dataset_v2.jsonl 저장 중...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → 저장 완료: {output_path}")

    # 통계 출력
    print("\n" + "=" * 60)
    print("최종 데이터 통계")
    print("=" * 60)
    print(f"  기존 (sion_dataset.jsonl):       {len(existing_data):>8,}건")
    print(f"  korean-role-playing:             {len(converted_roleplay):>8,}건")
    print(f"  korean-persona-chat:             {len(converted_persona):>8,}건")
    print(f"  ShareGPT-74k-ko:                 {len(converted_sharegpt):>8,}건")
    print(f"  ─────────────────────────────────────────")
    print(f"  총합 (sion_dataset_v2.jsonl):     {len(all_data):>8,}건")

    # 평균 턴 수
    total_turns = sum(len(item["conversations"]) for item in all_data)
    avg_turns = total_turns / len(all_data) if all_data else 0
    print(f"\n  전체 평균 턴 수: {avg_turns:.2f}")

    # 소스별 평균 턴 수
    for name, data in [
        ("기존 데이터", existing_data),
        ("korean-role-playing", converted_roleplay),
        ("korean-persona-chat", converted_persona),
        ("ShareGPT-74k-ko", converted_sharegpt),
    ]:
        if data:
            turns = sum(len(d["conversations"]) for d in data)
            print(f"  {name} 평균 턴 수: {turns / len(data):.2f}")

    # 감정 태그 분포 (변환된 데이터 기준)
    emotion_counts = Counter()
    for item in converted_roleplay + converted_persona + converted_sharegpt:
        for conv in item["conversations"]:
            if conv["from"] == "gpt":
                match = re.match(r"\[감정:(\w+)\]", conv["value"])
                if match:
                    emotion_counts[match.group(1)] += 1

    if emotion_counts:
        print(f"\n  감정 태그 분포 (새로 추가된 데이터):")
        for emotion, count in emotion_counts.most_common():
            pct = count / sum(emotion_counts.values()) * 100
            print(f"    {emotion:>12}: {count:>6,}건 ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
