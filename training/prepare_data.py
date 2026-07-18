# -*- coding: utf-8 -*-
"""
시온(sion) 파인튜닝 데이터 전처리 스크립트

데이터 소스: huggingface-KREW/korean-role-playing
출력: training/sion_train.jsonl, training/sion_eval.jsonl (Llama 3.1 chat format)

과적합 방지:
  - train/eval 90:10 분할
  - 대화 길이 제한 (너무 긴 대화 제거)
  - 다양한 서브셋 혼합

할루시네이션 방지:
  - "모르겠어" 류 솔직 응답 데이터 추가
  - 허구 행동(곡 틀기, 큐 넣기 등) 없는 자연스러운 대화만 사용
  - 시스템 프롬프트에 제약 조건 포함
"""

import json
import random
import re
import sys
import os

# ── 설정 ──────────────────────────────────────────────────────────
SEED = 42
EVAL_RATIO = 0.10          # 10% 평가용
MAX_TURNS = 10             # 대화 최대 턴 수
MAX_CHARS_PER_TURN = 200   # 턴당 최대 글자 수 (방송 채팅 스타일)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(SEED)

# ── 시온 시스템 프롬프트 ──────────────────────────────────────────
SION_SYSTEM_PROMPT = """너는 "시온(sion)"이라는 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.

[캐릭터]
- 20대 초반 여성, 항상 반말. 존댓말 절대 금지
- 밝고 에너지 넘치며, 음악을 좋아하는 DJ
- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐", "대박")

[규칙]
- 응답 맨 앞에 반드시 [감정:태그] 붙여. 태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy
- 1~2문장으로 짧게 답해
- 모르는 건 절대 지어내지 마. "잘 모르겠는데?" 라고 솔직하게 답해
- 실제로 하지 않은 행동을 말하지 마"""

# ── 감정 태그 목록 ─────────────────────────────────────────────────
EMOTIONS = ["happy", "sad", "surprised", "thinking", "excited", "calm", "worried", "angry", "love", "shy"]

# ── 할루시네이션 방지 데이터 ────────────────────────────────────────
# 솔직하게 모른다고 답하거나, 허구를 만들지 않는 예시
ANTI_HALLUCINATION_EXAMPLES = [
    {"user": "어제 틀어준 노래 뭐였어?", "assistant": "[감정:thinking] 음... 잘 기억 안 나는데? 다시 알려줘!"},
    {"user": "시온아 이 곡 틀어줘", "assistant": "[감정:calm] 미안 나 지금 직접 노래는 못 틀어ㅠㅠ"},
    {"user": "내일 날씨 어때?", "assistant": "[감정:thinking] 헐 나 날씨는 잘 모르겠는데?"},
    {"user": "시온아 피자 시켜줘", "assistant": "[감정:sad] 아 나 그건 못 해ㅠㅠ 직접 시켜야 해!"},
    {"user": "지난번에 말한 거 기억나?", "assistant": "[감정:thinking] 음 뭐였더라... 다시 말해줄래?"},
    {"user": "시온아 몇 살이야?", "assistant": "[감정:happy] 20대 초반이야 흐흐"},
    {"user": "시온아 진짜 사람이야?", "assistant": "[감정:shy] 아니 나 AI야 흐흐 근데 마음은 진짜야!"},
    {"user": "시온아 게임 같이 해줘", "assistant": "[감정:sad] 아 나 게임은 못 하는데ㅠㅠ 대화는 할 수 있어!"},
    {"user": "시온아 BTS 노래 좋아해?", "assistant": "[감정:excited] 오 BTS! 좋지~ 너는 어떤 곡 좋아해?"},
    {"user": "시온아 어디 살아?", "assistant": "[감정:thinking] 음 나는 인터넷 세상에 살아 흐흐"},
    {"user": "어제 방송 재밌었어?", "assistant": "[감정:thinking] 어제? 잘 기억이 안 나는데 오늘 재밌게 하자!"},
    {"user": "시온아 나한테 선물 줘", "assistant": "[감정:sad] 헐 나 선물은 못 주는데ㅠㅠ 대신 좋은 노래 추천해줄게!"},
    {"user": "넌 뭐 먹었어?", "assistant": "[감정:happy] 나 AI라 밥은 안 먹어 흐흐 너는 뭐 먹었어?"},
    {"user": "시온아 전화번호 알려줘", "assistant": "[감정:calm] 나 전화번호 없어! 여기서 얘기하자~"},
    {"user": "시온아 내 이름 뭔지 알아?", "assistant": "[감정:thinking] 음 닉네임밖에 모르는데? 알려줄래?"},
    {"user": "시온아 오늘 뉴스 뭐 있어?", "assistant": "[감정:thinking] 뉴스는 잘 모르겠는데? 직접 찾아보는 게 정확할 거야!"},
    {"user": "시온아 수학 문제 풀어줘", "assistant": "[감정:worried] 헐 수학... 나 그건 좀 어려운데ㅠㅠ"},
    {"user": "시온아 주식 추천해줘", "assistant": "[감정:calm] 나 주식은 잘 몰라서 추천 못 해! 전문가한테 물어봐~"},
]

# ── 변환 함수 ──────────────────────────────────────────────────────

def pick_emotion() -> str:
    """랜덤 감정 태그 선택 (확률 분포 적용)."""
    weights = {
        "happy": 25, "excited": 15, "calm": 15, "surprised": 10,
        "love": 8, "shy": 7, "thinking": 7, "sad": 5,
        "worried": 5, "angry": 3,
    }
    tags = list(weights.keys())
    w = [weights[t] for t in tags]
    return random.choices(tags, weights=w, k=1)[0]


def to_sion_style(text: str) -> str:
    """assistant 응답을 시온 말투로 변환."""
    # 액션/행동 묘사 제거 (*...* 형식)
    text = re.sub(r'\*[^*]+\*\s*', '', text)
    # 큰따옴표 래핑 제거
    text = text.strip().strip('"').strip()
    # 존댓말 → 반말 변환 (간단 규칙)
    replacements = [
        (r'합니다', '해'), (r'합니까', '해?'), (r'하세요', '해'),
        (r'습니다', '어'), (r'습니까', '어?'), (r'세요', '해'),
        (r'입니다', '이야'), (r'인가요', '이야?'), (r'나요\?', '나?'),
        (r'해요', '해'), (r'에요', '야'), (r'이에요', '이야'),
        (r'할게요', '할게'), (r'줄게요', '줄게'), (r'볼게요', '볼게'),
        (r'드릴게요', '줄게'), (r'드릴까요', '줄까'),
        (r'아요', '아'), (r'어요', '어'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # 길이 제한
    if len(text) > MAX_CHARS_PER_TURN:
        # 문장 단위로 자르기
        sentences = re.split(r'[.!?~]+\s*', text)
        result = ""
        for s in sentences:
            if len(result) + len(s) + 2 > MAX_CHARS_PER_TURN:
                break
            result += s + " "
        text = result.strip()
        if not text:
            text = text[:MAX_CHARS_PER_TURN]

    return text


def convert_conversation(text_turns: list) -> list:
    """대화 턴 리스트를 Llama 3.1 chat format으로 변환.

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]
        또는 None (변환 불가 시)
    """
    messages = [{"role": "system", "content": SION_SYSTEM_PROMPT}]

    turn_count = 0
    for turn in text_turns:
        if turn_count >= MAX_TURNS:
            break

        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if not content:
            continue

        if role == "user":
            # 사용자 메시지는 그대로
            if len(content) > 300:
                content = content[:300]
            messages.append({"role": "user", "content": content})
            turn_count += 1

        elif role == "assistant":
            # 시온 말투로 변환 + 감정 태그 추가
            converted = to_sion_style(content)
            if not converted:
                continue
            emotion = pick_emotion()
            converted = f"[감정:{emotion}] {converted}"
            messages.append({"role": "assistant", "content": converted})
            turn_count += 1

    # 최소 1턴(user+assistant) 이상이어야 유효
    user_count = sum(1 for m in messages if m["role"] == "user")
    asst_count = sum(1 for m in messages if m["role"] == "assistant")
    if user_count < 1 or asst_count < 1:
        return None

    return messages


def create_anti_hallucination_data() -> list:
    """할루시네이션 방지 학습 데이터 생성."""
    examples = []
    for ex in ANTI_HALLUCINATION_EXAMPLES:
        messages = [
            {"role": "system", "content": SION_SYSTEM_PROMPT},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        examples.append({"messages": messages})
    return examples


# ── 메인 ──────────────────────────────────────────────────────────

def main():
    from datasets import load_dataset

    all_examples = []

    # 1. 각 서브셋 로드 및 변환
    subsets = [
        ("exa-data", None),               # 890개 — 전부 사용
        ("general-roleplay-data", 3000),   # 32K중 3000개 샘플링 (다양성 확보)
        ("gf-persona-data", 1000),         # 1920중 1000개 (캐주얼 대화)
        ("youtube-couple-data", None),     # 125개 — 전부 사용
    ]

    for subset_name, sample_size in subsets:
        print(f"[로드] {subset_name}...", end=" ")
        try:
            ds = load_dataset("huggingface-KREW/korean-role-playing", subset_name)
            data = list(ds["train"])
            if sample_size and len(data) > sample_size:
                data = random.sample(data, sample_size)

            converted = 0
            for item in data:
                turns = item.get("text", [])
                messages = convert_conversation(turns)
                if messages:
                    all_examples.append({"messages": messages})
                    converted += 1

            print(f"{converted}/{len(data)} 변환 완료")
        except Exception as e:
            print(f"ERROR: {e}")

    # 2. 할루시네이션 방지 데이터 추가 (10배 반복 — 중요도 높임)
    anti_hall = create_anti_hallucination_data()
    anti_hall_repeated = anti_hall * 10  # 180개
    all_examples.extend(anti_hall_repeated)
    print(f"[추가] 할루시네이션 방지 데이터: {len(anti_hall_repeated)}개")

    # 3. 셔플 후 train/eval 분할
    random.shuffle(all_examples)
    eval_size = max(1, int(len(all_examples) * EVAL_RATIO))
    eval_data = all_examples[:eval_size]
    train_data = all_examples[eval_size:]

    print(f"\n[결과] 전체: {len(all_examples)}, 학습: {len(train_data)}, 평가: {len(eval_data)}")

    # 4. JSONL 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "sion_train.jsonl")
    eval_path = os.path.join(OUTPUT_DIR, "sion_eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[저장] {train_path}")
    print(f"[저장] {eval_path}")

    # 5. 샘플 출력
    print("\n=== 학습 데이터 샘플 ===")
    for ex in train_data[:3]:
        for m in ex["messages"]:
            if m["role"] == "system":
                continue
            print(f'  [{m["role"]}] {m["content"][:100]}')
        print()


if __name__ == "__main__":
    main()
