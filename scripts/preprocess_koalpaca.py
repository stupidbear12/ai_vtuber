# -*- coding: utf-8 -*-
"""
KoAlpaca → 시온(sion) 말투 변환 전처리 스크립트

1. ko_alpaca_data.json 다운로드 (49,620건)
2. 일상 대화에 적합한 데이터 필터링 (일상/감정/관계/취미 관련)
3. output을 시온 말투 규칙으로 변환
4. QLoRA 학습용 JSONL 저장 → training/koalpaca_sion.jsonl

사용법:
  python scripts/preprocess_koalpaca.py
  python scripts/preprocess_koalpaca.py --max 3000   # 최대 3000건만 저장
  python scripts/preprocess_koalpaca.py --no-cache   # 강제 재다운로드
"""

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────
SEED = 42
DATA_URL = "https://raw.githubusercontent.com/Beomi/KoAlpaca/main/ko_alpaca_data.json"

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "training"
OUTPUT_PATH = OUTPUT_DIR / "koalpaca_sion.jsonl"
CACHE_PATH = OUTPUT_DIR / "ko_alpaca_raw.json"

random.seed(SEED)

# ── 시온 시스템 프롬프트 (기존 학습 데이터와 동일) ─────────────────────
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

# ── 필터: 포함 카테고리 키워드 ────────────────────────────────────────
# 이 키워드 중 하나라도 instruction+input+output에 있으면 통과 후보
INCLUDE_KEYWORDS = [
    # 일상/생활
    "일상", "하루", "오늘", "어제", "내일", "주말", "평일",
    # 감정/심리
    "기분", "감정", "행복", "슬프", "우울", "설레", "화나", "짜증", "걱정", "불안",
    "힘들", "고민", "스트레스", "위로", "격려", "응원",
    # 관계/대인
    "친구", "가족", "부모", "엄마", "아빠", "형", "언니", "오빠", "동생",
    "연인", "남자친구", "여자친구", "연애", "사랑", "짝사랑", "이별",
    "선생님", "선배", "후배", "동료",
    # 취미/여가
    "취미", "여가", "음악", "노래", "악기", "DJ", "방송", "유튜브", "라이브",
    "영화", "드라마", "애니", "웹툰", "책", "독서", "게임",
    "운동", "헬스", "달리기", "산책", "등산", "요가",
    "여행", "나들이", "카페", "맛집", "요리", "베이킹",
    "쇼핑", "패션", "뷰티", "메이크업",
    "반려동물", "고양이", "강아지",
    # 계절/날씨/환경
    "봄", "여름", "가을", "겨울", "날씨", "비", "눈", "더위", "추위",
    # 가벼운 지식/추천
    "추천", "좋아하는", "싫어하는", "재밌", "재미있", "좋아해",
    # SNS/인터넷 문화
    "인스타", "틱톡", "트위터", "유행", "밈", "챌린지",
]

# ── 필터: 제외 카테고리 키워드 ────────────────────────────────────────
# 이 키워드가 하나라도 있으면 무조건 제외 (전문적/학술)
EXCLUDE_KEYWORDS = [
    # 법/제도
    "법률", "판례", "조항", "소송", "계약서", "법원", "헌법", "형법", "민법",
    # 의학
    "의학", "진단", "처방", "약물", "수술", "치료", "병원", "임상",
    # 수학/과학
    "방정식", "미적분", "행렬", "화학식", "원소", "물리법칙", "양자역학",
    "증명", "정리", "공리", "수렴",
    # 프로그래밍
    "코드", "알고리즘", "함수", "프로그래밍", "파이썬", "자바", "컴파일",
    "데이터베이스", "SQL", "API",
    # 학술/논문
    "논문", "연구", "실험", "통계적", "유의미", "가설", "변수", "샘플링",
    # 역사/정치 (딱딱한)
    "조선시대", "삼국시대", "고려", "신라", "세계대전", "냉전", "혁명",
    "정치체제", "경제정책", "외교",
    # 특수 형식
    "번역해", "translate", "영어로",
]

# ── 감정 → 가중치 ─────────────────────────────────────────────────────
EMOTION_WEIGHTS = {
    "happy": 25, "excited": 15, "calm": 15, "surprised": 10,
    "love": 8, "shy": 7, "thinking": 7, "sad": 5,
    "worried": 5, "angry": 3,
}

# 텍스트 키워드로 감정 유추
EMOTION_HINTS = [
    (["좋아", "행복", "기쁘", "즐거", "웃음", "신나", "흐흐", "ㅎㅎ"], "happy"),
    (["슬프", "ㅠㅠ", "ㅜㅜ", "힘들", "속상", "아쉽", "안타깝"], "sad"),
    (["헐", "대박", "놀라", "신기", "와", "오오", "믿기지"], "surprised"),
    (["음...", "글쎄", "생각해보면", "사실", "어렵", "모르겠"], "thinking"),
    (["설레", "두근", "너무 좋", "완전", "최고", "짱"], "excited"),
    (["괜찮", "알겠어", "그렇구나", "맞아", "이해"], "calm"),
    (["걱정", "조심", "불안", "무서", "어떡하"], "worried"),
    (["화나", "짜증", "열받", "억울", "싫어"], "angry"),
    (["사랑", "보고싶", "그리워", "소중", "따뜻"], "love"),
    (["부끄", "쑥스", "창피", "민망"], "shy"),
]


def pick_emotion(text: str) -> str:
    for keywords, emotion in EMOTION_HINTS:
        if any(kw in text for kw in keywords):
            return emotion
    tags = list(EMOTION_WEIGHTS.keys())
    weights = [EMOTION_WEIGHTS[t] for t in tags]
    return random.choices(tags, weights=weights, k=1)[0]


def is_conversational(item: dict) -> bool:
    """일상 대화에 적합한 항목인지 판별."""
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    inp = item.get("input", "")
    full_text = instruction + " " + inp + " " + output

    # 제외 키워드 우선 검사
    for kw in EXCLUDE_KEYWORDS:
        if kw in full_text:
            return False

    # output 길이 필터 (너무 짧거나 너무 긴 것 제외)
    out_len = len(output.strip())
    if out_len < 20 or out_len > 600:
        return False

    # 번호 나열식 응답 제외 (3줄 이상 번호 목록)
    numbered_lines = len(re.findall(r'^\d+[.\.]\s+', output, re.MULTILINE))
    if numbered_lines >= 3:
        return False

    # 영어 비율이 높으면 제외 (50% 초과)
    korean_chars = len(re.findall(r'[가-힣]', full_text))
    ascii_chars = len(re.findall(r'[a-zA-Z]', full_text))
    if ascii_chars > 0 and korean_chars / (korean_chars + ascii_chars + 1) < 0.5:
        return False

    # 포함 키워드 체크
    for kw in INCLUDE_KEYWORDS:
        if kw in full_text:
            return True

    # 짧은 일상 질문 형태 (키워드 없어도 허용)
    if len(instruction) < 60 and re.search(r'(어때|뭐야|어떻게|좋아|싫어|했어|했니|할까|\?)', instruction):
        return True

    return False


def to_sion_style(text: str) -> str:
    """텍스트를 시온 말투(반말, 짧고 에너지 넘침)로 변환."""
    # 마크다운 제거
    text = re.sub(r'^[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+[.\.]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 줄바꿈 → 공백 (대화 형식에 맞게)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    # 존댓말 → 반말 변환 (어미 순서: 긴/구체적 패턴 먼저)
    replacements = [
        # ── 자주 쓰는 관용 표현 (먼저 처리) ──
        (r'알겠습니다', '알겠어'),
        (r'감사합니다', '고마워'),
        (r'죄송합니다', '미안해'),
        (r'모릅니다', '몰라'),
        (r'됩니다', '돼'),
        (r'됩니까\??', '돼?'),
        # ── ㅂ니다/ㅂ니까 계열 (vowel-stem 동사) ──
        (r'드립니다', '줄게'),
        (r'드립니까\??', '줄게?'),
        (r'겁니다', '거야'),
        (r'겁니까\??', '거야?'),
        (r'갑니다', '가'),
        (r'갑니까\??', '가?'),
        (r'옵니다', '와'),
        (r'옵니까\??', '와?'),
        (r'봅니다', '봐'),
        (r'봅니까\??', '봐?'),
        (r'줍니다', '줘'),
        (r'줍니까\??', '줘?'),
        (r'쉽니다', '쉬워'),
        (r'섭니다', '서'),
        (r'납니다', '나'),
        (r'납니까\??', '나?'),
        # ── 습니다/습니까 계열 (consonant-stem 동사) ──
        (r'습니다', '어'),
        (r'습니까\??', '어?'),
        # ── 합니다 계열 ──
        (r'합니다', '해'),
        (r'합니까\??', '해?'),
        (r'하십시오', '해'),
        (r'하세요', '해'),
        (r'하셔요', '해'),
        # ── 입니다 계열 ──
        (r'입니다', '이야'),
        (r'입니까\??', '이야?'),
        (r'인가요\??', '이야?'),
        # ── 요 계열 어미 ──
        (r'나요\??', '나?'),
        (r'이에요', '이야'),
        (r'에요', '야'),
        (r'어요', '어'),
        (r'아요', '아'),
        (r'해요', '해'),
        (r'세요', '해'),
        (r'겠어요', '겠어'),
        (r'겠죠', '겠지'),
        (r'거예요', '거야'),
        (r'거에요', '거야'),
        # ── 청유/의문 ──
        (r'할까요\??', '할까?'),
        (r'볼까요\??', '볼까?'),
        (r'줄까요\??', '줄까?'),
        (r'할게요', '할게'),
        (r'줄게요', '줄게'),
        (r'볼게요', '볼게'),
        (r'드릴게요', '줄게'),
        (r'드릴까요\??', '줄까?'),
        (r'해볼게요', '해볼게'),
        (r'해봐요', '해봐'),
        # ── 높임 동사 ──
        (r'드세요', '먹어'),
        (r'드셔요', '먹어'),
        (r'드셨', '먹었'),
        (r'하셨', '했'),
        (r'보셨', '봤'),
        (r'가셨', '갔'),
        (r'오셨', '왔'),
        # ── 기타 ──
        (r'있어요', '있어'),
        (r'없어요', '없어'),
        (r'좋아요', '좋아'),
        (r'괜찮아요', '괜찮아'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # 길이 제한: 최대 150자 (방송 채팅 스타일)
    MAX_LEN = 150
    if len(text) > MAX_LEN:
        # 문장 단위로 자르기
        parts = re.split(r'(?<=[.!?~ㅠㅜ])\s+', text)
        result = ""
        for part in parts:
            candidate = (result + " " + part).strip()
            if len(candidate) > MAX_LEN:
                break
            result = candidate
        text = result.strip() if result.strip() else text[:MAX_LEN]

    return text.strip()


def download_data(use_cache: bool = True) -> list:
    if use_cache and CACHE_PATH.exists():
        print(f"[캐시] {CACHE_PATH} 사용 (--no-cache로 강제 재다운로드)")
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    print(f"[다운로드] {DATA_URL}")
    print("  (약 20MB, 잠시 기다려주세요...)")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"[오류] 다운로드 실패: {e}")
        sys.exit(1)

    data = json.loads(raw)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[저장] 캐시: {CACHE_PATH} ({len(data):,}건)")
    return data


def main():
    parser = argparse.ArgumentParser(description="KoAlpaca → 시온 말투 전처리")
    parser.add_argument("--max", type=int, default=0, help="최대 출력 건수 (0=제한없음)")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시하고 재다운로드")
    args = parser.parse_args()

    # 1. 다운로드
    data = download_data(use_cache=not args.no_cache)
    print(f"[로드] 총 {len(data):,}건")

    # 2. 필터링
    print("[필터링] 일상 대화 적합 항목 선별 중...")
    filtered = [item for item in data if is_conversational(item)]
    print(f"[필터] {len(filtered):,}건 선택 "
          f"({len(filtered)/len(data)*100:.1f}% / 전체 {len(data):,}건)")

    # 3. 셔플 (다양성 확보)
    random.shuffle(filtered)

    # 4. 최대 건수 제한
    if args.max > 0:
        filtered = filtered[:args.max]
        print(f"[제한] --max {args.max} 적용 → {len(filtered):,}건")

    # 5. 변환 및 JSONL 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    print(f"[변환] 시온 말투로 변환 중...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in filtered:
            instruction = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            output = item.get("output", "").strip()

            # user 메시지 구성
            user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
            if not user_msg:
                skipped += 1
                continue

            # 시온 말투 변환
            sion_output = to_sion_style(output)
            if not sion_output or len(sion_output) < 10:
                skipped += 1
                continue

            # 감정 태그 추가
            emotion = pick_emotion(sion_output)
            sion_output = f"[감정:{emotion}] {sion_output}"

            record = {
                "messages": [
                    {"role": "system", "content": SION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": sion_output},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"\n[완료] 변환: {converted:,}건 | 스킵: {skipped:,}건")
    print(f"[저장] {OUTPUT_PATH}")

    # 6. 샘플 출력
    print("\n" + "=" * 60)
    print("변환 샘플 (5건)")
    print("=" * 60)
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line)
            msgs = rec["messages"]
            user_text = next(m["content"] for m in msgs if m["role"] == "user")
            asst_text = next(m["content"] for m in msgs if m["role"] == "assistant")
            print(f"[{i+1}] Q: {user_text[:70]}")
            print(f"     A: {asst_text}")
            print()

    # 7. 기존 학습 데이터와 합산 안내
    existing_train = OUTPUT_DIR / "sion_train.jsonl"
    if existing_train.exists():
        existing_count = sum(1 for _ in open(existing_train, encoding="utf-8"))
        print(f"[참고] 기존 sion_train.jsonl: {existing_count:,}건")
        print(f"[참고] 합산 시 총 {existing_count + converted:,}건 학습 가능")
        print()
        print("기존 데이터와 합치려면:")
        print(f"  cat training/sion_train.jsonl training/koalpaca_sion.jsonl > training/sion_combined.jsonl")


if __name__ == "__main__":
    main()
