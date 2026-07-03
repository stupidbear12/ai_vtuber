# -*- coding: utf-8 -*-
"""
HuggingFace 한국어 대화 데이터셋 → 시온 파인튜닝 포맷 변환
- NLPBada/korean-persona-chat-dataset (반말 대화 10K+)
- huggingface-KREW/korean-role-playing (자연스러운 롤플레이 대화)
"""
import json, os, random, re, sys, time

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_TRAIN = os.path.join(TRAINING_DIR, "sion_hf_emotion.jsonl")
OUTPUT_EVAL = os.path.join(TRAINING_DIR, "sion_hf_emotion_eval.jsonl")
OUTPUT_COMBINED = os.path.join(TRAINING_DIR, "sion_combined_v2.jsonl")
EXISTING_DATA = os.path.join(TRAINING_DIR, "sion_combined_clean.jsonl")

SYSTEM_PROMPT = '''너는 "시온(sion)"이라는 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.

캐릭터 설명
- 20대 초반 여성, 항상 반말. 존댓말 절대 금지
- 밝고 에너지 넘치며, 음악을 좋아하는 DJ
- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐", "대박")

규칙
- 응답 맨 앞에 반드시 [감정:태그] 붙여. 태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy
- 1~2문장으로 짧게 답해
- 모르는 건 절대 지어내지 마. "잘 모르겠는데?" 라고 솔직하게 답해
- 실제로 하지 않은 행동을 말하지 마
- 응답에 절대 [시온], [반말], [캐릭터 설정], [sion] 같은 메타 태그를 넣지 마. [감정:태그]만 허용'''

EMOTIONS = ["happy", "sad", "surprised", "thinking", "excited", "calm", "worried", "angry", "love", "shy"]

# 감정 키워드 매핑
EMOTION_KEYWORDS = {
    "happy": ["좋아", "좋겠", "재밌", "웃기", "신나", "즐거", "기뻐", "행복", "축하", "잘했", "대박", "멋지"],
    "sad": ["슬퍼", "아쉬", "우울", "힘들", "외로", "그리워", "안타까", "속상", "눈물"],
    "surprised": ["헐", "대박", "진짜", "설마", "우와", "깜짝", "놀라", "신기", "어머"],
    "excited": ["완전", "너무", "최고", "짱", "미쳤", "대단", "엄청"],
    "thinking": ["흠", "글쎄", "모르", "생각", "고민", "궁금", "왜", "어떻"],
    "calm": ["그래", "응", "맞아", "알겠", "그렇구나", "편하", "조용"],
    "worried": ["걱정", "불안", "무서", "두려", "조심", "위험"],
    "angry": ["짜증", "화나", "싫어", "나빠", "미워", "열받"],
    "love": ["사랑", "좋아해", "예뻐", "귀여", "설레", "두근"],
    "shy": ["부끄", "쑥스", "민망", "창피", "수줍"],
}

def detect_emotion(text):
    """텍스트에서 감정 키워드를 찾아 태그 반환"""
    scores = {}
    for emo, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[emo] = score
    if scores:
        return max(scores, key=scores.get)
    return random.choice(["happy", "calm", "surprised", "thinking", "excited"])

def to_banmal(text):
    """존댓말 → 반말 변환"""
    replacements = [
        (r'합니다[.]?', '해.'), (r'입니다[.]?', '이야.'), (r'됩니다[.]?', '돼.'),
        (r'습니다[.]?', '어.'), (r'됩니까[?]?', '돼?'), (r'합니까[?]?', '해?'),
        (r'하세요[.]?', '해.'), (r'세요[.]?', '해.'), (r'드릴게요[.]?', '줄게.'),
        (r'드립니다[.]?', '줄게.'), (r'에요[.]', '야.'), (r'에요[?]', '야?'),
        (r'이에요[.]', '이야.'), (r'이에요[?]', '이야?'),
        (r'거예요[.]', '거야.'), (r'거예요[?]', '거야?'),
        (r'는데요[.]', '는데.'), (r'는데요[?]', '는데?'),
        (r'을까요[?]?', '을까?'), (r'ㄹ까요[?]?', 'ㄹ까?'),
        (r'네요[.]', '네.'), (r'네요[?]', '네?'), (r'네요[!]', '네!'),
        (r'죠[?]', '지?'), (r'죠[.]', '지.'),
        (r'어요[.]', '어.'), (r'어요[?]', '어?'), (r'어요[!]', '어!'),
        (r'나요[?]', '나?'), (r'나요[.]', '나.'),
        (r'군요[.]', '구나.'), (r'군요[!]', '구나!'),
        (r'ㅂ니다[.]?', '어.'), (r'겠습니다[.]?', '겠어.'),
        (r'주세요[.]?', '줘.'), (r'보세요[.]?', '봐.'),
        (r'으세요[.]?', '어.'), (r'셨어요[.]?', '했어.'),
        (r'하셨[어나]', '했'), (r'드릴까요[?]?', '줄까?'),
        (r'십시오[.]?', '해.'), (r'시겠어요[?]?', '할래?'),
    ]
    result = text
    for pat, repl in replacements:
        result = re.sub(pat, repl, result)
    return result

def is_good_response(text):
    """품질 필터"""
    if len(text) < 5 or len(text) > 200:
        return False
    if any(x in text for x in ['http', 'www.', '.com', '.kr']):
        return False
    if text.count('\n') > 3:
        return False
    return True

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_persona_chat():
    """NLPBada/korean-persona-chat-dataset 로드"""
    log("NLPBada/korean-persona-chat-dataset 다운로드 중...")
    from datasets import load_dataset
    ds = load_dataset("NLPBada/korean-persona-chat-dataset", split="train")
    log(f"  {len(ds)}건 로드됨")

    results = []
    for item in ds:
        # 데이터 구조 확인
        if 'dialog' in item:
            dialog = item['dialog']
        elif 'conversation' in item:
            dialog = item['conversation']
        elif 'text' in item:
            # 텍스트에서 대화 추출
            text = item['text']
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) >= 2:
                for i in range(0, len(lines) - 1, 2):
                    user_msg = lines[i].strip()
                    asst_msg = lines[i+1].strip() if i+1 < len(lines) else None
                    if asst_msg and is_good_response(asst_msg):
                        asst_msg = to_banmal(asst_msg)
                        emo = detect_emotion(asst_msg)
                        results.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": f"[감정:{emo}] {asst_msg}"}
                            ]
                        })
            continue
        else:
            # 키 출력해서 구조 파악
            keys = list(item.keys())
            if not results:
                log(f"  알 수 없는 데이터 구조: {keys}")
                log(f"  샘플: {dict(list(item.items())[:3])}")

            # persona/chat 컬럼 시도
            chat = None
            for k in ['chat', 'dialogue', 'utterance', 'response', 'answer']:
                if k in item:
                    chat = item[k]
                    break

            if chat is None:
                # 첫 번째/두 번째 컬럼을 user/assistant로 사용
                vals = list(item.values())
                if len(vals) >= 2:
                    user_msg = str(vals[0]).strip()
                    asst_msg = str(vals[1]).strip()
                    if is_good_response(asst_msg):
                        asst_msg = to_banmal(asst_msg)
                        emo = detect_emotion(asst_msg)
                        results.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": f"[감정:{emo}] {asst_msg}"}
                            ]
                        })
                continue

            if isinstance(chat, list) and len(chat) >= 2:
                for i in range(0, len(chat) - 1, 2):
                    user_msg = str(chat[i]).strip()
                    asst_msg = str(chat[i+1]).strip()
                    if is_good_response(asst_msg):
                        asst_msg = to_banmal(asst_msg)
                        emo = detect_emotion(asst_msg)
                        results.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": f"[감정:{emo}] {asst_msg}"}
                            ]
                        })

    log(f"  persona-chat 변환 완료: {len(results)}건")
    return results

def load_role_playing():
    """huggingface-KREW/korean-role-playing 로드"""
    log("huggingface-KREW/korean-role-playing 다운로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("huggingface-KREW/korean-role-playing", split="train")
        log(f"  {len(ds)}건 로드됨")
    except Exception as e:
        log(f"  로드 실패: {e}")
        return []

    results = []
    for item in ds:
        # 구조 파악
        if not results and len(results) == 0:
            log(f"  데이터 키: {list(item.keys())}")

        # 대화 추출 시도
        user_msg = None
        asst_msg = None

        for uk in ['instruction', 'input', 'question', 'user', 'human']:
            if uk in item and item[uk]:
                user_msg = str(item[uk]).strip()
                break

        for ak in ['output', 'response', 'answer', 'assistant', 'bot']:
            if ak in item and item[ak]:
                asst_msg = str(item[ak]).strip()
                break

        if user_msg is None or asst_msg is None:
            # conversation 형식 시도
            for ck in ['conversation', 'conversations', 'dialog', 'dialogue', 'messages']:
                if ck in item and item[ck]:
                    conv = item[ck]
                    if isinstance(conv, list) and len(conv) >= 2:
                        for i in range(len(conv) - 1):
                            c1 = conv[i]
                            c2 = conv[i+1]
                            if isinstance(c1, dict) and isinstance(c2, dict):
                                u = c1.get('content', c1.get('value', c1.get('text', '')))
                                a = c2.get('content', c2.get('value', c2.get('text', '')))
                                r1 = c1.get('role', c1.get('from', ''))
                                r2 = c2.get('role', c2.get('from', ''))
                                if r1 in ['user', 'human'] and r2 in ['assistant', 'gpt', 'bot']:
                                    user_msg = str(u).strip()
                                    asst_msg = str(a).strip()
                                    break
                    break

        if user_msg and asst_msg and is_good_response(asst_msg):
            asst_msg = to_banmal(asst_msg)
            emo = detect_emotion(asst_msg)
            results.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"[감정:{emo}] {asst_msg}"}
                ]
            })

    log(f"  role-playing 변환 완료: {len(results)}건")
    return results

def main():
    log("=" * 60)
    log("HuggingFace 한국어 대화 → 시온 학습 데이터 변환")
    log("=" * 60)

    random.seed(42)
    all_data = []

    # 1. Persona chat
    try:
        data1 = load_persona_chat()
        all_data.extend(data1)
    except Exception as e:
        log(f"persona-chat 실패: {e}")

    # 2. Role playing
    try:
        data2 = load_role_playing()
        all_data.extend(data2)
    except Exception as e:
        log(f"role-playing 실패: {e}")

    if not all_data:
        log("변환된 데이터 없음!")
        sys.exit(1)

    # 셔플
    random.shuffle(all_data)

    # Train/eval 분리 (90/10)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    # 저장
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"학습 데이터 저장: {OUTPUT_TRAIN} ({len(train_data)}건)")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"평가 데이터 저장: {OUTPUT_EVAL} ({len(eval_data)}건)")

    # 기존 데이터와 병합
    if os.path.exists(EXISTING_DATA):
        log(f"기존 데이터 로드: {EXISTING_DATA}")
        existing = []
        with open(EXISTING_DATA, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))
        log(f"  기존: {len(existing)}건")

        combined = existing + train_data
        random.shuffle(combined)

        with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
            for item in combined:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"병합 데이터 저장: {OUTPUT_COMBINED} ({len(combined)}건)")

    # 샘플 출력
    log("\n=== 샘플 데이터 (3건) ===")
    for i, item in enumerate(all_data[:3]):
        user = item["messages"][1]["content"] if len(item["messages"]) > 1 else "N/A"
        asst = item["messages"][-1]["content"]
        log(f"  [{i+1}] User: {user[:50]}...")
        log(f"       Sion: {asst[:80]}...")

    log("=" * 60)
    log(f"변환 완료! 총 {len(all_data)}건 (학습 {len(train_data)} + 평가 {len(eval_data)})")
    log("=" * 60)

if __name__ == "__main__":
    main()
