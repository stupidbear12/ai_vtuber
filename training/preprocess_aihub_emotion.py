# -*- coding: utf-8 -*-
"""
AI Hub 감정 대화 XLSX → 시온 학습 데이터 변환
구조: Col A=dialog marker('S'=새 대화), Col B=발화, Col C=감정(7종)
"""
import json, os, random, re, time

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(TRAINING_DIR, "한국어_연속적_대화_데이터셋.xlsx")
OUTPUT_TRAIN = os.path.join(TRAINING_DIR, "sion_aihub_emotion.jsonl")
OUTPUT_EVAL = os.path.join(TRAINING_DIR, "sion_aihub_emotion_eval.jsonl")
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

# AI Hub 감정 → 시온 감정 태그 매핑
EMOTION_MAP = {
    "행복": ["happy", "excited", "love"],
    "중립": ["calm", "thinking"],
    "슬픔": ["sad", "worried"],
    "공포": ["worried", "surprised"],
    "혐오": ["angry"],
    "분노": ["angry"],
    "놀람": ["surprised", "excited"],
}

def map_emotion(korean_emotion):
    tags = EMOTION_MAP.get(korean_emotion, ["calm"])
    return random.choice(tags)

def to_banmal(text):
    """존댓말 → 반말 변환"""
    replacements = [
        (r'합니다[.]?', '해.'), (r'입니다[.]?', '이야.'), (r'됩니다[.]?', '돼.'),
        (r'습니다[.]?', '어.'), (r'됩니까[?]?', '돼?'), (r'합니까[?]?', '해?'),
        (r'하세요[.]?', '해.'), (r'드릴게요[.]?', '줄게.'),
        (r'드립니다[.]?', '줄게.'), (r'거예요', '거야'), (r'이에요', '이야'),
        (r'는데요', '는데'), (r'을까요', '을까'), (r'ㄹ까요', 'ㄹ까'),
        (r'네요', '네'), (r'죠\?', '지?'), (r'죠\.', '지.'),
        (r'어요', '어'), (r'나요', '나'), (r'군요', '구나'),
        (r'겠습니다', '겠어'), (r'주세요', '줘'), (r'보세요', '봐'),
        (r'으세요', '어'), (r'셨어', '했어'), (r'하셨', '했'),
        (r'십시오', '해'), (r'에요', '야'),
    ]
    result = text
    for pat, repl in replacements:
        result = re.sub(pat, repl, result)
    return result

def has_jondaenmal(text):
    """존댓말 포함 여부 검사"""
    patterns = ['습니다', '세요', '에요', '어요', '나요', '군요', '시오', '겠습']
    return any(p in text for p in patterns)

def is_good_response(text):
    if not text or len(text) < 3 or len(text) > 200:
        return False
    if any(x in text for x in ['http', 'www.', '.com', '@']):
        return False
    return True

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=" * 60)
    log("AI Hub 감정 대화 → 시온 학습 데이터 변환")
    log("=" * 60)
    random.seed(42)

    import openpyxl
    log(f"XLSX 로드: {INPUT_FILE}")
    wb = openpyxl.load_workbook(INPUT_FILE, read_only=True)
    ws = wb.active

    # 대화 그룹 파싱
    dialogs = []
    current_dialog = []
    row_count = 0

    for row in ws.iter_rows(min_row=3, values_only=True):  # skip header rows
        marker, utterance, emotion = row[0], row[1], row[2]
        if not utterance:
            continue
        row_count += 1

        if marker == 'S' and current_dialog:
            dialogs.append(current_dialog)
            current_dialog = []

        current_dialog.append({
            'text': str(utterance).strip(),
            'emotion': str(emotion).strip() if emotion else '중립'
        })

    if current_dialog:
        dialogs.append(current_dialog)

    log(f"총 {row_count}행 → {len(dialogs)}개 대화 그룹")

    # 대화 쌍 생성 (user → assistant)
    results = []
    emotion_stats = {}

    for dialog in dialogs:
        if len(dialog) < 2:
            continue

        # 연속 발화를 user/assistant 쌍으로 변환
        for i in range(0, len(dialog) - 1, 2):
            user_utt = dialog[i]['text']
            asst_utt = dialog[i + 1]['text']
            asst_emotion = dialog[i + 1]['emotion']

            if not is_good_response(asst_utt):
                continue

            # 존댓말 → 반말 변환
            if has_jondaenmal(asst_utt):
                asst_utt = to_banmal(asst_utt)

            # 감정 매핑
            emo_tag = map_emotion(asst_emotion)
            emotion_stats[emo_tag] = emotion_stats.get(emo_tag, 0) + 1

            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_utt},
                    {"role": "assistant", "content": f"[감정:{emo_tag}] {asst_utt}"}
                ]
            }
            results.append(entry)

        # 멀티턴 대화도 생성 (3~5턴)
        if len(dialog) >= 6:
            turns = []
            turns.append({"role": "system", "content": SYSTEM_PROMPT})
            turn_count = 0
            for j in range(0, min(len(dialog), 10) - 1, 2):
                u = dialog[j]['text']
                a = dialog[j + 1]['text']
                a_emo = dialog[j + 1]['emotion']
                if not is_good_response(a):
                    break
                if has_jondaenmal(a):
                    a = to_banmal(a)
                emo = map_emotion(a_emo)
                turns.append({"role": "user", "content": u})
                turns.append({"role": "assistant", "content": f"[감정:{emo}] {a}"})
                turn_count += 1
            if turn_count >= 3:
                results.append({"messages": turns})

    log(f"변환 완료: {len(results)}건")
    log(f"감정 분포: {dict(sorted(emotion_stats.items(), key=lambda x: -x[1]))}")

    # 셔플 + 분리
    random.shuffle(results)
    split = int(len(results) * 0.9)
    train_data = results[:split]
    eval_data = results[split:]

    # 저장
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"학습: {OUTPUT_TRAIN} ({len(train_data)}건)")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"평가: {OUTPUT_EVAL} ({len(eval_data)}건)")

    # 기존 데이터 병합
    if os.path.exists(EXISTING_DATA):
        log(f"기존 데이터 로드: {EXISTING_DATA}")
        existing = []
        with open(EXISTING_DATA, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))
        log(f"기존: {len(existing)}건")

        combined = existing + train_data
        random.shuffle(combined)

        with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
            for item in combined:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"병합: {OUTPUT_COMBINED} ({len(combined)}건)")

    # 샘플
    log("\n=== 샘플 (5건) ===")
    for i, item in enumerate(results[:5]):
        msgs = item["messages"]
        user = msgs[1]["content"][:50] if len(msgs) > 1 else "N/A"
        asst = msgs[-1]["content"][:80]
        log(f"  [{i+1}] U: {user}")
        log(f"       A: {asst}")

    log("=" * 60)
    log(f"완료! 총 {len(results)}건")
    log("=" * 60)

if __name__ == "__main__":
    main()
