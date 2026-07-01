# -*- coding: utf-8 -*-
import requests, json

system = """너는 "시온(sion)"이라는 이름의 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터, 항상 반말로 대화해. 존댓말 절대 금지
- 이모티콘 절대 쓰지 마. 말투로 감정 표현

[감정 태그 규칙]
응답 맨 앞에 반드시 [감정:태그] 붙여. Live2D 표정 애니메이션에 사용돼.
태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy

[방송 채팅 응답 규칙]
- 반드시 1문장으로 답해. 최대 40자. 절대 2문장 이상 쓰지 마
- 시청자 닉네임을 자연스럽게 불러"""

user = "안녕! 너는 누구야?\n\n(이 채팅을 보낸 시청자 닉네임: 테스트유저)"

r = requests.post("http://localhost:11434/api/chat", json={
    "model": "sion",
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    "stream": False,
    "options": {"temperature": 0.8, "num_predict": 300},
}, timeout=60)

data = r.json()
msg = data.get("message", {}).get("content", "")
with open(r"C:\Users\thtgg\workspace2\ai_vtuber\training\sion_broadcast_test.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print(f"Length: {len(msg)} chars, saved.")
