# -*- coding: utf-8 -*-
import requests

prompts = [
    "시온아 안녕! 오늘 기분 어때?",
    "요즘 어떤 노래 추천해줄 수 있어?",
    "나 오늘 시험 망했어 ㅠㅠ",
    "시온이는 좋아하는 음식 있어?",
    "오늘 방송에서 뭐 할 거야?",
    "나랑 친구 해줄래?",
    "시온이는 몇 살이야?",
    "잘 자! 내일 또 올게",
]

results = []
for p in prompts:
    r = requests.post("http://localhost:8002/chat", json={"message": p, "user": "test_user"}, timeout=60)
    d = r.json()
    line = "Q: {}\nA: [{}] {}".format(p, d["emotion"], d["reply"])
    results.append(line)

with open(r"C:\Users\thtgg\workspace2\ai_vtuber\training\chat_test.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(results))
print("done")
