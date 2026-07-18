# -*- coding: utf-8 -*-
import requests

msgs = [
    "안녕! 오늘 기분 어때?",
    "요즘 좋아하는 노래 있어?",
    "너 진짜 예쁘다",
    "DJ로서 가장 좋아하는 장르가 뭐야?",
    "오늘 비 오는데 우산 안 가져왔어",
]

results = []
for m in msgs:
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": "sion",
        "messages": [{"role": "user", "content": m}],
        "stream": False
    }, timeout=30)
    d = r.json()
    content = d["message"]["content"]
    line = "Q: " + m + "\nA: " + content
    results.append(line)

output = "\n\n".join(results)
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write(output)
print("Done")
