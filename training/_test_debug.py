# -*- coding: utf-8 -*-
"""Chat module을 거쳐서 실제 어떤 응답이 오는지 디버그"""
import requests, json, time

# Test 1: Ollama 직접 (broadcast system prompt)
print("=== Test 1: Ollama 직접 ===")
t0 = time.time()
r = requests.post("http://localhost:11434/api/chat", json={
    "model": "sion",
    "messages": [
        {"role": "system", "content": "너는 시온이야. 반말로 1문장 답해. 앞에 [감정:태그] 붙여."},
        {"role": "user", "content": "안녕!"},
    ],
    "stream": False,
    "options": {"temperature": 0.8, "num_predict": 300},
}, timeout=60)
data = r.json()
ollama_resp = data.get("message", {}).get("content", "")
print(f"  Time: {time.time()-t0:.1f}s")
print(f"  Response: {ollama_resp}")
print(f"  Length: {len(ollama_resp)}")

# Test 2: Chat module (broadcast mode)
print("\n=== Test 2: Chat Module (broadcast) ===")
t0 = time.time()
r2 = requests.post("http://localhost:8002/chat", json={
    "message": "안녕!",
    "mode": "broadcast",
    "viewer_name": "테스트",
}, timeout=120)
print(f"  Time: {time.time()-t0:.1f}s")
print(f"  Status: {r2.status_code}")
print(f"  Response: {json.dumps(r2.json(), ensure_ascii=False)}")

# Test 3: Chat module (broadcast mode)
print("\n=== Test 3: Chat Module (broadcast) ===")
t0 = time.time()
r3 = requests.post("http://localhost:8002/chat", json={
    "message": "안녕!",
    "mode": "broadcast",
}, timeout=120)
print(f"  Time: {time.time()-t0:.1f}s")
print(f"  Status: {r3.status_code}")
print(f"  Response: {json.dumps(r3.json(), ensure_ascii=False)}")
