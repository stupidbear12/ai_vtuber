import requests, json
r = requests.post("http://localhost:11434/api/generate", json={"model":"sion","prompt":"안녕! 너는 누구야?","stream":False}, timeout=120)
data = r.json()
with open(r"C:\Users\thtgg\workspace2\ai_vtuber\training\sion_api_test.txt", "w", encoding="utf-8") as f:
    f.write(data.get("response","NO RESPONSE"))
print("Done:", data.get("response","")[:200])
