from urllib.parse import urlparse, parse_qs, unquote
import json

# Simulate what the API returns
result_str = '[{"file": "/v1/audio?path=C%3A%5CUsers%5Cthtgg%5Cworkspace2%5Cai_vtuber%5CACE-Step-1.5%5C.cache%5Cacestep%5Ctmp%5Capi_audio%5Ceb2eb91a-c39b-c885-afed-72d531fbfa46.mp3"}]'
result_parsed = json.loads(result_str)
first = result_parsed[0]
remote_path = first.get("file", "")

print(f"remote_path: {remote_path!r}")

if remote_path.startswith("/v1/audio"):
    parsed = urlparse(remote_path)
    qs = parse_qs(parsed.query)
    actual_path = qs.get("path", [remote_path])[0]
    actual_path = unquote(actual_path)
else:
    actual_path = remote_path

print(f"actual_path: {actual_path!r}")

# Test aiohttp URL construction
import aiohttp
from yarl import URL
base = "http://localhost:8006/v1/audio"
url = URL(base).with_query({"path": actual_path})
print(f"final_url: {url}")
