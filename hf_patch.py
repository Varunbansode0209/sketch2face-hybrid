import base64
import json
import urllib.request

def get_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

config_path = r"c:\Users\ADMIN\OneDrive\Desktop\sketch2face-hybrid-backup\sketch2face-web\backend\app\core\config.py"
docker_path = r"c:\Users\ADMIN\OneDrive\Desktop\sketch2face-hybrid-backup\sketch2face-web\backend\Dockerfile"

payload = {
    "commit_message": "Force API patch for Vercel CORS via Urllib",
    "operations": [
        {"operation": "add", "path": "sketch2face-web/backend/app/core/config.py", "encoding": "base64", "content": get_b64(config_path)},
        {"operation": "add", "path": "sketch2face-web/backend/Dockerfile", "encoding": "base64", "content": get_b64(docker_path)}
    ]
}

req = urllib.request.Request(
    "https://huggingface.co/api/spaces/VarunB2/sketch2face-api/commit/main",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer YOUR_HF_TOKEN_HERE",
        "Content-Type": "application/json"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        print(resp.status, resp.read().decode("utf-8"))
except Exception as e:
    print(e)
    if hasattr(e, "read"):
        print(e.read().decode("utf-8"))
