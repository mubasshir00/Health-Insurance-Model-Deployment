import hmac
import hashlib
import json
from .settings import settings

HEADER_NAME = "x-ml-signature"

def sign_payload(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":")).encode()
    mac = hmac.new(settings.HMAC_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"hmac-sha256={mac}"
