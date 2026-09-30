"""Image transport shared by the HTTP and Realtime APIs."""

import base64
import urllib.request


def validate_image_detail(detail="auto"):
    if detail not in ("auto", "low", "high"):
        raise ValueError("Image detail must be auto, low, or high.")


def image_bytes_from_url(url, *, max_bytes=50 * 1024**2, inline_only=False):
    if not isinstance(url, str):
        raise ValueError("image_url must be a string.")
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if not header.startswith("data:image/") or not header.endswith(";base64") or not payload:
            raise ValueError("Image data URL must be base64 encoded.")
        if inline_only and header not in ("data:image/png;base64", "data:image/jpeg;base64"):
            raise ValueError("Realtime images must be inline PNG or JPEG.")
        if len(payload) > 4 * ((max_bytes + 2) // 3):
            raise ValueError("Image exceeds the byte limit.")
        raw = base64.b64decode(payload, validate=True)
    else:
        if inline_only or not url.startswith(("http://", "https://")):
            raise ValueError("Supply an inline image." if inline_only else "image_url must be a data URL or an HTTP(S) URL.")
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError("Image is empty or exceeds the byte limit.")
    return raw
