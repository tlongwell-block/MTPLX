"""OpenAI image transport contracts, without loading model weights."""

import base64
import io

import pytest

from mtplx.vision.media import image_bytes_from_url, validate_image_detail


@pytest.mark.parametrize("detail", ["auto", "low", "high"])
def test_standard_image_detail(detail):
    validate_image_detail(detail)


@pytest.mark.parametrize("detail", [None, 1, "original", "medium"])
def test_invalid_image_detail(detail):
    with pytest.raises(ValueError):
        validate_image_detail(detail)


@pytest.mark.parametrize("mime", ["png", "jpeg"])
def test_realtime_data_url(mime):
    data = b"image transport fixture"
    url = f"data:image/{mime};base64," + base64.b64encode(data).decode()
    assert image_bytes_from_url(url, inline_only=True) == data
    assert image_bytes_from_url(url) == data


@pytest.mark.parametrize("url", [
    "data:image/png;base64,???", "data:text/plain;base64,YQ==",
    "data:image/png,YQ==", "data:image/png;base64,", "file:///image.png",
    "ftp://example.com/image.png", {"url": "data:image/png;base64,YQ=="},
])
def test_invalid_image_url(url):
    with pytest.raises(ValueError):
        image_bytes_from_url(url)


def test_remote_images_are_http_only_and_bounded(monkeypatch):
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b"abcd")

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    assert image_bytes_from_url("https://example.com/image.png", max_bytes=4) == b"abcd"
    with pytest.raises(ValueError, match="limit"):
        image_bytes_from_url("http://example.com/image.png", max_bytes=3)
    with pytest.raises(ValueError, match="inline"):
        image_bytes_from_url("https://example.com/image.png", inline_only=True)
    assert len(calls) == 2


def test_oversized_data_url_is_rejected_before_decoding():
    with pytest.raises(ValueError, match="limit"):
        image_bytes_from_url("data:image/png;base64," + "YQ==" * 100, max_bytes=4)


def test_http_image_validation_and_auth_do_not_load_or_schedule_model(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from mtplx.frankie.completions import attach_routes

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid or unauthenticated requests must not fetch images or use the model.")

    monkeypatch.setattr("mtplx.vision.media.image_bytes_from_url", unexpected)
    app = FastAPI()
    with ThreadPoolExecutor(max_workers=1) as executor:
        attach_routes(app, unexpected, executor, "test-access-token")
        with TestClient(app) as client:
            image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            body = {"model": "Frankie", "messages": [{"role": "user", "content": [image]}]}
            assert client.post("/v1/chat/completions", json=body).status_code == 401
            auth = {"Authorization": "Bearer test-access-token"}
            assert client.get("/v1/models").status_code == 401
            assert client.get("/v1/models", headers=auth).json()["data"][0]["id"] == "Frankie"
            for part in (
                "image", {"type": "input_image"}, {"type": "text", "text": 5},
                {"type": "image_url", "image_url": "https://example.com/image.png"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "invalid"}},
            ):
                body["messages"][0]["content"] = [part]
                response = client.post("/v1/chat/completions", headers=auth, json=body)
                assert response.status_code == 400, response.text
                assert response.json()["error"]["type"] == "invalid_request_error"
