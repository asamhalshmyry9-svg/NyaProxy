import json

from httpx import Headers

from nya.services.gateway_logs import GatewayLogStore, resolve_gateway_log_dir
from tests.unit.core_helpers import make_request


def prepared_request():
    request = make_request(
        method="POST",
        headers={
            "authorization": "Bearer local-proxy-key",
            "content-type": "application/json",
            "x-client-detail": "preserve-me",
        },
        content=b'{"model":"gemini","messages":[{"role":"user","content":"hi"}]}',
    )
    request.api_name = "gemini"
    request.api_key = "upstream-secret-key"
    request.attempts = 2
    request.url = "https://upstream.test/v1/chat/completions"
    request.headers = Headers(
        {
            "authorization": "Bearer upstream-secret-key",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
    )
    request.content = (
        b'{"model":"gemini","stream":true,"messages":[{"role":"user","content":"hi"}]}'
    )
    return request


def test_gateway_log_keeps_full_request_headers_bodies_and_response(tmp_path):
    store = GatewayLogStore(tmp_path / "gateway-logs", retention=100)
    entry = store.begin(prepared_request())
    entry.record_response_head(
        200,
        Headers(
            {
                "content-type": "text/event-stream",
                "x-upstream-detail": "complete",
            }
        ),
    )
    entry.write_response_chunk(b'data: {"choices":[{"delta":{"content":"O"}}]}\n\n')
    entry.write_response_chunk(b'data: {"choices":[{"delta":{"content":"K"}}]}\n\n')
    entry.write_response_chunk(b"data: [DONE]\n\n")
    entry.finish()

    request_doc = json.loads(entry.request_path.read_text(encoding="utf-8"))
    meta = json.loads(entry.meta_path.read_text(encoding="utf-8"))
    index = json.loads((store.directory / "gateway-index.json").read_text("utf-8"))

    assert ["authorization", "Bearer local-proxy-key"] in request_doc["client"][
        "headers"
    ]
    assert ["authorization", "Bearer upstream-secret-key"] in request_doc["upstream"][
        "headers"
    ]
    assert request_doc["upstream"]["body"]["value"]["model"] == "gemini"
    assert entry.client_raw_path.read_bytes() != entry.request_raw_path.read_bytes()
    assert b'"stream":true' in entry.request_raw_path.read_bytes()
    assert entry.response_raw_path.read_bytes().endswith(b"data: [DONE]\n\n")
    assert entry.response_text_path.read_text("utf-8").endswith("data: [DONE]\n\n")
    assert meta["completed"] is True
    assert meta["ok"] is True
    assert meta["statusCode"] == 200
    assert meta["model"] == "gemini"
    assert meta["streamRequested"] is True
    assert meta["responseBytes"] == entry.response_raw_path.stat().st_size
    assert meta["responseChunks"] == 3
    assert meta["contentEncoding"] == ""
    assert index == [meta]


def test_gateway_log_preserves_encoded_response_without_claiming_it_is_text(tmp_path):
    store = GatewayLogStore(tmp_path / "gateway-logs")
    entry = store.begin(prepared_request())
    entry.record_response_head(
        200,
        Headers(
            {
                "content-type": "text/event-stream",
                "content-encoding": "gzip",
            }
        ),
    )
    compressed = b"\x1f\x8b\x08\x00not-decoded"
    entry.write_response_chunk(compressed)
    entry.finish()

    meta = json.loads(entry.meta_path.read_text("utf-8"))
    assert entry.response_raw_path.read_bytes() == compressed
    assert not entry.response_text_path.exists()
    assert meta["contentEncoding"] == "gzip"
    assert meta["responseBytes"] == len(compressed)


def test_gateway_log_retains_only_the_newest_one_hundred_attempts(tmp_path):
    store = GatewayLogStore(tmp_path / "gateway-logs", retention=100)
    for _ in range(101):
        entry = store.begin(prepared_request())
        entry.finish()

    index = json.loads((store.directory / "gateway-index.json").read_text("utf-8"))
    assert len(index) == 100
    assert [entry["id"] for entry in index] == list(range(2, 102))
    assert not list(store.directory.glob("gateway-1.*"))
    assert list(store.directory.glob("gateway-101.*"))


def test_relative_gateway_log_path_is_next_to_configuration(tmp_path):
    config = tmp_path / "data" / "config.yaml"
    assert resolve_gateway_log_dir(str(config), "gateway-logs") == (
        config.parent / "gateway-logs"
    )
