import json
from types import SimpleNamespace

import pytest
from httpx import Headers

from nya.core.request import RequestExecutor
from tests.unit.core_helpers import (
    CoreConfig,
    FakeHttpxResponse,
    FakeStreamContext,
    make_request,
)


@pytest.mark.asyncio
async def test_request_executor_normal_streaming_proxy_and_metrics_paths(monkeypatch):
    config = CoreConfig()
    metrics = SimpleNamespace(
        requests=[],
        responses=[],
        record_request=lambda api, key, path=None: metrics.requests.append(
            (api, key, path)
        ),
        record_response=lambda api, key, status, elapsed, path=None: (
            metrics.responses.append((api, key, status, path))
        ),
    )
    executor = RequestExecutor(config, metrics)
    request = make_request()
    request.api_name = "mock"
    request.api_key = "key-a"
    request.url = "https://upstream.test/v1"
    request._rate_limited = False

    async def fake_execute_request(request, timeout):
        return FakeHttpxResponse(
            [b"hello"], headers={"content-type": "text/plain", "content-length": "5"}
        )

    executor.execute_request = fake_execute_request
    response = await executor.execute(request)
    assert response.body == b"hello"
    # The path travels with the metric so the dashboard can group by it.
    assert metrics.requests == [("mock", "key-a", request.trail_path)]
    assert metrics.responses[0][3] == request.trail_path
    assert metrics.responses[0][:3] == ("mock", "key-a", 200)

    async def fake_stream(request, timeout):
        return FakeHttpxResponse(
            [b"data: ok\n\n"], headers={"content-type": "text/event-stream"}
        )

    executor.execute_request = fake_stream
    streaming = await executor.execute(request)
    assert len(metrics.responses) == 1
    assert [chunk async for chunk in streaming.body_iterator] == [b"data: ok\n\n"]
    assert metrics.responses[-1][:3] == ("mock", "key-a", 200)

    async def boom(request, timeout):
        raise RuntimeError("network")

    executor.execute_request = boom
    with pytest.raises(RuntimeError):
        await executor.execute(request)
    assert metrics.responses[-1][:3] == ("mock", "key-a", 0)

    await executor.close()


@pytest.mark.asyncio
async def test_request_executor_records_exact_stream_before_forwarding(tmp_path):
    config = CoreConfig()
    config.config_path = str(tmp_path / "config.yaml")
    config.get_gateway_logging_config = lambda: {
        "enabled": True,
        "directory": "gateway-logs",
        "retention": 100,
    }
    executor = RequestExecutor(config)
    request = make_request(
        method="POST",
        headers={"content-type": "application/json"},
        content=b'{"model":"gemini"}',
    )
    request.api_name = "gemini"
    request.api_key = "secret"
    request.url = "https://upstream.test/chat/completions"

    raw = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'

    async def fake_stream(request, timeout):
        return FakeHttpxResponse(
            [raw[:17], raw[17:]], headers={"content-type": "text/event-stream"}
        )

    executor.execute_request = fake_stream
    response = await executor.execute(request)
    assert b"".join([chunk async for chunk in response.body_iterator]) == raw

    log_dir = tmp_path / "gateway-logs"
    assert (log_dir / "gateway-1.response.raw").read_bytes() == raw
    request_doc = json.loads((log_dir / "gateway-1.request.json").read_text("utf-8"))
    assert ["accept-encoding", "identity"] in request_doc["upstream"]["headers"]
    meta = json.loads((log_dir / "gateway-1.meta.json").read_text("utf-8"))
    assert meta["completed"] is True
    assert meta["responseChunks"] == 2
    assert meta["responseBytes"] == len(raw)
    await executor.close()


@pytest.mark.asyncio
async def test_request_executor_records_stream_failure(tmp_path):
    config = CoreConfig()
    config.config_path = str(tmp_path / "config.yaml")
    config.get_gateway_logging_config = lambda: {
        "enabled": True,
        "directory": "gateway-logs",
        "retention": 100,
    }
    executor = RequestExecutor(config)
    request = make_request(method="POST", content=b'{"stream":true}')
    request.api_name = "gemini"
    request.api_key = "secret"
    request.url = "https://upstream.test/chat/completions"

    async def broken_stream(request, timeout):
        return FakeHttpxResponse(
            [], headers={"content-type": "text/event-stream"}, fail=True
        )

    executor.execute_request = broken_stream
    response = await executor.execute(request)
    with pytest.raises(RuntimeError, match="stream failed"):
        [chunk async for chunk in response.body_iterator]

    meta = json.loads(
        (tmp_path / "gateway-logs" / "gateway-1.meta.json").read_text("utf-8")
    )
    assert meta["completed"] is True
    assert meta["ok"] is False
    assert meta["error"] == "RuntimeError: stream failed"
    assert meta["responseBytes"] == 0
    await executor.close()


@pytest.mark.asyncio
async def test_request_executor_execute_request_and_cleanup_on_processing_error():
    config = CoreConfig()
    executor = RequestExecutor(config)
    request = make_request()
    request.url = "https://upstream.test/v1"

    class Client:
        def stream(self, **kwargs):
            response = FakeHttpxResponse(
                [b"broken"],
                headers={"content-type": "text/plain", "content-length": "6"},
                fail=True,
            )
            return FakeStreamContext(response)

    executor.client = Client()
    response = await executor.execute_request(request, executor._get_timeout("mock"))
    stream_ctx = response._stream_ctx

    with pytest.raises(RuntimeError):
        await executor.process_response(response)
    assert stream_ctx.closed is True


@pytest.mark.asyncio
async def test_request_executor_close_is_noop_without_client():
    executor = object.__new__(RequestExecutor)
    executor.client = None

    await executor.close()


@pytest.mark.asyncio
async def test_response_headers_preserve_duplicates_and_strip_hop_by_hop():
    config = CoreConfig()
    executor = RequestExecutor(config)
    upstream = FakeHttpxResponse(
        [b"ok"],
        headers=[
            ("content-type", "text/plain"),
            ("content-length", "2"),
            ("set-cookie", "a=1; Path=/"),
            ("set-cookie", "b=2; Path=/"),
            ("connection", "keep-alive, x-private"),
            ("keep-alive", "timeout=5"),
            ("x-private", "remove-me"),
        ],
    )

    response = await executor.handle_normal_response(upstream)
    raw = response.raw_headers
    assert [value for name, value in raw if name == b"set-cookie"] == [
        b"a=1; Path=/",
        b"b=2; Path=/",
    ]
    assert b"connection" not in {name for name, _ in raw}
    assert b"keep-alive" not in {name for name, _ in raw}
    assert b"x-private" not in {name for name, _ in raw}
    await executor.close()


def test_request_executor_timeout_and_proxy_client_options(monkeypatch):
    config = CoreConfig()
    config.proxy_enabled = True
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def aclose(self):
            captured["closed"] = True

    monkeypatch.setattr("nya.core.request.httpx.AsyncClient", FakeClient)

    executor = RequestExecutor(config)

    assert captured["proxy"] == "http://proxy.test"
    assert executor._get_timeout("mock").read == pytest.approx(4.75)


def test_upstream_encoding_is_not_negotiated_on_behalf_of_client():
    without_capability = RequestExecutor._prepare_request_headers(
        Headers({"content-type": "application/json"})
    )
    assert ("accept-encoding", "identity") in without_capability

    client_supports_gzip = RequestExecutor._prepare_request_headers(
        Headers({"accept-encoding": "gzip, deflate"})
    )
    assert ("accept-encoding", "gzip, deflate") in client_supports_gzip
    assert ("accept-encoding", "identity") not in client_supports_gzip
