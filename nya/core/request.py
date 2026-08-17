"""
Simplified request executor focused on HTTP execution only.
"""

import logging
import time
from typing import TYPE_CHECKING, Optional, Union

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from ..utils.formatting import format_elapsed_time, json_safe_dumps
from ..utils.redaction import redact_sensitive_data
from ..services.gateway_logs import GatewayLogEntry, GatewayLogStore
from .streaming import (
    _HOP_BY_HOP_HEADERS,
    apply_response_headers,
    detect_streaming_content,
    handle_streaming_response,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..common.models import ProxyRequest
    from ..config.manager import ConfigManager
    from ..services.metrics import MetricsCollector


class RequestExecutor:
    """
    Simple request executor that does one thing well: execute HTTP requests.

    All retry logic is moved to the queue system. This class focuses
    purely on making HTTP calls and returning responses.
    """

    def __init__(
        self,
        config: "ConfigManager",
        metrics_collector: Optional["MetricsCollector"] = None,
    ):
        """
        Initialize the simple request executor.
        """
        self.config = config
        self.client = self._create_client()
        self.metrics_collector = metrics_collector
        self.gateway_logs = GatewayLogStore.from_config(config)
        self._close_callbacks = []

    def _create_client(self) -> httpx.AsyncClient:
        """
        Create HTTP client with optimized settings.
        """
        proxy_timeout = self.config.get_default_timeout()
        timeout = self._get_timeout()

        client_kwargs = {
            "follow_redirects": True,
            "timeout": timeout,
            "limits": httpx.Limits(
                max_connections=2000,
                max_keepalive_connections=500,
                keepalive_expiry=min(120.0, proxy_timeout),
            ),
        }

        # Add proxy support if configured
        if self.config.get_proxy_enabled():
            proxy_address = self.config.get_proxy_address()
            if proxy_address:
                client_kwargs["proxy"] = proxy_address

        return httpx.AsyncClient(**client_kwargs)

    async def execute(
        self, request: "ProxyRequest"
    ) -> Union[Response, JSONResponse, StreamingResponse]:
        """
        Execute a single HTTP request.

        Any retries are handled by the queue system.
        """
        actual_start_time = time.time()
        api_name = request.api_name
        # Materialize the exact header set that will be sent upstream before
        # the audit entry is created. Re-preparing this set in
        # ``execute_request`` is idempotent.
        request.headers = httpx.Headers(self._prepare_request_headers(request.headers))
        gateway_entry = self.gateway_logs.begin(request) if self.gateway_logs else None

        track = bool(self.metrics_collector)

        if track:
            self.metrics_collector.record_request(
                api_name, request.api_key, request.trail_path
            )

        logger.debug(f"[Request] Method: {request.method.upper()}, URL: {request.url}")

        # Execute HTTP request. If it fails before producing a response, still
        # record an outcome (status 0) so the active-request gauge is balanced
        # and the failure is counted — otherwise active leaks upward forever.
        try:
            response = await self.execute_request(request, self._get_timeout(api_name))
        except BaseException as error:
            if gateway_entry:
                gateway_entry.finish(error)
            if track:
                self.metrics_collector.record_response(
                    api_name,
                    request.api_key,
                    0,
                    time.time() - actual_start_time,
                    request.trail_path,
                )
            raise

        if gateway_entry:
            gateway_entry.record_response_head(
                response.status_code,
                response.headers,
                http_version=getattr(response, "http_version", ""),
                reason_phrase=getattr(response, "reason_phrase", ""),
            )

        # Log request/response details on error response
        if response.status_code >= 400:
            logger.debug(f"[Request] Content: {json_safe_dumps(request.content)}")

        logger.debug(
            f"[Request] Headers: {json_safe_dumps(redact_sensitive_data(request.headers))}"
        )
        logger.debug(
            f"[Response] Headers: {json_safe_dumps(redact_sensitive_data(response.headers))}"
        )

        logger.debug(
            f"[Response] URL: {request.url}, Status: {response.status_code} "
            f"({format_elapsed_time(time.time() - actual_start_time)})"
        )

        if detect_streaming_content(response.headers):
            streaming = await handle_streaming_response(
                response,
                on_chunk=(
                    gateway_entry.write_response_chunk if gateway_entry else None
                ),
                on_error=(gateway_entry.mark_error if gateway_entry else None),
            )
            if gateway_entry:
                streaming._nya_add_finalizer(gateway_entry.finish)
            if track:
                streaming._nya_add_finalizer(
                    lambda: self.metrics_collector.record_response(
                        api_name,
                        request.api_key,
                        response.status_code,
                        time.time() - actual_start_time,
                        request.trail_path,
                    )
                )
            return streaming

        try:
            result = await self.handle_normal_response(response, gateway_entry)
        except BaseException as error:
            if gateway_entry:
                gateway_entry.finish(error)
            if track:
                self.metrics_collector.record_response(
                    api_name,
                    request.api_key,
                    0,
                    time.time() - actual_start_time,
                    request.trail_path,
                )
            raise

        if gateway_entry:
            gateway_entry.finish()

        if track:
            self.metrics_collector.record_response(
                api_name,
                request.api_key,
                response.status_code,
                time.time() - actual_start_time,
                request.trail_path,
            )
        return result

    async def execute_request(
        self, request: "ProxyRequest", timeout: httpx.Timeout
    ) -> httpx.Response:
        """
        Execute the actual HTTP request.
        """
        stream = self.client.stream(
            method=request.method,
            url=request.url,
            headers=self._prepare_request_headers(request.headers),
            content=request.content,
            timeout=timeout,
        )

        response = await stream.__aenter__()
        response._stream_ctx = stream
        return response

    def _get_timeout(self, api_name: Optional[str] = None) -> httpx.Timeout:
        """
        Get timeout configuration for API.
        """
        timeout = (
            self.config.get_api_default_timeout(api_name)
            if api_name
            else self.config.get_default_timeout()
        )
        return httpx.Timeout(
            connect=5.0,
            read=timeout * 0.95,
            write=min(60.0, timeout * 0.2),
            pool=10.0,
        )

    async def process_response(
        self,
        response: httpx.Response,
    ) -> Union[Response, JSONResponse, StreamingResponse]:
        """
        Process and forward the response to the client.
        """
        try:
            if detect_streaming_content(response.headers):
                return await handle_streaming_response(response)

            return await self.handle_normal_response(response)
        except Exception:
            # Ensure cleanup happens even if processing fails
            if hasattr(response, "_stream_ctx") and response._stream_ctx:
                await response._stream_ctx.__aexit__(None, None, None)
            raise

    async def handle_normal_response(
        self,
        response: httpx.Response,
        gateway_entry: Optional[GatewayLogEntry] = None,
    ) -> Union[Response, JSONResponse]:
        """
        Create the response to send back to client.
        """
        try:
            content_type = response.headers.get("content-type", "")
            media_type = (
                content_type.split(";")[0].strip().lower()
                if content_type
                else "application/json"
            )

            logger.debug(
                f"Handling normal response: {response.status_code} {media_type}"
            )

            # Read raw bytes without any decoding/processing
            chunks = []
            async for chunk in response.aiter_raw():
                if gateway_entry:
                    gateway_entry.write_response_chunk(chunk)
                chunks.append(chunk)

            proxy_response = Response(
                content=b"".join(chunks),
                status_code=response.status_code,
                media_type=media_type,
            )
            apply_response_headers(proxy_response, response.headers, streaming=False)
            return proxy_response

        finally:
            # Properly close the stream context if it exists
            if hasattr(response, "_stream_ctx") and response._stream_ctx:
                await response._stream_ctx.__aexit__(None, None, None)
                response._stream_ctx = None

    async def close(self):
        """
        Close the HTTP client.
        """
        callbacks, self._close_callbacks = getattr(self, "_close_callbacks", []), []
        for callback in callbacks:
            await callback()
        if self.client:
            await self.client.aclose()

    def add_close_callback(self, callback) -> None:
        """Register async cleanup that must run before the client closes."""
        self._close_callbacks.append(callback)

    @staticmethod
    def _prepare_request_headers(headers: httpx.Headers):
        """Strip hop-by-hop headers and preserve content-coding capability.

        httpx advertises gzip by default. A transparent proxy must not do that
        on behalf of a downstream client that did not advertise decompression
        support, otherwise encoded bytes can reach a client that only expects
        identity content (notably TauriTavern's SSE parser).
        """
        connection_tokens = {
            token.strip().lower()
            for token in headers.get("connection", "").split(",")
            if token.strip()
        }
        excluded = _HOP_BY_HOP_HEADERS | connection_tokens
        prepared = [
            (name, value)
            for name, value in headers.multi_items()
            if name.lower() not in excluded
        ]
        if not any(name.lower() == "accept-encoding" for name, _ in prepared):
            prepared.append(("accept-encoding", "identity"))
        return prepared
