"""Full-fidelity gateway request and response logging.

Each upstream attempt receives a stable numeric id and a small bundle of
files, modelled after TauriTavern's LLM API logs.  Request and response bodies
are kept byte-for-byte so transport problems can be distinguished from parser
problems after the fact.  An index and metadata file provide a readable view
without changing the captured bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from ..common.models import ProxyRequest


INDEX_NAME = "gateway-index.json"


def resolve_gateway_log_dir(
    config_path: str | None, configured_directory: str | None
) -> Path:
    """Resolve a configured log directory next to the active configuration."""
    if configured_directory:
        configured = Path(configured_directory)
        if configured.is_absolute():
            return configured
        if config_path:
            return Path(config_path).resolve().parent / configured
        return Path.cwd() / configured
    if config_path:
        return Path(config_path).resolve().parent / "gateway-logs"
    return Path.cwd() / "gateway-logs"


def _header_pairs(headers: httpx.Headers | None) -> list[list[str]]:
    if headers is None:
        return []
    return [[name, value] for name, value in headers.multi_items()]


def _body_readable(body: bytes, headers: httpx.Headers | None) -> dict[str, Any]:
    """Return a complete, human-readable representation of ``body``."""
    content_type = headers.get("content-type", "") if headers is not None else ""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "kind": "base64",
            "value": base64.b64encode(body).decode("ascii"),
        }

    if "json" in content_type.lower():
        try:
            return {"kind": "json", "value": json.loads(text)}
        except ValueError:
            pass
    return {"kind": "text", "value": text}


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class GatewayLogEntry:
    """One upstream request attempt and its exact response stream."""

    def __init__(
        self,
        store: GatewayLogStore,
        log_id: int,
        request: ProxyRequest,
    ) -> None:
        self.store = store
        self.id = log_id
        self.trace_id = uuid.uuid4().hex
        self.started_at = time.time()
        self._lock = threading.RLock()
        self._finished = False
        self._error: str | None = None
        self._response_status: int | None = None
        self._response_headers: list[list[str]] = []
        self._response_content_type = ""
        self._response_content_encoding = ""
        self._response_bytes = 0
        self._response_chunks = 0
        self._response_hash = hashlib.sha256()

        self._api_name = request.api_name
        self._attempt = request.attempts
        self._method = request.method.upper()
        self._client_url = str(request._url)
        self._upstream_url = request.url
        self._selected_api_key = request.api_key
        self._model: str | None = None
        self._stream_requested: bool | None = None
        self._response_http_version = ""
        self._response_reason_phrase = ""

        self.base_name = f"gateway-{self.id}"
        self.request_path = store.directory / f"{self.base_name}.request.json"
        self.client_raw_path = store.directory / f"{self.base_name}.client-request.raw"
        self.request_raw_path = store.directory / f"{self.base_name}.request.raw"
        self.response_raw_path = store.directory / f"{self.base_name}.response.raw"
        self.response_text_path = store.directory / f"{self.base_name}.response.txt"
        self.meta_path = store.directory / f"{self.base_name}.meta.json"

        client_body = bytes(request.original_content or b"")
        upstream_body = bytes(request.content or b"")
        client_headers = request.original_headers
        upstream_headers = request.headers
        self._request_bytes = len(upstream_body)
        self._request_sha256 = hashlib.sha256(upstream_body).hexdigest()
        try:
            upstream_json = json.loads(upstream_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            upstream_json = None
        if isinstance(upstream_json, dict):
            model = upstream_json.get("model")
            self._model = model if isinstance(model, str) else None
            stream = upstream_json.get("stream")
            self._stream_requested = stream if isinstance(stream, bool) else None

        self.client_raw_path.write_bytes(client_body)
        self.request_raw_path.write_bytes(upstream_body)
        self.response_raw_path.write_bytes(b"")
        self._response_handle = self.response_raw_path.open("ab", buffering=0)

        request_document = {
            "id": self.id,
            "traceId": self.trace_id,
            "timestampMs": round(self.started_at * 1000),
            "attempt": request.attempts,
            "apiName": request.api_name,
            "selectedApiKey": request.api_key,
            "client": {
                "method": request.method.upper(),
                "url": str(request._url),
                "ip": request.ip,
                "user": request.user,
                "headers": _header_pairs(client_headers),
                "body": _body_readable(client_body, client_headers),
            },
            "upstream": {
                "method": request.method.upper(),
                "url": request.url,
                "headers": _header_pairs(upstream_headers),
                "body": _body_readable(upstream_body, upstream_headers),
            },
        }
        _atomic_json_write(self.request_path, request_document)
        self._write_meta(completed=False)

    def record_response_head(
        self,
        status_code: int,
        headers: httpx.Headers,
        *,
        http_version: str = "",
        reason_phrase: str = "",
    ) -> None:
        with self._lock:
            self._response_status = status_code
            self._response_headers = _header_pairs(headers)
            self._response_content_type = headers.get("content-type", "")
            self._response_content_encoding = headers.get("content-encoding", "")
            self._response_http_version = http_version
            self._response_reason_phrase = reason_phrase
            self._write_meta(completed=False)

    def write_response_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            if self._finished:
                return
            self._response_handle.write(chunk)
            self._response_bytes += len(chunk)
            self._response_chunks += 1
            self._response_hash.update(chunk)

    def mark_error(self, error: BaseException) -> None:
        with self._lock:
            self._error = f"{type(error).__name__}: {error}"

    def finish(self, error: BaseException | None = None) -> None:
        with self._lock:
            if self._finished:
                return
            if error is not None:
                self.mark_error(error)
            self._finished = True
            self._response_handle.flush()
            self._response_handle.close()
            self._write_readable_response()
            self._write_meta(completed=True)
            self.store._entry_finished(self)

    def _write_readable_response(self) -> None:
        content_type = self._response_content_type.lower()
        content_encoding = self._response_content_encoding.strip().lower()
        is_text = (
            content_type.startswith("text/")
            or "json" in content_type
            or "xml" in content_type
        )
        if content_encoding or not is_text:
            try:
                self.response_text_path.unlink()
            except FileNotFoundError:
                pass
            return
        try:
            text = self.response_raw_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return
        self.response_text_path.write_text(text, encoding="utf-8", newline="\n")

    def metadata(self, completed: bool) -> dict[str, Any]:
        now = time.time()
        return {
            "id": self.id,
            "traceId": self.trace_id,
            "timestampMs": round(self.started_at * 1000),
            "apiName": self._api_name,
            "attempt": self._attempt,
            "method": self._method,
            "clientUrl": self._client_url,
            "upstreamUrl": self._upstream_url,
            "selectedApiKey": self._selected_api_key,
            "model": self._model,
            "streamRequested": self._stream_requested,
            "completedAtMs": round(now * 1000) if completed else None,
            "durationMs": round((now - self.started_at) * 1000),
            "completed": completed,
            "ok": completed and self._error is None,
            "error": self._error,
            "statusCode": self._response_status,
            "httpVersion": self._response_http_version,
            "reasonPhrase": self._response_reason_phrase,
            "responseHeaders": self._response_headers,
            "contentType": self._response_content_type,
            "contentEncoding": self._response_content_encoding,
            "requestBytes": self._request_bytes,
            "requestSha256": self._request_sha256,
            "responseBytes": self._response_bytes,
            "responseChunks": self._response_chunks,
            "responseSha256": (
                self._response_hash.hexdigest() if self._response_chunks else None
            ),
            "files": {
                "request": self.request_path.name,
                "clientRequestRaw": self.client_raw_path.name,
                "requestRaw": self.request_raw_path.name,
                "responseRaw": self.response_raw_path.name,
                "responseText": (
                    self.response_text_path.name
                    if self.response_text_path.exists()
                    else None
                ),
                "meta": self.meta_path.name,
            },
        }

    def _write_meta(self, completed: bool) -> None:
        _atomic_json_write(self.meta_path, self.metadata(completed))


class GatewayLogStore:
    """Allocate gateway ids, persist bundles, and retain the newest entries."""

    def __init__(self, directory: str | Path, retention: int = 100) -> None:
        self.directory = Path(directory)
        self.retention = max(1, int(retention))
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._next_id = self._discover_next_id()
        with self._lock:
            self._prune_locked()
            self._write_index_locked()

    @classmethod
    def from_config(cls, config: Any) -> GatewayLogStore | None:
        getter = getattr(config, "get_gateway_logging_config", None)
        if getter is None:
            return None
        settings = getter()
        if not settings.get("enabled", True):
            return None
        directory = resolve_gateway_log_dir(
            getattr(config, "config_path", None), settings.get("directory")
        )
        return cls(directory, settings.get("retention", 100))

    def begin(self, request: ProxyRequest) -> GatewayLogEntry:
        with self._lock:
            log_id = self._next_id
            self._next_id += 1
            try:
                entry = GatewayLogEntry(self, log_id, request)
            except BaseException:
                for path in self.directory.glob(f"gateway-{log_id}.*"):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                raise
            self._prune_locked()
            return entry

    def _entry_finished(self, entry: GatewayLogEntry) -> None:
        del entry
        with self._lock:
            self._prune_locked()
            self._write_index_locked()

    def _discover_next_id(self) -> int:
        ids = self._existing_ids()
        return (max(ids) + 1) if ids else 1

    def _existing_ids(self) -> list[int]:
        ids: list[int] = []
        for path in self.directory.glob("gateway-*.meta.json"):
            stem = path.name.removeprefix("gateway-").removesuffix(".meta.json")
            if stem.isdigit():
                ids.append(int(stem))
        return sorted(ids)

    def _prune_locked(self) -> None:
        ids = self._existing_ids()
        for log_id in ids[: max(0, len(ids) - self.retention)]:
            for path in self.directory.glob(f"gateway-{log_id}.*"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _write_index_locked(self) -> None:
        entries: list[dict[str, Any]] = []
        for log_id in self._existing_ids():
            path = self.directory / f"gateway-{log_id}.meta.json"
            try:
                with path.open("r", encoding="utf-8") as handle:
                    entry = json.load(handle)
            except (OSError, ValueError):
                continue
            entries.append(entry)
        _atomic_json_write(self.directory / INDEX_NAME, entries)


def gateway_bundle_paths(directory: str | Path, log_id: int) -> Iterable[Path]:
    """Expose bundle paths for tests and future dashboard/download routes."""
    return Path(directory).glob(f"gateway-{log_id}.*")
