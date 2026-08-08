from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .secrets import LocalSecretStore, SecretStoreError


DEFAULT_CHAT_MODEL = "openrouter/free"
DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        code: str | int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after


class StreamControl:
    def __init__(self):
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._response: Any = None

    def attach(self, response: Any) -> None:
        with self._lock:
            self._response = response
            if self.cancelled.is_set():
                response.close()

    def detach(self) -> None:
        with self._lock:
            self._response = None

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            if self._response is not None:
                self._response.close()


class OpenRouterClient:
    """Small standard-library OpenRouter client with explicit streaming parsing."""

    def __init__(
        self,
        secret_store: LocalSecretStore,
        timeout: float = 120.0,
    ):
        self.secret_store = secret_store
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout = timeout

    def has_api_key(self) -> bool:
        try:
            return bool(self.secret_store.get())
        except SecretStoreError:
            return False

    def save_api_key(self, value: str) -> None:
        self.secret_store.set(value)

    def _api_key(self) -> str:
        try:
            return self.secret_store.get()
        except SecretStoreError as exc:
            raise OpenRouterError(str(exc)) from exc

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "Accept": accept,
            "HTTP-Referer": "http://127.0.0.1",
            "X-OpenRouter-Title": "EvoAgent Local",
            "User-Agent": "EvoAgent/0.2",
        }

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        control: StreamControl | None = None,
        data_collection: str = "deny",
    ) -> Iterator[dict[str, Any]]:
        if not model.strip() or "embed" in model.lower():
            raise OpenRouterError("嵌入模型不能用于生成聊天回复")
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "max_tokens": max(64, min(int(max_tokens), 32768)),
            "provider": {"data_collection": _data_policy(data_collection)},
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers("text/event-stream"),
            method="POST",
        )
        control = control or StreamControl()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                control.attach(response)
                generation_id = response.headers.get("X-Generation-Id")
                data_lines: list[str] = []
                saw_done = False
                for raw_line in response:
                    if control.cancelled.is_set():
                        raise OpenRouterError("生成已取消", code="cancelled")
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        event = self._parse_stream_event(data_lines, generation_id)
                        data_lines = []
                        if event is None:
                            continue
                        yield event
                        if event.get("type") == "done":
                            saw_done = True
                            break
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                event = self._parse_stream_event(data_lines, generation_id)
                if event:
                    yield event
                    saw_done = saw_done or event.get("type") == "done"
                if not saw_done and not control.cancelled.is_set():
                    raise OpenRouterError("OpenRouter 流意外中断，未收到 [DONE]", code="protocol_error")
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except URLError as exc:
            raise OpenRouterError(f"无法连接 OpenRouter：{exc.reason}") from exc
        except TimeoutError as exc:
            raise OpenRouterError("OpenRouter 请求超时") from exc
        finally:
            control.detach()

    @staticmethod
    def _parse_stream_event(data_lines: list[str], generation_id: str | None) -> dict[str, Any] | None:
        if not data_lines:
            return None
        payload_text = "\n".join(data_lines)
        if payload_text == "[DONE]":
            return {"type": "done"}
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter 返回了损坏的 SSE 数据", code="protocol_error") from exc
        if payload.get("error"):
            error = payload["error"]
            message = error.get("message", "OpenRouter 流式响应失败") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            raise OpenRouterError(message, code=code)
        choice = (payload.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return {
            "type": "chunk",
            "content": content if isinstance(content, str) else "",
            "model": payload.get("model"),
            "provider": payload.get("provider"),
            "usage": payload.get("usage") or {},
            "finish_reason": choice.get("finish_reason"),
            "generation_id": generation_id or payload.get("id"),
        }

    def embed(
        self, text: str, model: str, input_type: str, *, data_collection: str = "deny"
    ) -> dict[str, Any]:
        content = text.strip()
        if not content:
            raise ValueError("嵌入文本不能为空")
        if input_type not in {"query", "passage"}:
            raise ValueError("input_type 必须是 query 或 passage")
        payload = {
            "model": model,
            "input": content[:8000],
            "input_type": input_type,
            "encoding_format": "float",
            "provider": {"data_collection": _data_policy(data_collection)},
        }
        response: dict[str, Any] | None = None
        last_error: OpenRouterError | None = None
        for attempt in range(3):
            try:
                response = self._json_request("POST", "/embeddings", payload)
                break
            except OpenRouterError as exc:
                last_error = exc
                if exc.status not in {429, 502, 503} or attempt == 2:
                    raise
                time.sleep(min(exc.retry_after or (0.4 * (2**attempt)), 3.0))
        if response is None:
            raise last_error or OpenRouterError("嵌入请求失败")
        data = response.get("data") or []
        vector = data[0].get("embedding") if data and isinstance(data[0], dict) else None
        expected_dimensions = 2048 if model == DEFAULT_EMBEDDING_MODEL else None
        if not isinstance(vector, list) or not vector:
            raise OpenRouterError("嵌入模型返回了无效向量")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise OpenRouterError("嵌入向量包含无效数值") from exc
        if expected_dimensions is not None and len(values) != expected_dimensions:
            raise OpenRouterError(
                f"嵌入维度漂移：期望 {expected_dimensions}，实际 {len(values)}",
                code="dimension_mismatch",
            )
        if len(values) > 16384 or any(not math.isfinite(value) for value in values):
            raise OpenRouterError("嵌入向量包含非有限数值或维度过大")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise OpenRouterError("嵌入向量范数为零")
        return {
            "vector": [value / norm for value in values],
            "response_model": response.get("model") or model,
        }

    def list_models(self) -> list[dict[str, Any]]:
        response = self._json_request("GET", "/models")
        result: list[dict[str, Any]] = []
        for item in response.get("data", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            architecture = item.get("architecture") or {}
            output_modalities = architecture.get("output_modalities") or []
            pricing = item.get("pricing") or {}
            model_id = str(item["id"])
            is_embedding = "embeddings" in output_modalities or "embed" in model_id.lower()
            zero_price = (
                "prompt" in pricing
                and "completion" in pricing
                and _zero_price(pricing.get("prompt"))
                and _zero_price(pricing.get("completion"))
            )
            supports_text = not output_modalities or "text" in output_modalities
            result.append(
                {
                    "id": model_id,
                    "name": item.get("name") or model_id,
                    "context_length": item.get("context_length"),
                    "is_embedding": is_embedding,
                    "is_free": model_id.endswith(":free") or zero_price,
                    "supports_text": supports_text and not is_embedding,
                }
            )
        result.sort(key=lambda item: (not item["is_free"], item["is_embedding"], item["name"].lower()))
        return result[:500]

    def _json_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method=method,
        )
        try:
            with urlopen(request, timeout=min(self.timeout, 45.0)) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise OpenRouterError("OpenRouter 返回了无效 JSON")
                return parsed
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except URLError as exc:
            raise OpenRouterError(f"无法连接 OpenRouter：{exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter 返回了无法解析的响应") from exc

    @staticmethod
    def _http_error(exc: HTTPError) -> OpenRouterError:
        message = f"OpenRouter 请求失败（HTTP {exc.code}）"
        code: str | int | None = exc.code
        try:
            payload = json.loads(exc.read(65537).decode("utf-8"))
            error = payload.get("error", payload) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                code = error.get("code", code)
        except Exception:
            pass
        lowered = message.lower()
        if "data policy" in lowered or "free model training" in lowered:
            message = (
                "当前隐私设置不允许免费模型提供商处理数据。请在应用设置中开启"
                "“允许免费模型提供商处理数据”，或切换到兼容模型。"
            )
        elif exc.code == 401:
            message = "OpenRouter API Key 无效或已失效，请在设置中重新保存"
        retry_after: float | None = None
        try:
            retry_after = float(exc.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            pass
        return OpenRouterError(message, status=exc.code, code=code, retry_after=retry_after)


def _zero_price(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


def _data_policy(value: str) -> str:
    if value not in {"allow", "deny"}:
        raise ValueError("data_collection 必须是 allow 或 deny")
    return value
