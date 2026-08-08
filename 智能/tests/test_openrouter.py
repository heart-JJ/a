from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evoagent.openrouter import (
    DEFAULT_EMBEDDING_MODEL,
    OpenRouterClient,
    OpenRouterError,
)
from evoagent.secrets import LocalSecretStore, MemorySecretStore


class FakeStreamingResponse:
    def __init__(self, lines: list[bytes], headers: dict[str, str] | None = None):
        self._lines = lines
        self.headers = headers or {}
        self.closed = False

    def __enter__(self) -> "FakeStreamingResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


class OpenRouterClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenRouterClient(MemorySecretStore("sk-or-test-only-not-a-real-key"))

    def test_stream_accepts_comments_multiline_data_and_done(self) -> None:
        response = FakeStreamingResponse(
            [
                b": keep-alive\n",
                b"\n",
                b'data: {"id":"payload-id",\n',
                b'data: "model":"mock/chat","provider":"mock",\n',
                'data: "choices":[{"delta":{"content":"你"}}]}\n'.encode(),
                b"\n",
                'data: {"choices":[{"delta":{"content":"好"},"finish_reason":"stop"}]}\n'.encode(),
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ],
            {"X-Generation-Id": "header-generation-id"},
        )

        with patch("evoagent.openrouter.urlopen", return_value=response) as mocked_urlopen:
            events = list(
                self.client.stream_chat(
                    [{"role": "user", "content": "你好"}],
                    "mock/chat",
                )
            )

        self.assertEqual([event["type"] for event in events], ["chunk", "chunk", "done"])
        self.assertEqual("".join(event.get("content", "") for event in events), "你好")
        self.assertEqual(events[0]["generation_id"], "header-generation-id")
        self.assertEqual(events[0]["model"], "mock/chat")
        self.assertEqual(events[0]["provider"], "mock")
        self.assertEqual(events[1]["finish_reason"], "stop")
        self.assertTrue(response.closed)
        mocked_urlopen.assert_called_once()

    def test_stream_raises_on_midstream_error_event(self) -> None:
        response = FakeStreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                b"\n",
                b'data: {"error":{"message":"provider failed","code":"provider_error"}}\n',
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ]
        )

        with patch("evoagent.openrouter.urlopen", return_value=response):
            stream = self.client.stream_chat(
                [{"role": "user", "content": "test"}],
                "mock/chat",
            )
            self.assertEqual(next(stream)["content"], "partial")
            with self.assertRaises(OpenRouterError) as raised:
                next(stream)

        self.assertEqual(str(raised.exception), "provider failed")
        self.assertEqual(raised.exception.code, "provider_error")
        self.assertTrue(response.closed)

    def test_stream_raises_protocol_error_on_unexpected_eof(self) -> None:
        response = FakeStreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                b"\n",
            ]
        )

        with patch("evoagent.openrouter.urlopen", return_value=response):
            with self.assertRaises(OpenRouterError) as raised:
                list(
                    self.client.stream_chat(
                        [{"role": "user", "content": "test"}],
                        "mock/chat",
                    )
                )

        self.assertEqual(raised.exception.code, "protocol_error")
        self.assertIn("[DONE]", str(raised.exception))
        self.assertTrue(response.closed)

    def test_default_embedding_is_2048_dimensions_and_normalized(self) -> None:
        source = [3.0, 4.0, *([0.0] * 2046)]
        with patch.object(
            self.client,
            "_json_request",
            return_value={"data": [{"embedding": source}], "model": "resolved/embed"},
        ) as request:
            result = self.client.embed("需要检索的文本", DEFAULT_EMBEDDING_MODEL, "passage")

        vector = result["vector"]
        self.assertEqual(len(vector), 2048)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)
        self.assertEqual(result["response_model"], "resolved/embed")
        request.assert_called_once()
        method, path, payload = request.call_args.args
        self.assertEqual((method, path), ("POST", "/embeddings"))
        self.assertEqual(payload["model"], DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(payload["input_type"], "passage")

    def test_default_embedding_rejects_dimension_drift(self) -> None:
        with patch.object(
            self.client,
            "_json_request",
            return_value={"data": [{"embedding": [1.0] * 2047}]},
        ):
            with self.assertRaises(OpenRouterError) as raised:
                self.client.embed("query", DEFAULT_EMBEDDING_MODEL, "query")

        self.assertEqual(raised.exception.code, "dimension_mismatch")
        self.assertIn("期望 2048", str(raised.exception))
        self.assertIn("实际 2047", str(raised.exception))

    def test_embedding_rejects_non_finite_values(self) -> None:
        vector = [1.0] * 2048
        vector[100] = float("nan")
        with patch.object(
            self.client,
            "_json_request",
            return_value={"data": [{"embedding": vector}]},
        ):
            with self.assertRaises(OpenRouterError) as raised:
                self.client.embed("query", DEFAULT_EMBEDDING_MODEL, "query")

        self.assertIn("非有限数值", str(raised.exception))

    def test_missing_key_is_safe_and_never_attempts_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "openrouter-key.bin"
            store = LocalSecretStore(secret_path)
            client = OpenRouterClient(store)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
                with patch("evoagent.openrouter.urlopen") as mocked_urlopen:
                    with self.assertRaises(OpenRouterError) as raised:
                        client.list_models()

        message = str(raised.exception)
        self.assertIn("尚未配置", message)
        self.assertNotIn("Authorization", message)
        self.assertNotIn("Bearer", message)
        self.assertNotIn("sk-or-", message)
        mocked_urlopen.assert_not_called()
        self.assertFalse(secret_path.exists())


if __name__ == "__main__":
    unittest.main()
