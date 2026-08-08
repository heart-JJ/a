from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evoagent.chat import ChatService
from evoagent.db import Database
from evoagent.memory import MemoryStore
from evoagent.openrouter import DEFAULT_EMBEDDING_MODEL, OpenRouterError
from evoagent.secrets import MemorySecretStore
from evoagent.skills import SkillRegistry


CHAT_MODEL = "mock/provider-chat"


class FakeGateway:
    """Deterministic in-memory gateway. It never opens a network connection."""

    def __init__(self, secret_store: MemorySecretStore):
        self.secret_store = secret_store
        self.chat_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self.replies: list[str] = []
        self.failure: OpenRouterError | None = None

    def has_api_key(self) -> bool:
        return self.secret_store.configured()

    def save_api_key(self, value: str) -> None:
        self.secret_store.set(value)

    def list_models(self) -> list[dict]:
        return [
            {
                "id": CHAT_MODEL,
                "name": "Mock Chat",
                "supports_text": True,
                "is_embedding": False,
                "is_free": True,
                "context_length": 8192,
            }
        ]

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        control,
        **kwargs,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "control": control,
                "options": kwargs,
            }
        )
        if self.failure is not None:
            failure = self.failure
            self.failure = None
            yield {
                "type": "chunk",
                "content": "partial",
                "model": model,
                "provider": "fake",
                "generation_id": "fake-failed-generation",
            }
            raise failure

        reply = self.replies.pop(0) if self.replies else "这是 FakeGateway 的流式回答。"
        split = max(1, len(reply) // 2)
        yield {
            "type": "chunk",
            "content": reply[:split],
            "model": f"resolved/{model}",
            "provider": "fake",
            "generation_id": f"fake-generation-{len(self.chat_calls)}",
        }
        yield {
            "type": "chunk",
            "content": reply[split:],
            "model": f"resolved/{model}",
            "provider": "fake",
            "generation_id": f"fake-generation-{len(self.chat_calls)}",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        yield {"type": "done"}

    def embed(self, text: str, model: str, input_type: str, **kwargs) -> dict:
        self.embed_calls.append(
            {"text": text, "model": model, "input_type": input_type, "options": kwargs}
        )
        return {
            "vector": [1.0, *([0.0] * 2047)],
            "response_model": model,
        }


class ChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "chat.db")
        self.database.initialize()
        self.skills = SkillRegistry(self.database)
        self.skills.seed()
        self.memory = MemoryStore(self.database)
        self.secrets = MemorySecretStore("sk-or-v1-test-only-not-a-real-key")
        self.gateway = FakeGateway(self.secrets)
        self.chat = ChatService(
            self.database,
            self.skills,
            self.memory,
            gateway=self.gateway,
            secret_store=self.secrets,
        )
        self.chat.update_settings({"chat_model": CHAT_MODEL, "memory_enabled": False})

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def event(events: list[dict], name: str) -> dict:
        return next(item for item in events if item["event"] == name)

    def test_auto_creates_conversation_streams_and_records_skills_tags_experience(self) -> None:
        self.gateway.replies = ["摘要和关键词已经整理完成。"]
        events = list(
            self.chat.stream_reply(
                "请总结下面内容并提取关键词",
                "request-auto-0001",
            )
        )

        accepted = self.event(events, "accepted")
        done = self.event(events, "done")
        deltas = [item["text"] for item in events if item["event"] == "delta"]
        self.assertEqual("".join(deltas), "摘要和关键词已经整理完成。")
        self.assertEqual(done["conversation_id"], accepted["conversation_id"])
        self.assertEqual(done["requested_model"], CHAT_MODEL)
        self.assertEqual(len(self.chat.list_conversations()), 1)

        messages = self.chat.list_messages(accepted["conversation_id"])
        self.assertEqual([(item["role"], item["status"]) for item in messages], [
            ("user", "complete"),
            ("assistant", "complete"),
        ])
        assistant = messages[1]
        selected_names = {item["name"] for item in assistant["selected_skills"]}
        self.assertIn("文本摘要", selected_names)
        self.assertIn("关键词提取", selected_names)
        self.assertTrue(assistant["tags"])
        self.assertTrue(assistant["experience_id"])
        self.assertEqual(assistant["experience_id"], done["experience_id"])

        experience = self.memory.get_experience(assistant["experience_id"])
        self.assertEqual(experience["task"], "请总结下面内容并提取关键词")
        self.assertEqual(
            {item["skill_id"] for item in experience["skills"]},
            {item["skill_id"] for item in assistant["selected_skills"]},
        )
        self.assertEqual(self.gateway.chat_calls[0]["model"], CHAT_MODEL)
        self.assertEqual(self.gateway.chat_calls[0]["messages"][-1]["content"], messages[0]["content"])

    def test_chat_and_embedding_models_are_strictly_separated(self) -> None:
        self.chat.update_settings(
            {
                "chat_model": CHAT_MODEL,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "memory_enabled": True,
            }
        )
        with patch.object(
            self.chat,
            "_start_embedding",
            side_effect=self.chat._index_embedding,
        ):
            events = list(
                self.chat.stream_reply(
                    "请总结这段关于模型分工的文字",
                    "request-models-0001",
                )
            )

        self.assertEqual(self.event(events, "done")["requested_model"], CHAT_MODEL)
        self.assertEqual([call["model"] for call in self.gateway.chat_calls], [CHAT_MODEL])
        self.assertEqual(
            {call["options"].get("data_collection") for call in self.gateway.chat_calls},
            {"deny"},
        )
        self.assertGreaterEqual(len(self.gateway.embed_calls), 2)
        self.assertEqual(
            {call["model"] for call in self.gateway.embed_calls},
            {DEFAULT_EMBEDDING_MODEL},
        )
        self.assertEqual(
            {call["input_type"] for call in self.gateway.embed_calls},
            {"query", "passage"},
        )
        self.assertEqual(
            {call["options"].get("data_collection") for call in self.gateway.embed_calls},
            {"deny"},
        )
        self.assertNotEqual(CHAT_MODEL, DEFAULT_EMBEDDING_MODEL)

    def test_client_request_id_replay_does_not_call_model_twice(self) -> None:
        first = list(
            self.chat.stream_reply(
                "请概括这段话",
                "request-replay-0001",
            )
        )
        conversation_id = self.event(first, "accepted")["conversation_id"]
        first_answer = "".join(item["text"] for item in first if item["event"] == "delta")
        self.assertEqual(len(self.gateway.chat_calls), 1)

        replay = list(
            self.chat.stream_reply(
                "即使正文不同，也应按同一个请求 ID 重放",
                "request-replay-0001",
                conversation_id=conversation_id,
            )
        )

        self.assertEqual(len(self.gateway.chat_calls), 1)
        self.assertTrue(self.event(replay, "accepted")["replayed"])
        self.assertTrue(self.event(replay, "done")["replayed"])
        replay_answer = "".join(item["text"] for item in replay if item["event"] == "delta")
        self.assertEqual(replay_answer, first_answer)
        self.assertEqual(len(self.chat.list_messages(conversation_id)), 2)

    def test_gateway_failure_marks_run_and_assistant_failed(self) -> None:
        self.gateway.failure = OpenRouterError(
            "fake provider unavailable",
            status=503,
            code="provider_unavailable",
        )
        events = list(
            self.chat.stream_reply(
                "请总结失败场景",
                "request-failure-0001",
            )
        )

        accepted = self.event(events, "accepted")
        error = self.event(events, "error")
        self.assertEqual(error["code"], "provider_unavailable")
        messages = self.chat.list_messages(accepted["conversation_id"])
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertEqual(messages[-1]["content"], "partial")
        self.assertIsNone(messages[-1]["experience_id"])
        with self.database.read() as connection:
            run = connection.execute(
                "SELECT status FROM chat_runs WHERE id=?", (accepted["run_id"],)
            ).fetchone()
        self.assertEqual(run["status"], "failed")

    def test_cancel_run_marks_run_and_assistant_cancelled(self) -> None:
        stream = self.chat.stream_reply(
            "请生成一段可以中断的长回复",
            "request-cancel-0001",
        )
        accepted = next(stream)
        self.assertEqual(accepted["event"], "accepted")
        cancellation = self.chat.cancel_run(accepted["run_id"])
        self.assertEqual(cancellation["status"], "cancelling")
        remaining = list(stream)
        self.assertEqual(self.event(remaining, "cancelled")["run_id"], accepted["run_id"])

        messages = self.chat.list_messages(accepted["conversation_id"])
        self.assertEqual(messages[-1]["status"], "cancelled")
        self.assertIsNone(messages[-1]["experience_id"])
        with self.database.read() as connection:
            run = connection.execute(
                "SELECT status FROM chat_runs WHERE id=?", (accepted["run_id"],)
            ).fetchone()
        self.assertEqual(run["status"], "cancelled")

    def test_settings_never_return_or_persist_plaintext_api_key(self) -> None:
        raw_key = "sk-or-v1-this-key-must-never-be-returned-1234567890"
        updated = self.chat.update_settings(
            {
                "api_key": raw_key,
                "chat_model": CHAT_MODEL,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            }
        )
        settings = self.chat.get_settings()
        models = self.chat.list_models(refresh=True)

        for payload in (updated, settings, models):
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn(raw_key, serialized)
            self.assertNotIn("api_key\"", serialized)
        self.assertTrue(settings["api_key_configured"])
        with self.database.read() as connection:
            stored = connection.execute(
                "SELECT value_json FROM app_settings WHERE key='chat'"
            ).fetchone()["value_json"]
        self.assertNotIn(raw_key, stored)
        self.assertNotIn(raw_key, self.chat._safe_error(RuntimeError(f"provider echoed {raw_key}")))

    def test_ready_semantic_embedding_is_retrieved_in_a_new_conversation(self) -> None:
        self.chat.update_settings(
            {
                "chat_model": CHAT_MODEL,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "memory_enabled": True,
            }
        )
        self.gateway.replies = ["第一段旧对话回答。", "第二段新对话回答。"]
        with patch.object(
            self.chat,
            "_start_embedding",
            side_effect=self.chat._index_embedding,
        ):
            first = list(
                self.chat.stream_reply(
                    "记住量子猫项目的发布安排",
                    "request-memory-0001",
                )
            )
            first_accepted = self.event(first, "accepted")
            first_messages = self.chat.list_messages(first_accepted["conversation_id"])
            first_assistant_id = first_messages[-1]["id"]
            with self.database.read() as connection:
                embedding = connection.execute(
                    "SELECT status, normalized, dimensions FROM memory_embeddings WHERE message_id=?",
                    (first_assistant_id,),
                ).fetchone()
            self.assertEqual(
                (embedding["status"], embedding["normalized"], embedding["dimensions"]),
                ("ready", 1, 2048),
            )

            second = list(
                self.chat.stream_reply(
                    "量子猫项目之前怎么安排的？",
                    "request-memory-0002",
                )
            )

        second_meta = self.event(second, "meta")
        self.assertIn(first_assistant_id, {item["id"] for item in second_meta["memories"]})
        second_conversation_id = self.event(second, "accepted")["conversation_id"]
        self.assertNotEqual(second_conversation_id, first_accepted["conversation_id"])
        second_assistant = self.chat.list_messages(second_conversation_id)[-1]
        memory_ref = next(
            item for item in second_assistant["memory_refs"] if item["id"] == first_assistant_id
        )
        self.assertEqual(memory_ref["source"], "semantic")


if __name__ == "__main__":
    unittest.main()
