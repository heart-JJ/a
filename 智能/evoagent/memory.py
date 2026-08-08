from __future__ import annotations

import sqlite3
from typing import Any

from .db import Database
from .utils import json_dumps, json_loads, new_id, pattern_key, similarity, utc_now


class MemoryStore:
    def __init__(self, database: Database):
        self.db = database

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
        actor: str,
    ) -> None:
        connection.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (new_id("audit"), actor, event_type, entity_type, entity_id, json_dumps(payload), utc_now()),
        )

    def record_experience(
        self,
        task: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        selected_skills: list[dict[str, Any]],
        technical_success: bool,
        latency_ms: float,
        tags: list[str] | None = None,
        salience: float = 0.5,
        reflection: str = "",
        invocations: list[dict[str, Any]] | None = None,
    ) -> str:
        with self.db.transaction() as connection:
            return self.record_experience_in_connection(
                connection=connection,
                task=task,
                input_payload=input_payload,
                output_payload=output_payload,
                selected_skills=selected_skills,
                technical_success=technical_success,
                latency_ms=latency_ms,
                tags=tags,
                salience=salience,
                reflection=reflection,
                invocations=invocations,
            )

    def record_experience_in_connection(
        self,
        connection: sqlite3.Connection,
        task: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        selected_skills: list[dict[str, Any]],
        technical_success: bool,
        latency_ms: float,
        tags: list[str] | None = None,
        salience: float = 0.5,
        reflection: str = "",
        invocations: list[dict[str, Any]] | None = None,
    ) -> str:
        experience_id = new_id("exp")
        tags = sorted({str(tag).strip() for tag in (tags or []) if str(tag).strip()})
        now = utc_now()
        connection.execute(
            """
            INSERT INTO experiences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                experience_id,
                task,
                json_dumps(input_payload),
                json_dumps(output_payload),
                pattern_key(task, tags),
                json_dumps(tags),
                max(0.0, min(1.0, float(salience))),
                int(bool(technical_success)),
                max(0.0, float(latency_ms)),
                reflection,
                now,
            ),
        )
        for position, selection in enumerate(selected_skills):
            decision = {
                "score": selection.get("score", 0),
                "components": selection.get("components", {}),
                "weights": selection.get("weights", {}),
                "reason": selection.get("reason", ""),
            }
            connection.execute(
                "INSERT INTO experience_skills VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experience_id,
                    selection["skill_id"],
                    int(selection["version"]),
                    position,
                    float(selection.get("score", 0)),
                    json_dumps(decision),
                ),
            )
        usage_records = invocations if invocations is not None else selected_skills
        seen_invocations: set[tuple[str, int]] = set()
        default_latency = max(0.0, float(latency_ms)) / max(1, len(usage_records))
        for invocation in usage_records:
            key = (invocation["skill_id"], int(invocation["version"]))
            if key in seen_invocations:
                continue
            seen_invocations.add(key)
            connection.execute(
                "INSERT INTO skill_usage_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id("usage"),
                    experience_id,
                    key[0],
                    key[1],
                    max(0.0, float(invocation.get("duration_ms", default_latency))),
                    int(bool(technical_success)),
                    now,
                ),
            )
        self._audit(
            connection,
            "experience.recorded",
            "experience",
            experience_id,
            {"technical_success": technical_success, "skills": len(selected_skills)},
            "system",
        )
        return experience_id

    def add_feedback(
        self,
        experience_id: str,
        success: bool | None,
        score: float | None,
        notes: str = "",
        source: str = "user",
        confidence: float = 1.0,
        actor: str = "user",
    ) -> str:
        if source not in {"user", "test", "tool", "model"}:
            raise ValueError("不支持的评价来源")
        if score is not None and not 0 <= float(score) <= 1:
            raise ValueError("score 必须在 0 到 1 之间")
        evaluation_id = new_id("eval")
        with self.db.transaction() as connection:
            exists = connection.execute("SELECT 1 FROM experiences WHERE id = ?", (experience_id,)).fetchone()
            if not exists:
                raise KeyError("经验不存在")
            connection.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation_id,
                    experience_id,
                    source,
                    None if success is None else int(bool(success)),
                    score,
                    max(0.0, min(1.0, float(confidence))),
                    notes,
                    utc_now(),
                ),
            )
            self._audit(
                connection,
                "evaluation.recorded",
                "experience",
                experience_id,
                {"evaluation_id": evaluation_id, "source": source, "success": success, "score": score},
                actor,
            )
        return evaluation_id

    def set_evolution_eligibility(self, experience_id: str, eligible: bool, actor: str = "user") -> None:
        with self.db.transaction() as connection:
            result = connection.execute(
                "UPDATE experiences SET eligible_for_evolution = ? WHERE id = ?",
                (int(bool(eligible)), experience_id),
            )
            if result.rowcount == 0:
                raise KeyError("经验不存在")
            self._audit(
                connection,
                "experience.evolution_eligibility_changed",
                "experience",
                experience_id,
                {"eligible": eligible},
                actor,
            )

    def list_experiences(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.read() as connection:
            rows = connection.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC, id DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
            result = []
            for row in rows:
                data = self._hydrate_experience(connection, row)
                result.append(data)
            return result

    def get_experience(self, experience_id: str) -> dict[str, Any]:
        with self.db.read() as connection:
            row = connection.execute("SELECT * FROM experiences WHERE id = ?", (experience_id,)).fetchone()
            if not row:
                raise KeyError("经验不存在")
            return self._hydrate_experience(connection, row)

    def _hydrate_experience(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        skills = connection.execute(
            """
            SELECT es.*, s.name FROM experience_skills es
            JOIN skills s ON s.id = es.skill_id
            WHERE es.experience_id = ? ORDER BY es.position
            """,
            (row["id"],),
        ).fetchall()
        evaluations = connection.execute(
            "SELECT * FROM evaluations WHERE experience_id = ? ORDER BY created_at DESC, rowid DESC",
            (row["id"],),
        ).fetchall()
        data = dict(row)
        data["input"] = json_loads(row["input_json"], {})
        data["output"] = json_loads(row["output_json"], {})
        data["tags"] = json_loads(row["tags_json"], [])
        data["skills"] = [
            {**dict(item), "decision": json_loads(item["decision_json"], {})} for item in skills
        ]
        data["evaluations"] = [dict(item) for item in evaluations]
        data["latest_evaluation"] = data["evaluations"][0] if data["evaluations"] else None
        data["latest_trusted_evaluation"] = next(
            (
                item for item in data["evaluations"]
                if item["source"] in {"user", "test", "tool"}
                and item["confidence"] >= 0.6
                and item["success"] is not None
            ),
            None,
        )
        return data

    def similar(self, task: str, limit: int = 5) -> list[dict[str, Any]]:
        candidates = self.list_experiences(limit=300)
        ranked = []
        for item in candidates:
            item = dict(item)
            item["similarity"] = round(similarity(task, item["task"]), 4)
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["similarity"], item["created_at"]))
        return ranked[: max(1, min(limit, 20))]
