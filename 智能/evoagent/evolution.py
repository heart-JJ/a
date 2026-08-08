from __future__ import annotations

import sqlite3
import threading
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from .db import Database
from .skills import SkillRegistry, validate_spec
from .utils import content_hash, json_dumps, json_loads, new_id, slugify, utc_now


def _feedback_issue(notes: str) -> str | None:
    normalized = notes.strip().lower()
    if any(marker in normalized for marker in ("误路由", "选错技能", "不应命中", "不该命中", "不该使用")):
        return "routing"
    if any(marker in normalized for marker in ("耗时", "处理时间", "响应时间", "延迟", "运行时间")):
        return None
    output_subject = r"(?:摘要|输出|内容|结果|文字|篇幅)"
    length_complaint = r"(?:太长|过长|冗长|需要更短|不够简洁)"
    if re.search(rf"{output_subject}.{{0,8}}{length_complaint}|{length_complaint}.{{0,8}}{output_subject}", normalized):
        return "summary_length"
    return None


class EvolutionEngine:
    """Creates governed proposals from trusted feedback.

    It never writes executable code and never silently changes an active skill.
    Candidate approval is an explicit release action and remains auditable.
    """

    def __init__(self, database: Database, registry: SkillRegistry, min_support: int = 3):
        self.db = database
        self.registry = registry
        self.min_support = max(2, int(min_support))
        self._lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("已有进化任务正在运行")
        run_id = new_id("evo")
        started = utc_now()
        try:
            with self.db.transaction() as connection:
                connection.execute(
                    "INSERT INTO evolution_runs VALUES (?, 'running', 0, 0, ?, ?, NULL)",
                    (run_id, json_dumps({"phase": "scanning"}), started),
                )
                trusted = self._trusted_experiences(connection)
                successful = [item for item in trusted if bool(item["evaluation_success"])]
                failed = [item for item in trusted if not bool(item["evaluation_success"])]
                created: list[str] = []
                created.extend(self._discover_patterns(connection, successful))
                created.extend(self._discover_revisions(connection, failed))
                summary = {
                    "trusted_successes": len(successful),
                    "trusted_failures": len(failed),
                    "created_candidates": created,
                    "policy": "单一快照内扫描；仅可信反馈参与；候选不会自动发布",
                }
                connection.execute(
                    """
                    UPDATE evolution_runs
                    SET status = 'completed', input_count = ?, candidate_count = ?, summary_json = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (len(successful) + len(failed), len(created), json_dumps(summary), utc_now(), run_id),
                )
                self._audit(
                    connection,
                    "evolution.completed",
                    "evolution_run",
                    run_id,
                    summary,
                    "system",
                )
            return {"id": run_id, "status": "completed", **summary}
        except Exception as exc:
            with self.db.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO evolution_runs
                    VALUES (?, 'failed', 0, 0, ?, ?, ?)
                    """,
                    (run_id, json_dumps({"error": str(exc)}), started, utc_now()),
                )
            raise
        finally:
            self._lock.release()

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

    def _trusted_experiences(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = connection.execute(
                """
                WITH ranked AS (
                    SELECT e.*, ROW_NUMBER() OVER (
                        PARTITION BY e.experience_id
                        ORDER BY CASE e.source WHEN 'test' THEN 4 WHEN 'tool' THEN 3 WHEN 'user' THEN 2 ELSE 1 END DESC,
                                 e.created_at DESC, e.rowid DESC
                    ) AS rn
                    FROM evaluations e
                    WHERE e.source IN ('user', 'test', 'tool') AND e.confidence >= 0.6 AND e.success IS NOT NULL
                )
                SELECT x.*, r.success AS evaluation_success, r.score AS evaluation_score,
                       r.source AS evaluation_source, r.confidence AS evaluation_confidence,
                       r.notes AS evaluation_notes
                FROM experiences x JOIN ranked r ON r.experience_id = x.id AND r.rn = 1
                WHERE x.eligible_for_evolution = 1 AND x.technical_success = 1
                ORDER BY x.created_at
                """,
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            skill_rows = connection.execute(
                "SELECT skill_id, skill_version, position FROM experience_skills WHERE experience_id = ? ORDER BY position",
                (row["id"],),
            ).fetchall()
            data["skills"] = [dict(item) for item in skill_rows]
            data["tags"] = json_loads(row["tags_json"], [])
            result.append(data)
        return result

    def _candidate_exists(
        self,
        connection: sqlite3.Connection,
        behavior_key: str,
        candidate_key: str,
        kind: str,
    ) -> bool:
        rows = connection.execute(
            "SELECT pattern_key, status FROM candidates WHERE kind = ?",
            (kind,),
        ).fetchall()
        prefix = behavior_key + "|evidence:"
        for row in rows:
            if row["pattern_key"] == candidate_key:
                return True
            if row["status"] in {"pending", "approved"} and row["pattern_key"].startswith(prefix):
                return True
        return False

    def _discover_patterns(
        self, connection: sqlite3.Connection, experiences: list[dict[str, Any]]
    ) -> list[str]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for experience in experiences:
            groups[experience["pattern_key"]].append(experience)
        created: list[str] = []
        for key, group in groups.items():
            if len(group) < self.min_support:
                continue
            sequences = [
                tuple((item["skill_id"], int(item["skill_version"])) for item in experience["skills"])
                for experience in group
                if experience["skills"]
            ]
            if not sequences:
                continue
            sequence, sequence_support = Counter(sequences).most_common(1)[0]
            if sequence_support < self.min_support:
                continue
            kind = "workflow" if len(sequence) > 1 else "new_skill"
            experience_ids = [item["id"] for item in group]
            behavior_key = f"{key}|sequence:{content_hash(list(sequence))[:16]}"
            candidate_key = f"{behavior_key}|evidence:{content_hash(sorted(experience_ids))[:16]}"
            if self._candidate_exists(connection, behavior_key, candidate_key, kind):
                continue
            sample_tasks = list(dict.fromkeys(item["task"] for item in group))[:5]
            label = group[0]["tags"][0] if group[0]["tags"] else key.replace("kw:", "").replace("tag:", "")
            name = f"经验模式：{label[:28]}"
            if len(sequence) > 1:
                steps = [
                    {
                        "op": "skill_ref",
                        "skill_id": skill_id,
                        "version": version,
                        "as": f"step_{index}",
                    }
                    for index, (skill_id, version) in enumerate(sequence, start=1)
                ]
                steps.append({"op": "combine_outputs"})
            else:
                skill_id, version = sequence[0]
                steps = [{"op": "skill_ref", "skill_id": skill_id, "version": version}]
            spec = {
                "schema_version": 1,
                "executable": True,
                "triggers": {"include": [label], "examples": sample_tasks[:3], "exclude": []},
                "input_schema": {"type": "object", "required": ["text"]},
                "output_schema": {"type": "object"},
                "executor": {"type": "pipeline", "steps": steps},
                "permissions": {"filesystem": [], "network": [], "commands": []},
                "provenance": {"experience_ids": [item["id"] for item in group]},
            }
            validate_spec(spec)
            risk_tier = self._sequence_risk(connection, sequence)
            self.registry._validate_references(connection, spec, risk_tier)
            candidate_id = new_id("candidate")
            scores = [item["evaluation_score"] for item in group if item["evaluation_score"] is not None]
            evidence = {
                "support": len(group),
                "sequence_support": sequence_support,
                "experience_ids": experience_ids,
                "source_pattern_key": key,
                "source_breakdown": dict(Counter(item["evaluation_source"] for item in group)),
                "locked_skill_versions": [
                    {"skill_id": skill_id, "version": version} for skill_id, version in sequence
                ],
            }
            validation = {
                "mode": "evidence_only",
                "mean_feedback_score": round(sum(scores) / len(scores), 4) if scores else None,
                "sample_count": len(group),
                "holdout_count": 0,
                "warning": "MVP 尚未执行独立留出集回放，必须人工审批且只进入实验状态。",
            }
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO candidates VALUES (?, ?, NULL, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL)
                """,
                (
                    candidate_id,
                    kind,
                    candidate_key,
                    name,
                    json_dumps(spec),
                    json_dumps(evidence),
                    json_dumps(validation),
                    risk_tier,
                    utc_now(),
                ),
            )
            if inserted.rowcount == 0:
                continue
            self._audit(
                connection,
                "candidate.created",
                "candidate",
                candidate_id,
                {
                    "kind": kind,
                    "pattern_key": key,
                    "candidate_key": candidate_key,
                    "support": len(group),
                    "risk_tier": risk_tier,
                },
                "system",
            )
            created.append(candidate_id)
        return created

    @staticmethod
    def _sequence_risk(
        connection: sqlite3.Connection, sequence: tuple[tuple[str, int], ...]
    ) -> str:
        risk_order = {"low": 0, "medium": 1, "high": 2}
        risks: list[str] = []
        for skill_id, _ in sequence:
            row = connection.execute("SELECT risk_tier FROM skills WHERE id = ?", (skill_id,)).fetchone()
            if not row:
                raise KeyError(f"经验引用的技能不存在：{skill_id}")
            risks.append(row["risk_tier"])
        return max(risks, key=risk_order.__getitem__)

    def _discover_revisions(
        self, connection: sqlite3.Connection, failures: list[dict[str, Any]]
    ) -> list[str]:
        groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for experience in failures:
            issue = _feedback_issue(experience.get("evaluation_notes", ""))
            if issue is None:
                continue
            for skill in experience["skills"]:
                groups[(skill["skill_id"], int(skill["skill_version"]), issue)].append(experience)
        created: list[str] = []
        for (skill_id, version, issue), group in groups.items():
            if len(group) < self.min_support:
                continue
            key = f"revision:{skill_id}:v{version}:{issue}"
            experience_ids = [item["id"] for item in group]
            candidate_key = f"{key}|evidence:{content_hash(sorted(experience_ids))[:16]}"
            if self._candidate_exists(connection, key, candidate_key, "revision"):
                continue
            row = connection.execute(
                """
                SELECT s.name, s.risk_tier, sv.spec_json FROM skills s
                JOIN skill_versions sv ON sv.skill_id=s.id
                WHERE s.id=? AND sv.version=?
                """,
                (skill_id, version),
            ).fetchone()
            if not row:
                continue
            spec = deepcopy(json_loads(row["spec_json"], {}))
            if issue == "routing":
                triggers = spec.setdefault("triggers", {})
                exclusions = list(triggers.get("exclude", []))
                exclusions.extend(item["task"] for item in group[:5])
                triggers["exclude"] = list(dict.fromkeys(exclusions))
                change = "把明确标注为误路由的任务加入触发反例"
                title = f"{row['name']} 触发边界修订"
            else:
                changed = False
                for step in spec.get("executor", {}).get("steps", []):
                    if step.get("op") == "summarize" and int(step.get("max_sentences", 3)) > 1:
                        step["max_sentences"] = int(step.get("max_sentences", 3)) - 1
                        changed = True
                if not changed:
                    continue
                change = "根据明确的长度反馈缩短摘要句数，不改变触发边界"
                title = f"{row['name']} 输出长度修订"
            validate_spec(spec)
            self.registry._validate_references(connection, spec, row["risk_tier"])
            candidate_id = new_id("candidate")
            evidence = {
                "failure_count": len(group),
                "experience_ids": experience_ids,
                "base_version": version,
                "feedback_issue": issue,
                "change": change,
            }
            validation = {
                "mode": "static_policy_check",
                "permissions_unchanged": True,
                "holdout_count": 0,
                "warning": "只完成静态检查；批准后进入实验状态并可回滚。",
            }
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO candidates VALUES (?, 'revision', ?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL)
                """,
                (
                    candidate_id,
                    skill_id,
                    candidate_key,
                    title,
                    json_dumps(spec),
                    json_dumps(evidence),
                    json_dumps(validation),
                    row["risk_tier"],
                    utc_now(),
                ),
            )
            if inserted.rowcount == 0:
                continue
            self._audit(
                connection,
                "candidate.created",
                "candidate",
                candidate_id,
                {"kind": "revision", "target_skill_id": skill_id, "failures": len(group), "issue": issue},
                "system",
            )
            created.append(candidate_id)
        return created

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.db.read() as connection:
            if status:
                rows = connection.execute(
                    "SELECT * FROM candidates WHERE status = ? ORDER BY created_at DESC", (status,)
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM candidates ORDER BY created_at DESC").fetchall()
            return [self._hydrate_candidate(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.db.read() as connection:
            row = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
            if not row:
                raise KeyError("候选不存在")
            return self._hydrate_candidate(row)

    @staticmethod
    def _hydrate_candidate(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["proposed_spec"] = json_loads(row["proposed_spec_json"], {})
        data["evidence"] = json_loads(row["evidence_json"], {})
        data["validation"] = json_loads(row["validation_json"], {})
        return data

    def approve(self, candidate_id: str, note: str, actor: str = "user") -> dict[str, Any]:
        with self.db.transaction() as connection:
            candidate = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
            if not candidate:
                raise KeyError("候选不存在")
            if candidate["status"] != "pending":
                raise ValueError("候选已经处理")
            spec = json_loads(candidate["proposed_spec_json"], {})
            validate_spec(spec)
            if candidate["kind"] == "revision":
                skill_id = candidate["target_skill_id"]
                skill = connection.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
                if not skill:
                    raise KeyError("目标技能不存在")
                if skill["protected"]:
                    raise PermissionError("受保护的控制核技能不能通过候选审批修改")
                if skill["lifecycle"] in {"quarantined", "archived"}:
                    raise PermissionError(
                        f"目标技能当前为 {skill['lifecycle']}，必须先由人工解除治理状态"
                    )
                evidence = json_loads(candidate["evidence_json"], {})
                base_version = int(evidence.get("base_version", 0))
                if int(skill["active_version"]) != base_version:
                    raise ValueError(
                        f"候选基于 v{base_version}，但当前活跃版本已是 v{skill['active_version']}；"
                        "为避免覆盖新行为，请重新收集证据并生成候选。"
                    )
                version = int(skill["latest_version"]) + 1
                parent = base_version
                self.registry._validate_references(connection, spec, skill["risk_tier"])
                connection.execute(
                    "INSERT INTO skill_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        skill_id,
                        version,
                        parent,
                        json_dumps(spec),
                        content_hash(spec),
                        note or "批准候选修订",
                        actor,
                        candidate_id,
                        utc_now(),
                    ),
                )
                previous_lifecycle = skill["lifecycle"]
                connection.execute("UPDATE skills SET latest_version = ?, lifecycle = 'experimental' WHERE id = ?", (version, skill_id))
                if previous_lifecycle != "experimental":
                    connection.execute(
                        "INSERT INTO skill_lifecycle_events VALUES (?, ?, ?, 'experimental', ?, ?, ?)",
                        (
                            new_id("life"),
                            skill_id,
                            previous_lifecycle,
                            "批准候选修订，进入实验观察",
                            actor,
                            utc_now(),
                        ),
                    )
                self.registry._activate_version(connection, skill_id, version, note or "批准候选修订", actor)
                result = {"skill_id": skill_id, "version": version, "action": "revision_activated"}
            else:
                payload = {
                    "name": candidate["name"],
                    "slug": slugify(candidate["name"]) + "-" + candidate_id[-6:],
                    "description": "由重复可信经验提议，当前仅处于实验状态。",
                    "kind": "workflow" if candidate["kind"] == "workflow" else "atomic",
                    "scope": "domain",
                    "risk_tier": candidate["risk_tier"],
                    "lifecycle": "experimental",
                    "spec": spec,
                    "changelog": note or "由候选批准进入实验",
                }
                skill_id = self.registry._create_skill(
                    connection,
                    payload,
                    actor=actor,
                    origin="evolution",
                    source_candidate_id=candidate_id,
                )
                result = {"skill_id": skill_id, "version": 1, "action": "experimental_skill_created"}
            now = utc_now()
            connection.execute(
                "UPDATE candidates SET status = 'approved', decided_at = ?, decision_note = ? WHERE id = ?",
                (now, note, candidate_id),
            )
            connection.execute(
                "INSERT INTO candidate_decisions VALUES (?, ?, 'approved', ?, ?, ?)",
                (new_id("decision"), candidate_id, note, actor, now),
            )
            self._audit(
                connection,
                "candidate.approved",
                "candidate",
                candidate_id,
                result,
                actor,
            )
            return result

    def reject(self, candidate_id: str, note: str, actor: str = "user") -> None:
        if not note.strip():
            raise ValueError("驳回候选必须填写原因")
        with self.db.transaction() as connection:
            candidate = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
            if not candidate:
                raise KeyError("候选不存在")
            if candidate["status"] != "pending":
                raise ValueError("候选已经处理")
            now = utc_now()
            connection.execute(
                "UPDATE candidates SET status = 'rejected', decided_at = ?, decision_note = ? WHERE id = ?",
                (now, note, candidate_id),
            )
            connection.execute(
                "INSERT INTO candidate_decisions VALUES (?, ?, 'rejected', ?, ?, ?)",
                (new_id("decision"), candidate_id, note, actor, now),
            )
            self._audit(
                connection,
                "candidate.rejected",
                "candidate",
                candidate_id,
                {"note": note},
                actor,
            )

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.db.read() as connection:
            rows = connection.execute(
                "SELECT * FROM evolution_runs ORDER BY started_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
            return [{**dict(row), "summary": json_loads(row["summary_json"], {})} for row in rows]
