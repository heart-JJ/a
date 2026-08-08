from __future__ import annotations

import sqlite3
import string
import re
from copy import deepcopy
from typing import Any

from .db import Database
from .utils import (
    bayesian_reliability,
    clamp,
    content_hash,
    json_dumps,
    json_loads,
    mastery_score,
    new_id,
    similarity,
    slugify,
    utc_now,
)


ALLOWED_OPERATIONS = {
    "normalize_text",
    "summarize",
    "extract_keywords",
    "make_checklist",
    "extract_decisions",
    "structure_notes",
    "format_template",
    "combine_outputs",
    "skill_ref",
}

ALLOWED_TEMPLATE_FIELDS = {
    "text",
    "summary",
    "keywords",
    "checklist",
    "checklist_markdown",
    "decisions",
    "decisions_markdown",
}

DEFAULT_WEIGHTS = {
    "trigger": 0.62,
    "reliability": 0.18,
    "mastery": 0.08,
    "efficiency": 0.07,
    "context": 0.05,
}


SEED_SKILLS: list[dict[str, Any]] = [
    {
        "id": "skill_meta_selector",
        "slug": "meta-skill-selector",
        "name": "技能选择策略",
        "description": "可信控制核中的可解释技能召回与重排策略。只允许优化有界数值参数。",
        "kind": "router",
        "scope": "general",
        "risk_tier": "high",
        "lifecycle": "active",
        "protected": True,
        "spec": {
            "schema_version": 1,
            "executable": False,
            "triggers": {"include": [], "examples": [], "exclude": []},
            "input_schema": {"type": "object", "required": ["task"]},
            "output_schema": {"type": "array"},
            "executor": {"type": "trusted_core", "name": "weighted_selector"},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_meta_evolution_governor",
        "slug": "meta-evolution-governor",
        "name": "进化治理策略",
        "description": "从可信反馈中提出候选，禁止自动改写或直接发布生产技能。",
        "kind": "evaluator",
        "scope": "general",
        "risk_tier": "high",
        "lifecycle": "active",
        "protected": True,
        "spec": {
            "schema_version": 1,
            "executable": False,
            "triggers": {"include": [], "examples": [], "exclude": []},
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "executor": {"type": "trusted_core", "name": "candidate_governor"},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_text_summary",
        "slug": "text-summary",
        "name": "文本摘要",
        "description": "从较长文本中提炼核心句，生成简洁摘要。",
        "kind": "atomic",
        "scope": "domain",
        "risk_tier": "low",
        "lifecycle": "active",
        "spec": {
            "schema_version": 1,
            "executable": True,
            "triggers": {
                "include": ["总结", "摘要", "概括", "提炼", "summarize"],
                "examples": ["请总结这段内容", "把会议记录概括成三点"],
                "exclude": ["逐字翻译", "只提取关键词"],
            },
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            "executor": {"type": "pipeline", "steps": [{"op": "summarize", "max_sentences": 3}]},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_keyword_extraction",
        "slug": "keyword-extraction",
        "name": "关键词提取",
        "description": "从文本中提取高频且有区分度的关键词。",
        "kind": "atomic",
        "scope": "domain",
        "risk_tier": "low",
        "lifecycle": "active",
        "spec": {
            "schema_version": 1,
            "executable": True,
            "triggers": {
                "include": ["关键词", "核心词", "关键概念", "keyword"],
                "examples": ["提取这段话的关键词", "列出八个核心概念"],
                "exclude": ["写完整文章", "翻译全文"],
            },
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object", "properties": {"keywords": {"type": "array"}}},
            "executor": {"type": "pipeline", "steps": [{"op": "extract_keywords", "limit": 8}]},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_checklist_planning",
        "slug": "checklist-planning",
        "name": "任务清单规划",
        "description": "把目标或散乱内容整理为可执行的编号清单。",
        "kind": "atomic",
        "scope": "general",
        "risk_tier": "low",
        "lifecycle": "active",
        "spec": {
            "schema_version": 1,
            "executable": True,
            "triggers": {
                "include": ["计划", "步骤", "清单", "待办", "怎么做", "plan", "checklist"],
                "examples": ["给我一个实施步骤", "把下面内容整理成待办清单"],
                "exclude": ["只需要摘要", "不要行动项"],
            },
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            "executor": {"type": "pipeline", "steps": [{"op": "make_checklist", "limit": 8}]},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_decision_extraction",
        "slug": "decision-extraction",
        "name": "决策提取",
        "description": "从会议或讨论文本中提取明确决定、结论和已确认事项。",
        "kind": "atomic",
        "scope": "domain",
        "risk_tier": "low",
        "lifecycle": "active",
        "spec": {
            "schema_version": 1,
            "executable": True,
            "triggers": {
                "include": ["决策", "决定", "结论", "确认事项", "decision"],
                "examples": ["提取会议中的关键决策", "列出已经确认的结论"],
                "exclude": ["只列待办", "未确定的想法"],
            },
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object", "properties": {"decisions": {"type": "array"}}},
            "executor": {"type": "pipeline", "steps": [{"op": "extract_decisions", "limit": 6}]},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_meeting_notes",
        "slug": "meeting-notes-structure",
        "name": "会议纪要整理",
        "description": "将会议记录整理为摘要、关键词与行动清单。",
        "kind": "workflow",
        "scope": "domain",
        "risk_tier": "low",
        "lifecycle": "active",
        "spec": {
            "schema_version": 1,
            "executable": True,
            "triggers": {
                "include": ["会议纪要", "会议记录", "决策", "行动项", "负责人"],
                "examples": ["把会议记录整理为摘要和待办", "生成一份会议纪要"],
                "exclude": ["会议邀请", "日历排期"],
            },
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object"},
            "executor": {
                "type": "pipeline",
                "steps": [
                    {"op": "skill_ref", "skill_id": "skill_text_summary", "version": 1, "as": "summary"},
                    {"op": "skill_ref", "skill_id": "skill_decision_extraction", "version": 1, "as": "decisions"},
                    {"op": "skill_ref", "skill_id": "skill_checklist_planning", "version": 1, "as": "checklist"},
                    {"op": "skill_ref", "skill_id": "skill_keyword_extraction", "version": 1, "as": "keywords"},
                    {
                        "op": "format_template",
                        "template": "# 会议摘要\n{summary}\n\n# 关键决策\n{decisions_markdown}\n\n# 行动项\n{checklist_markdown}\n\n# 关键词\n{keywords}"
                    },
                ],
            },
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
    {
        "id": "skill_general_text_analysis",
        "slug": "general-text-analysis",
        "name": "通用文本分析",
        "description": "在没有高置信度专用技能时，提供有限、明确的本地文本分析回退。",
        "kind": "atomic",
        "scope": "general",
        "risk_tier": "low",
        "lifecycle": "active",
        "spec": {
            "schema_version": 1,
            "executable": True,
            "fallback": True,
            "triggers": {
                "include": ["分析", "整理", "文本", "内容"],
                "examples": ["分析一下这段内容"],
                "exclude": ["执行命令", "发送消息", "联网搜索"],
            },
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object"},
            "executor": {
                "type": "pipeline",
                "steps": [
                    {"op": "normalize_text"},
                    {"op": "extract_keywords", "limit": 6},
                    {"op": "summarize", "max_sentences": 3},
                    {
                        "op": "format_template",
                        "template": "分析摘要：\n{summary}\n\n关键词：{keywords}",
                    },
                ],
            },
            "permissions": {"filesystem": [], "network": [], "commands": []},
        },
    },
]


class SkillValidationError(ValueError):
    pass


def validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise SkillValidationError("技能定义必须是对象")
    if spec.get("schema_version") != 1:
        raise SkillValidationError("仅支持 schema_version=1")
    triggers = spec.get("triggers")
    if not isinstance(triggers, dict):
        raise SkillValidationError("triggers 必须是对象")
    for key in ("include", "examples", "exclude"):
        values = triggers.get(key, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise SkillValidationError(f"triggers.{key} 必须是字符串数组")
    if "prompt" in spec and (
        not isinstance(spec["prompt"], str) or len(spec["prompt"]) > 12000
    ):
        raise SkillValidationError("prompt 必须是长度不超过 12000 的字符串")
    if not isinstance(spec.get("input_schema", {}), dict) or not isinstance(spec.get("output_schema", {}), dict):
        raise SkillValidationError("input_schema 和 output_schema 必须是对象")
    permissions = spec.get("permissions", {})
    if not isinstance(permissions, dict):
        raise SkillValidationError("permissions 必须是对象")
    for key in ("filesystem", "network", "commands"):
        values = permissions.get(key, [])
        if not isinstance(values, list):
            raise SkillValidationError(f"permissions.{key} 必须是数组")
        if values:
            raise SkillValidationError(f"MVP 不允许技能申请 {key} 权限")
    executor = spec.get("executor")
    if not isinstance(executor, dict):
        raise SkillValidationError("缺少 executor")
    if not spec.get("executable", True):
        if executor.get("type") != "trusted_core":
            raise SkillValidationError("不可执行元技能必须由 trusted_core 表示")
        return
    if executor.get("type") != "pipeline":
        raise SkillValidationError("可执行技能只能使用声明式 pipeline")
    steps = executor.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SkillValidationError("pipeline 至少需要一个步骤")
    if len(steps) > 50:
        raise SkillValidationError("pipeline 最多允许 50 个步骤")
    for step in steps:
        if not isinstance(step, dict) or step.get("op") not in ALLOWED_OPERATIONS:
            raise SkillValidationError(f"包含未注册操作：{step!r}")
        if step.get("op") == "skill_ref" and (not step.get("skill_id") or not step.get("version")):
            raise SkillValidationError("skill_ref 必须锁定 skill_id 与 version")
        if step.get("op") == "skill_ref" and (
            not isinstance(step.get("skill_id"), str)
            or type(step.get("version")) is not int
            or step["version"] < 1
        ):
            raise SkillValidationError("skill_ref 的 skill_id/version 类型无效")
        for numeric_key in ("limit", "max_sentences"):
            if numeric_key in step and (
                type(step[numeric_key]) is not int or not 1 <= step[numeric_key] <= 50
            ):
                raise SkillValidationError(f"{numeric_key} 必须是 1 到 50 的整数")
        if "template" in step and (
            not isinstance(step["template"], str) or len(step["template"]) > 10000
        ):
            raise SkillValidationError("template 必须是长度不超过 10000 的字符串")
        if step.get("op") == "format_template":
            try:
                parsed = list(string.Formatter().parse(step.get("template", "{text}")))
            except ValueError as exc:
                raise SkillValidationError(f"template 格式无效：{exc}") from exc
            fields = [field_name for _, field_name, _, _ in parsed if field_name is not None]
            if len(fields) > 64:
                raise SkillValidationError("template 最多允许 64 个占位符")
            if any(field not in ALLOWED_TEMPLATE_FIELDS for field in fields):
                raise SkillValidationError("template 包含未授权字段或属性访问")
            if any(format_spec or conversion is not None for _, _, format_spec, conversion in parsed):
                raise SkillValidationError("template 不允许格式宽度或类型转换")


class SkillRegistry:
    def __init__(self, database: Database):
        self.db = database

    def seed(self) -> None:
        with self.db.transaction() as connection:
            for item in SEED_SKILLS:
                exists = connection.execute("SELECT 1 FROM skills WHERE slug = ?", (item["slug"],)).fetchone()
                if exists:
                    continue
                self._create_skill(connection, item, actor="system", origin="seed")
            connection.execute(
                "INSERT OR IGNORE INTO meta_parameters(key, value_json, version, updated_at) VALUES (?, ?, 1, ?)",
                ("selector_weights", json_dumps(DEFAULT_WEIGHTS), utc_now()),
            )

    def _audit(
        self,
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

    def _create_skill(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
        actor: str,
        origin: str,
        source_candidate_id: str | None = None,
    ) -> str:
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise SkillValidationError("技能名称必须是 1 到 120 个字符")
        if actor != "system" and payload.get("protected"):
            raise PermissionError("只有可信控制核可以创建受保护技能")
        kind = payload.get("kind", "atomic")
        scope = payload.get("scope", "domain")
        risk_tier = payload.get("risk_tier", "low")
        lifecycle = payload.get("lifecycle", "experimental")
        if kind not in {"atomic", "workflow", "router", "evaluator"}:
            raise SkillValidationError("无效技能类型")
        if scope not in {"general", "domain"}:
            raise SkillValidationError("无效技能作用域")
        if risk_tier not in {"low", "medium", "high"}:
            raise SkillValidationError("无效风险级别")
        if lifecycle not in {"draft", "experimental", "active", "deprecated", "archived", "quarantined"}:
            raise SkillValidationError("无效技能生命周期")
        if actor != "system" and lifecycle not in {"draft", "experimental"}:
            raise PermissionError("人工创建的技能必须先进入 draft 或 experimental，再单独发布")
        spec = deepcopy(payload["spec"])
        validate_spec(spec)
        skill_id = payload.get("id") or new_id("skill")
        self._validate_references(connection, spec, risk_tier)
        slug = payload.get("slug") or slugify(payload["name"])
        now = utc_now()
        connection.execute(
            """
            INSERT INTO skills(
                id, slug, name, description, kind, scope, risk_tier, lifecycle,
                latest_version, active_version, origin, protected, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?, ?)
            """,
            (
                skill_id,
                slug,
                name.strip(),
                payload.get("description", ""),
                kind,
                scope,
                risk_tier,
                lifecycle,
                origin,
                int(bool(payload.get("protected", False))),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO skill_versions(
                skill_id, version, parent_version, spec_json, content_hash, changelog,
                created_by, source_candidate_id, created_at
            ) VALUES (?, 1, NULL, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                json_dumps(spec),
                content_hash(spec),
                payload.get("changelog", "初始版本"),
                actor,
                source_candidate_id,
                now,
            ),
        )
        connection.execute("UPDATE skills SET active_version = 1 WHERE id = ?", (skill_id,))
        connection.execute(
            "INSERT INTO skill_lifecycle_events VALUES (?, ?, NULL, ?, ?, ?, ?)",
            (new_id("life"), skill_id, lifecycle, "创建技能", actor, now),
        )
        connection.execute(
            "INSERT INTO skill_release_events VALUES (?, ?, NULL, 1, ?, ?, ?)",
            (new_id("release"), skill_id, "发布初始版本", actor, now),
        )
        self._audit(connection, "skill.created", "skill", skill_id, {"slug": slug, "version": 1}, actor)
        return skill_id

    def create_skill(self, payload: dict[str, Any], actor: str = "user", origin: str = "manual") -> str:
        with self.db.transaction() as connection:
            return self._create_skill(connection, payload, actor=actor, origin=origin)

    def add_version(
        self,
        skill_id: str,
        spec: dict[str, Any],
        changelog: str,
        actor: str = "user",
        activate: bool = False,
        source_candidate_id: str | None = None,
    ) -> int:
        validate_spec(spec)
        with self.db.transaction() as connection:
            skill = connection.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
            if not skill:
                raise KeyError("技能不存在")
            if skill["protected"] and actor != "system":
                raise PermissionError("受保护的控制核技能不能通过通用版本接口修改")
            self._validate_references(connection, spec, skill["risk_tier"])
            version = int(skill["latest_version"]) + 1
            parent = int(skill["active_version"]) if skill["active_version"] is not None else None
            connection.execute(
                """
                INSERT INTO skill_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    version,
                    parent,
                    json_dumps(spec),
                    content_hash(spec),
                    changelog,
                    actor,
                    source_candidate_id,
                    utc_now(),
                ),
            )
            connection.execute("UPDATE skills SET latest_version = ? WHERE id = ?", (version, skill_id))
            self._audit(
                connection,
                "skill.version_created",
                "skill",
                skill_id,
                {"version": version, "parent_version": parent, "activated": activate},
                actor,
            )
            if activate:
                self._activate_version(connection, skill_id, version, changelog or "批准新版本", actor)
            return version

    @staticmethod
    def _validate_references(
        connection: sqlite3.Connection,
        spec: dict[str, Any],
        declared_risk: str | None = None,
    ) -> None:
        risk_order = {"low": 0, "medium": 1, "high": 2}
        executor = spec.get("executor", {})
        for step in executor.get("steps", []):
            if step.get("op") != "skill_ref":
                continue
            row = connection.execute(
                """
                SELECT sv.spec_json, s.lifecycle, s.risk_tier FROM skill_versions sv
                JOIN skills s ON s.id = sv.skill_id
                WHERE sv.skill_id = ? AND sv.version = ?
                """,
                (step["skill_id"], int(step["version"])),
            ).fetchone()
            if not row:
                raise SkillValidationError(
                    f"引用的技能版本不存在：{step['skill_id']} v{step['version']}"
                )
            child_spec = json_loads(row["spec_json"], {})
            if not child_spec.get("executable", True):
                raise SkillValidationError("组合技能不能引用不可执行的可信控制核")
            if row["lifecycle"] in {"quarantined", "archived", "draft"}:
                raise SkillValidationError(f"不能引用处于 {row['lifecycle']} 状态的技能")
            if declared_risk is not None and risk_order[declared_risk] < risk_order[row["risk_tier"]]:
                raise SkillValidationError(
                    f"组合技能风险级别不能低于子技能：{row['risk_tier']}"
                )

    def _activate_version(
        self,
        connection: sqlite3.Connection,
        skill_id: str,
        version: int,
        reason: str,
        actor: str,
    ) -> None:
        skill = connection.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        if not skill:
            raise KeyError("技能不存在")
        target = connection.execute(
            "SELECT 1 FROM skill_versions WHERE skill_id = ? AND version = ?", (skill_id, version)
        ).fetchone()
        if not target:
            raise KeyError("技能版本不存在")
        previous = skill["active_version"]
        connection.execute("UPDATE skills SET active_version = ? WHERE id = ?", (version, skill_id))
        connection.execute(
            "INSERT INTO skill_release_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (new_id("release"), skill_id, previous, version, reason, actor, utc_now()),
        )
        self._audit(
            connection,
            "skill.version_activated",
            "skill",
            skill_id,
            {"from_version": previous, "to_version": version, "reason": reason},
            actor,
        )

    def activate_version(self, skill_id: str, version: int, reason: str, actor: str = "user") -> None:
        with self.db.transaction() as connection:
            self._activate_version(connection, skill_id, version, reason, actor)

    def set_lifecycle(self, skill_id: str, status: str, reason: str, actor: str = "user") -> None:
        allowed = {"draft", "experimental", "active", "deprecated", "archived", "quarantined"}
        if status not in allowed:
            raise ValueError("无效生命周期")
        with self.db.transaction() as connection:
            skill = connection.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
            if not skill:
                raise KeyError("技能不存在")
            previous = skill["lifecycle"]
            if skill["protected"] and status != previous:
                raise PermissionError("受保护的控制核技能不能通过通用生命周期接口修改")
            connection.execute("UPDATE skills SET lifecycle = ? WHERE id = ?", (status, skill_id))
            connection.execute(
                "INSERT INTO skill_lifecycle_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (new_id("life"), skill_id, previous, status, reason, actor, utc_now()),
            )
            self._audit(
                connection,
                "skill.lifecycle_changed",
                "skill",
                skill_id,
                {"from": previous, "to": status, "reason": reason},
                actor,
            )

    def _stats(self, connection: sqlite3.Connection, skill_id: str, version: int) -> dict[str, Any]:
        usage = connection.execute(
            """
            SELECT COUNT(*) AS uses, COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(SUM(technical_success), 0) AS technical_successes
            FROM skill_usage_events WHERE skill_id = ? AND skill_version = ?
            """,
            (skill_id, version),
        ).fetchone()
        evaluation = connection.execute(
            """
            WITH ranked AS (
                SELECT e.experience_id, e.success,
                       ROW_NUMBER() OVER (PARTITION BY e.experience_id ORDER BY e.created_at DESC, e.rowid DESC) AS rn
                FROM evaluations e
                JOIN skill_usage_events u ON u.experience_id = e.experience_id
                WHERE u.skill_id = ? AND u.skill_version = ?
                  AND e.source IN ('user', 'test', 'tool') AND e.confidence >= 0.6
                  AND e.success IS NOT NULL
            )
            SELECT COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) AS successes,
                   COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0) AS failures
            FROM ranked WHERE rn = 1
            """,
            (skill_id, version),
        ).fetchone()
        successes = int(evaluation["successes"])
        failures = int(evaluation["failures"])
        uses = int(usage["uses"])
        return {
            "uses": uses,
            "successes": successes,
            "failures": failures,
            "unknowns": max(0, uses - successes - failures),
            "reliability": round(bayesian_reliability(successes, failures), 4),
            "mastery": round(mastery_score(uses), 4),
            "avg_latency_ms": round(float(usage["avg_latency_ms"]), 2),
        }

    def list_skills(self, include_inactive: bool = True) -> list[dict[str, Any]]:
        with self.db.read() as connection:
            query = "SELECT * FROM skills"
            params: tuple[Any, ...] = ()
            if not include_inactive:
                query += " WHERE lifecycle IN ('active', 'experimental')"
            query += " ORDER BY protected DESC, name"
            rows = connection.execute(query, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                if row["active_version"] is not None:
                    item["stats"] = self._stats(connection, row["id"], int(row["active_version"]))
                    version_row = connection.execute(
                        "SELECT spec_json FROM skill_versions WHERE skill_id=? AND version=?",
                        (row["id"], int(row["active_version"])),
                    ).fetchone()
                    item["active_spec"] = json_loads(version_row["spec_json"], {}) if version_row else None
                else:
                    item["stats"] = None
                    item["active_spec"] = None
                result.append(item)
            return result

    def get_skill(self, skill_id: str) -> dict[str, Any]:
        with self.db.read() as connection:
            skill = connection.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
            if not skill:
                raise KeyError("技能不存在")
            versions = connection.execute(
                "SELECT * FROM skill_versions WHERE skill_id = ? ORDER BY version DESC", (skill_id,)
            ).fetchall()
            lifecycle = connection.execute(
                "SELECT * FROM skill_lifecycle_events WHERE skill_id = ? ORDER BY created_at DESC", (skill_id,)
            ).fetchall()
            releases = connection.execute(
                "SELECT * FROM skill_release_events WHERE skill_id = ? ORDER BY created_at DESC", (skill_id,)
            ).fetchall()
            data = dict(skill)
            data["versions"] = [
                {**dict(row), "spec": json_loads(row["spec_json"])} for row in versions
            ]
            data["lifecycle_events"] = [dict(row) for row in lifecycle]
            data["release_events"] = [dict(row) for row in releases]
            if skill["active_version"] is not None:
                data["stats"] = self._stats(connection, skill_id, int(skill["active_version"]))
            return data

    def get_active_version(self, skill_id: str) -> dict[str, Any]:
        with self.db.read() as connection:
            row = connection.execute(
                """
                SELECT s.id, s.name, s.kind, s.lifecycle, s.risk_tier, s.active_version,
                       sv.spec_json, sv.content_hash
                FROM skills s
                JOIN skill_versions sv ON sv.skill_id = s.id AND sv.version = s.active_version
                WHERE s.id = ?
                """,
                (skill_id,),
            ).fetchone()
            if not row:
                raise KeyError("技能或活跃版本不存在")
            return {**dict(row), "spec": json_loads(row["spec_json"])}

    def _weights(self, connection: sqlite3.Connection) -> dict[str, float]:
        row = connection.execute("SELECT value_json FROM meta_parameters WHERE key = 'selector_weights'").fetchone()
        weights = json_loads(row["value_json"], DEFAULT_WEIGHTS) if row else DEFAULT_WEIGHTS
        total = sum(float(weights.get(key, 0)) for key in DEFAULT_WEIGHTS)
        if total <= 0:
            return DEFAULT_WEIGHTS.copy()
        return {key: float(weights.get(key, 0)) / total for key in DEFAULT_WEIGHTS}

    def match(self, task: str, context: str = "", limit: int = 5) -> list[dict[str, Any]]:
        with self.db.read() as connection:
            weights = self._weights(connection)
            rows = connection.execute(
                """
                SELECT s.*, sv.spec_json
                FROM skills s
                JOIN skill_versions sv ON sv.skill_id = s.id AND sv.version = s.active_version
                WHERE s.lifecycle IN ('active', 'experimental')
                ORDER BY s.name
                """
            ).fetchall()
            ranked: list[dict[str, Any]] = []
            for row in rows:
                spec = json_loads(row["spec_json"], {})
                if not spec.get("executable", True):
                    continue
                trigger = spec.get("triggers", {})
                positives = [row["name"], row["description"], *trigger.get("include", []), *trigger.get("examples", [])]
                normalized_task = "".join(task.lower().split())
                negated_triggers = [
                    term for term in trigger.get("include", [])
                    if term and _trigger_is_negated(normalized_task, str(term))
                ]
                matched_triggers = [
                    term for term in trigger.get("include", [])
                    if term
                    and "".join(str(term).lower().split()) in normalized_task
                    and term not in negated_triggers
                ]
                positive_scores = [similarity(task, item) for item in positives if item]
                trigger_score = max(positive_scores, default=0.0)
                negative_score = max(
                    (similarity(task, item) for item in trigger.get("exclude", []) if item),
                    default=0.0,
                )
                if negated_triggers:
                    negative_score = 1.0
                trigger_score = clamp(trigger_score - max(0.0, negative_score - 0.35) * 0.9)
                stats = self._stats(connection, row["id"], int(row["active_version"]))
                efficiency = 0.6 if not stats["uses"] else 1.0 / (1.0 + stats["avg_latency_ms"] / 750.0)
                context_score = similarity(context, row["description"]) if context else 0.5
                components = {
                    "trigger": round(trigger_score, 4),
                    "reliability": stats["reliability"],
                    "mastery": stats["mastery"],
                    "efficiency": round(efficiency, 4),
                    "context": round(context_score, 4),
                }
                score = sum(weights[key] * components[key] for key in weights)
                if row["lifecycle"] == "experimental":
                    score *= 0.92
                if spec.get("fallback"):
                    score = max(score, 0.24)
                ranked.append(
                    {
                        "skill_id": row["id"],
                        "name": row["name"],
                        "description": row["description"],
                        "kind": row["kind"],
                        "lifecycle": row["lifecycle"],
                        "risk_tier": row["risk_tier"],
                        "version": int(row["active_version"]),
                        "score": round(clamp(score), 4),
                        "components": components,
                        "weights": weights,
                        "stats": stats,
                        "fallback": bool(spec.get("fallback")),
                        "matched_triggers": matched_triggers,
                        "negated_triggers": negated_triggers,
                        "reason": self._match_reason(components, negative_score, stats),
                    }
                )
            ranked.sort(key=lambda item: (-item["score"], -item["components"]["trigger"], item["name"]))
            return ranked[: max(1, min(limit, 20))]

    @staticmethod
    def _match_reason(components: dict[str, float], negative: float, stats: dict[str, Any]) -> str:
        parts = [f"触发匹配 {components['trigger']:.0%}"]
        if stats["successes"] + stats["failures"]:
            parts.append(f"可信反馈 {stats['successes']}/{stats['successes'] + stats['failures']}")
        else:
            parts.append("暂无可信反馈，使用保守先验")
        if negative > 0.5:
            parts.append("命中反例，已降权")
        return "；".join(parts)


def _trigger_is_negated(normalized_task: str, term: str) -> bool:
    normalized_term = "".join(term.lower().split())
    if not normalized_term or normalized_term not in normalized_task:
        return False
    pattern = rf"(?:不要|无需|不需|别|禁止|不应该|不必|不).{{0,3}}{re.escape(normalized_term)}"
    return re.search(pattern, normalized_task) is not None
