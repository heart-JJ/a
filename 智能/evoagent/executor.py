from __future__ import annotations

import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .db import Database
from .skills import SkillRegistry
from .utils import json_loads, keyword_list, split_sentences, text_tokens


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecutionBudget:
    """A shared budget for an entire execution tree, not just one pipeline."""

    max_invocations: int = 64
    max_steps: int = 256
    max_output_chars: int = 1_000_000
    max_total_output_chars: int = 4_000_000
    timeout_seconds: float = 5.0
    invocations: int = 0
    steps: int = 0
    total_output_chars: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def _check_time(self) -> None:
        if time.perf_counter() - self.started_at > self.timeout_seconds:
            raise ExecutionError("技能执行超过时间预算")

    def consume_invocation(self) -> None:
        self._check_time()
        self.invocations += 1
        if self.invocations > self.max_invocations:
            raise ExecutionError("技能组合调用次数超过全局预算")

    def consume_step(self) -> None:
        self._check_time()
        self.steps += 1
        if self.steps > self.max_steps:
            raise ExecutionError("技能组合步骤数超过全局预算")

    def ensure_output_size(self, size: int) -> None:
        self._check_time()
        if size > self.max_output_chars:
            raise ExecutionError("技能输出超过 100 万字符限制")
        if self.total_output_chars + size > self.max_total_output_chars:
            raise ExecutionError("技能组合累计输出超过全局预算")

    def account_output(self, value: str) -> None:
        size = len(value)
        self.ensure_output_size(size)
        self.total_output_chars += size


class SafeExecutor:
    """Executes declaration-only pipelines.

    There is deliberately no eval, exec, subprocess, dynamic import, filesystem
    access or network access here. Evolved skills can only compose registered
    operations defined in this class.
    """

    def __init__(self, database: Database, registry: SkillRegistry):
        self.db = database
        self.registry = registry

    MAX_INPUT_CHARS = 200_000

    def execute(
        self,
        skill_id: str,
        payload: dict[str, Any],
        version: int | None = None,
        stack: tuple[tuple[str, int], ...] = (),
        budget: ExecutionBudget | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        budget = budget or ExecutionBudget()
        budget.consume_invocation()
        skill = self._load(skill_id, version)
        if skill["lifecycle"] in {"quarantined", "archived", "draft"}:
            raise ExecutionError(f"技能处于 {skill['lifecycle']} 状态，运行时已阻断")
        key = (skill_id, int(skill["version"]))
        if key in stack:
            raise ExecutionError("检测到技能递归引用")
        if len(stack) >= 8:
            raise ExecutionError("技能组合深度超过限制")
        spec = skill["spec"]
        if not spec.get("executable", True):
            raise ExecutionError("该技能属于可信控制核，不能作为任务技能直接执行")
        executor = spec.get("executor", {})
        if executor.get("type") != "pipeline":
            raise ExecutionError("仅允许声明式 pipeline 执行")
        text = str(payload.get("text") or payload.get("task") or "").strip()
        if not text:
            raise ExecutionError("输入文本不能为空")
        if len(text) > self.MAX_INPUT_CHARS:
            raise ExecutionError("输入文本超过 20 万字符限制")
        state: dict[str, Any] = {
            "input_text": text,
            "text": text,
            "summary": "",
            "keywords": [],
            "checklist": [],
            "decisions": [],
            "child_outputs": {},
            "child_invocations": [],
            "trace": [],
        }
        for index, step in enumerate(executor.get("steps", []), start=1):
            budget.consume_step()
            step_started = time.perf_counter()
            op = step["op"]
            self._run_operation(op, step, state, stack + (key,), budget)
            budget.account_output(str(state.get("text", "")))
            details = state.pop("_step_details", {})
            state["trace"].append(
                {
                    "index": index,
                    "operation": op,
                    "duration_ms": round((time.perf_counter() - step_started) * 1000, 3),
                    **details,
                }
            )
        if not state.get("text"):
            state["text"] = state.get("summary") or ""
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "skill_id": skill_id,
            "skill_name": skill["name"],
            "version": int(skill["version"]),
            "text": state.get("text", ""),
            "summary": state.get("summary", ""),
            "keywords": state.get("keywords", []),
            "checklist": state.get("checklist", []),
            "decisions": state.get("decisions", []),
            "trace": state["trace"],
            "duration_ms": duration_ms,
            "invocations": [
                {
                    "skill_id": skill_id,
                    "skill_name": skill["name"],
                    "version": int(skill["version"]),
                    "duration_ms": duration_ms,
                },
                *state["child_invocations"],
            ],
        }

    def _load(self, skill_id: str, version: int | None) -> dict[str, Any]:
        with self.db.read() as connection:
            if version is None:
                row = connection.execute(
                    """
                    SELECT s.name, s.lifecycle, s.active_version AS version, sv.spec_json
                    FROM skills s
                    JOIN skill_versions sv ON sv.skill_id = s.id AND sv.version = s.active_version
                    WHERE s.id = ?
                    """,
                    (skill_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT s.name, s.lifecycle, sv.version, sv.spec_json
                    FROM skills s JOIN skill_versions sv ON sv.skill_id = s.id
                    WHERE s.id = ? AND sv.version = ?
                    """,
                    (skill_id, version),
                ).fetchone()
            if not row:
                raise ExecutionError("技能版本不存在")
            return {**dict(row), "spec": json_loads(row["spec_json"])}

    def _run_operation(
        self,
        op: str,
        step: dict[str, Any],
        state: dict[str, Any],
        stack: tuple[tuple[str, int], ...],
        budget: ExecutionBudget,
    ) -> None:
        if op == "normalize_text":
            state["text"] = self._normalize(state["text"])
        elif op == "extract_keywords":
            state["keywords"] = keyword_list(state["text"], int(step.get("limit", 8)))
            if len(step) == 1 or step.get("as_output", True):
                state["text"] = "、".join(state["keywords"])
        elif op == "summarize":
            source = state.get("input_text") if state.get("keywords") else state["text"]
            state["summary"] = self._summarize(str(source), int(step.get("max_sentences", 3)))
            state["text"] = state["summary"]
        elif op == "make_checklist":
            state["checklist"] = self._checklist(state.get("input_text", state["text"]), int(step.get("limit", 8)))
            state["text"] = "\n".join(f"{index}. {item}" for index, item in enumerate(state["checklist"], 1))
        elif op == "extract_decisions":
            state["decisions"] = self._decisions(
                state.get("input_text", state["text"]), int(step.get("limit", 6))
            )
            state["text"] = "\n".join(f"- {item}" for item in state["decisions"])
        elif op == "structure_notes":
            self._structure_notes(state)
        elif op == "format_template":
            template = str(step.get("template", "{text}"))
            state["text"] = self._render_template(
                template,
                {
                    "text": str(state.get("text", "")),
                    "summary": str(state.get("summary", "")),
                    "keywords": "、".join(state.get("keywords", [])),
                    "checklist": "\n".join(state.get("checklist", [])),
                    "checklist_markdown": "\n".join(f"- [ ] {item}" for item in state.get("checklist", [])),
                    "decisions": "\n".join(state.get("decisions", [])),
                    "decisions_markdown": "\n".join(f"- {item}" for item in state.get("decisions", [])),
                },
                budget,
            )
        elif op == "combine_outputs":
            sections = []
            projected_size = 0
            for child in state.get("child_outputs", {}).values():
                section = f"## {child['skill_name']}\n{child['text']}"
                projected_size += len(section) + (2 if sections else 0)
                budget.ensure_output_size(projected_size)
                sections.append(section)
            if not sections:
                raise ExecutionError("组合输出没有可用的子技能结果")
            state["text"] = "\n\n".join(sections)
        elif op == "skill_ref":
            child = self.execute(
                step["skill_id"],
                {"text": state.get("input_text", state["text"])},
                version=int(step["version"]),
                stack=stack,
                budget=budget,
            )
            state["child_invocations"].extend(child.get("invocations", []))
            state["_step_details"] = {
                "child_skill_id": child["skill_id"],
                "child_skill_name": child["skill_name"],
                "child_version": child["version"],
                "child_trace": child["trace"],
            }
            alias = step.get("as")
            if alias:
                state["child_outputs"][str(alias)] = child
                if alias == "summary":
                    state["summary"] = child.get("summary") or child["text"]
                elif alias == "keywords":
                    state["keywords"] = child.get("keywords") or [child["text"]]
                elif alias == "checklist":
                    state["checklist"] = child.get("checklist") or [child["text"]]
                elif alias == "decisions":
                    state["decisions"] = child.get("decisions") or [child["text"]]
            if not alias or step.get("as_output", False):
                state["text"] = child["text"]
        else:
            raise ExecutionError(f"未注册操作：{op}")

    @staticmethod
    def _render_template(
        template: str, values: dict[str, str], budget: ExecutionBudget
    ) -> str:
        """Render fixed placeholders without Python's width/conversion machinery."""

        parts: list[str] = []
        projected_size = 0
        try:
            parsed = string.Formatter().parse(template)
            for literal, field_name, format_spec, conversion in parsed:
                if format_spec or conversion is not None:
                    raise ExecutionError("模板不允许格式宽度或类型转换")
                projected_size += len(literal)
                budget.ensure_output_size(projected_size)
                parts.append(literal)
                if field_name is not None:
                    if field_name not in values:
                        raise ExecutionError("模板包含未授权字段")
                    value = values[field_name]
                    projected_size += len(value)
                    budget.ensure_output_size(projected_size)
                    parts.append(value)
        except ValueError as exc:
            raise ExecutionError(f"模板格式无效：{exc}") from exc
        return "".join(parts)

    @staticmethod
    def _normalize(text: str) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r", "").split("\n")]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _summarize(text: str, max_sentences: int) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return text.strip()
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        document_frequency: Counter[str] = Counter()
        for sentence in sentences:
            document_frequency.update(text_tokens(sentence))
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            tokens = text_tokens(sentence)
            density = sum(document_frequency[token] for token in tokens) / max(1, len(tokens))
            position_bonus = 1.0 / (1.0 + index * 0.15)
            length_penalty = 0.75 if len(sentence) > 220 else 1.0
            scored.append((density * position_bonus * length_penalty, index, sentence))
        selected = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda item: item[1])
        return " ".join(sentence for _, _, sentence in selected)

    @staticmethod
    def _checklist(text: str, limit: int) -> list[str]:
        clauses: list[str] = []
        for sentence in split_sentences(text):
            clauses.extend(part.strip(" -—：:") for part in re.split(r"[,，、]|(?:\s+-\s+)", sentence) if part.strip())
        useful = []
        seen = set()
        for clause in clauses:
            normalized = clause.strip()
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            useful.append(normalized)
            if len(useful) >= limit:
                break
        if len(useful) <= 1:
            goal = useful[0] if useful else text.strip()
            return [
                f"明确目标与验收标准：{goal}",
                "收集必要输入并确认约束",
                "按优先级执行核心步骤",
                "验证结果并记录反馈",
            ][:limit]
        return useful[:limit]

    def _structure_notes(self, state: dict[str, Any]) -> None:
        source = state["input_text"]
        summary = self._summarize(source, 3)
        keywords = keyword_list(source, 8)
        checklist = self._checklist(source, 6)
        decision_lines = [
            sentence for sentence in split_sentences(source)
            if any(marker in sentence for marker in ("决定", "决策", "确认", "同意", "采用", "结论"))
        ][:5]
        state.update(summary=summary, keywords=keywords, checklist=checklist)
        decisions = decision_lines or ["暂未识别到明确决策，请人工确认。"]
        state["text"] = (
            "# 会议摘要\n"
            f"{summary}\n\n"
            "# 关键决策\n"
            + "\n".join(f"- {item}" for item in decisions)
            + "\n\n# 行动项\n"
            + "\n".join(f"- [ ] {item}" for item in checklist)
            + "\n\n# 关键词\n"
            + "、".join(keywords)
        )

    @staticmethod
    def _decisions(text: str, limit: int) -> list[str]:
        markers = ("决定", "决策", "确认", "同意", "采用", "结论", "确定", "批准")
        decisions = [sentence for sentence in split_sentences(text) if any(marker in sentence for marker in markers)]
        return decisions[:limit] or ["暂未识别到明确决策，请人工确认。"]

