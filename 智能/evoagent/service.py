from __future__ import annotations

import threading
import time
from pathlib import Path
import re
from typing import Any

from .chat import ChatService
from .db import Database
from .evolution import EvolutionEngine
from .executor import ExecutionBudget, ExecutionError, SafeExecutor
from .memory import MemoryStore
from .skills import SkillRegistry
from .utils import json_loads


UNSUPPORTED_CAPABILITY_RULES = (
    (
        "external.read",
        "读取外部来源",
        (
            r"(?:联网|上网|访问网站|浏览网页|抓取网页|下载|上传)",
            r"(?:搜索|检索|查询).{0,12}(?:天气|新闻|网络|网页|网站|数据库|客户记录)",
            r"(?:从|在)(?:数据库|网络|网页|网站).{0,16}(?:查询|搜索|读取|获取|抓取)",
            r"(?:读取|打开).{0,16}(?:文件|文件夹|目录|磁盘|[a-zA-Z]盘)",
            r"(?:浏览器|curl|API|接口).{0,16}(?:打开|获取|调用|请求|访问)",
            r"(?:用|使用|调用).{0,8}(?:浏览器|curl|API|接口)",
            r"(?:获取|拉取).{0,12}(?:网页|网站|外部数据|接口数据)",
            r"https?://|www\.",
        ),
    ),
    (
        "communication.send",
        "发送消息或邮件",
        (
            r"(?:发送|发|寄|投递|转发|回复|通知|发给|寄给).{0,16}(?:邮件|邮箱|消息|短信|微信|用户|客户|小王|小李|他|她)",
            r"(?:把|将).{0,16}(?:发给|寄给|投递到|发送到)",
            r"(?:邮件|邮箱|消息|短信|微信).{0,12}(?:发送|回复|转发|投递)",
            r"(?:同步|抄送|推送).{0,16}(?:给|到|用户|客户|小王|小李)",
        ),
    ),
    (
        "filesystem.write",
        "修改本地文件",
        (
            r"(?:删除|删掉|移除|移动|重命名|覆盖|写入|保存|创建|修改|清空|备份).{0,16}(?:文件|文件夹|目录|磁盘|[a-zA-Z]盘)",
            r"(?:把|将).{0,16}(?:文件|文件夹|目录).{0,12}(?:删除|删掉|移除|移动|重命名|覆盖|清空|备份|复制)",
            r"(?:复制).{0,16}(?:文件|文件夹|目录)",
        ),
    ),
    (
        "system.execute",
        "执行系统或外部操作",
        (
            r"(?:执行|运行|启动|停止|重启).{0,12}(?:命令|程序|脚本|服务|服务器)",
            r"(?:执行|运行)(?:一下)?\s*(?:ls|dir|cmd|powershell|bash|python|node|npm|pnpm)\b",
            r"(?:shell|终端|powershell).{0,12}(?:执行|运行|跑)",
            r"(?:部署|安装应用|控制硬件|服务器操作)",
            r"(?:付款|支付|转账)",
        ),
    ),
)

EXTERNAL_SOURCE_MARKERS = (
    "网页",
    "网站",
    "网络",
    "新闻",
    "最新内容",
    "天气",
    "数据库",
    "客户记录",
    "文件",
    "文件夹",
    "磁盘",
    "邮件",
    "浏览器",
)

COMBINATION_MARKERS = ("并且", "同时", "以及", "并", "与", "和", "+", "、")
MIN_TRIGGER_CONFIDENCE = 0.28
SOFT_INTENT_CONNECTORS = {"和", "与", "、", "，", ","}
ACTION_CLAUSE_PATTERN = re.compile(
    r"(?:请|把|将|给|分享|交给|发送|寄|投递|同步|抄送|推送|删|擦除|移除|清空|复制|移动|"
    r"重命名|备份|保存|写入|读取|查看|打开|搜索|查询|获取|调用|执行|运行|跑|部署|安装|"
    r"支付|转账|生成|输出|列出|提取|总结|摘要|概括|分析|整理|改写|翻译|通知|share|send|delete|run|open)",
    re.IGNORECASE,
)


def _unsupported_capabilities(task: str) -> list[dict[str, str]]:
    detected: list[dict[str, str]] = []
    for capability, label, patterns in UNSUPPORTED_CAPABILITY_RULES:
        matched = next(
            (
                match
                for pattern in patterns
                for match in re.finditer(pattern, task, re.IGNORECASE)
                if not _match_is_negated(task, match.start())
            ),
            None,
        )
        if matched:
            detected.append({"capability": capability, "label": label})
    return detected


def _match_is_negated(task: str, start: int) -> bool:
    prefix = task[max(0, start - 6):start]
    return re.search(r"(?:不要|无需|不需|别|禁止|不必|不再).{0,3}$", prefix) is not None


class EvoAgentService:
    def __init__(self, db_path: str | Path = "data/evoagent.db", min_support: int = 3):
        self.database = Database(db_path)
        self.database.initialize()
        self.skills = SkillRegistry(self.database)
        self.skills.seed()
        self.memory = MemoryStore(self.database)
        self.executor = SafeExecutor(self.database, self.skills)
        self.evolution = EvolutionEngine(self.database, self.skills, min_support=min_support)
        self.chat = ChatService(self.database, self.skills, self.memory)
        self._thread_lock = threading.Lock()
        self._background_thread: threading.Thread | None = None
        self._evolution_requested = False

    def preview_task(
        self, task: str, limit: int = 5, has_supplied_text: bool = False
    ) -> dict[str, Any]:
        if not task.strip():
            raise ValueError("任务不能为空")
        similar = self.memory.similar(task, limit=3)
        context = " ".join(item["task"] for item in similar if item["similarity"] >= 0.35)
        matches = self.skills.match(task, context=context, limit=limit)
        unsupported_capabilities = _unsupported_capabilities(task)
        uncovered = self._uncovered_sub_intents(task)
        if uncovered:
            unsupported_capabilities.append(
                {
                    "capability": "task.uncovered_sub_intent",
                    "label": f"未覆盖的子任务（{' / '.join(uncovered[:2])}）",
                }
            )
        external_reference = None
        if not has_supplied_text:
            external_reference = next(
                (marker for marker in EXTERNAL_SOURCE_MARKERS if marker in task), None
            )
            if not external_reference and re.search(r"https?://|www\.|[a-zA-Z]:[\\/]|[a-zA-Z]盘", task):
                external_reference = "外部地址或路径"
        low_confidence = bool(matches) and matches[0]["components"]["trigger"] < MIN_TRIGGER_CONFIDENCE
        supported = not unsupported_capabilities and external_reference is None and not low_confidence
        if unsupported_capabilities:
            labels = "、".join(item["label"] for item in unsupported_capabilities)
            unsupported_reason = (
                f"任务包含未授权能力：{labels}。为避免静默漏做其中一部分，"
                "本地 MVP 会拒绝整个复合任务，不能伪装已经完成。"
            )
        elif external_reference:
            unsupported_reason = (
                f"任务引用了“{external_reference}”，但没有提供可处理的正文。"
                "本地 MVP 不会自行读取外部来源；请把内容粘贴到待处理文本框。"
            )
        elif low_confidence:
            unsupported_reason = (
                f"最高技能触发置信度只有 {matches[0]['components']['trigger']:.0%}，"
                "系统选择拒答，而不是用通用技能假装完成。"
            )
        else:
            unsupported_reason = None
        return {
            "task": task,
            "matches": matches,
            "similar_experiences": [
                {
                    "id": item["id"],
                    "task": item["task"],
                    "similarity": item["similarity"],
                    "technical_success": bool(item["technical_success"]),
                }
                for item in similar
            ],
            "supported": supported,
            "unsupported_reason": unsupported_reason,
            "unsupported_capabilities": unsupported_capabilities,
        }

    def _uncovered_sub_intents(self, task: str) -> list[str]:
        connectors = sorted(
            set(
                marker for marker in COMBINATION_MARKERS
                if marker not in {"和", "与"}
            ) | {
                "然后", "之后", "后再", "再", "且", "而是", "而要", "但是", "但要", "改为",
                "，", ",", "；", ";",
            },
            key=len,
            reverse=True,
        )
        intent_start = (
            r"(?:请|把|将|给|分享|交给|发送|寄|投递|同步|抄送|推送|删|擦除|移除|清空|复制|"
            r"移动|重命名|备份|保存|写入|读取|查看|打开|搜索|查询|获取|调用|执行|运行|跑|部署|"
            r"安装|支付|转账|生成|输出|列出|提取|总结|摘要|概括|分析|整理|改写|翻译|通知|关键词|核心词|计划|"
            r"步骤|清单|待办|决策|决定|结论|确认事项)"
        )
        split_pattern = (
            "(?:" + "|".join(re.escape(marker) for marker in connectors)
            + rf"|(?:和|与|而|但)(?={intent_start}))"
        )
        pieces = re.split(f"({split_pattern})", task, flags=re.IGNORECASE)
        clauses = [
            (pieces[index].strip(" ，,。；;：:"), pieces[index - 1] if index else None)
            for index in range(0, len(pieces), 2)
            if pieces[index].strip(" ，,。；;：:")
        ]
        if len(clauses) < 2:
            return []
        uncovered: list[str] = []
        for clause, connector in clauses:
            if re.search(r"(?:不要|无需|不需|别|禁止|不必|不再)", clause):
                continue
            clause_matches = self.skills.match(clause, limit=5)
            covered = any(
                (match.get("matched_triggers") or match.get("negated_triggers"))
                and match["components"]["trigger"] >= MIN_TRIGGER_CONFIDENCE
                for match in clause_matches
            )
            if not covered:
                if connector in SOFT_INTENT_CONNECTORS and not ACTION_CLAUSE_PATTERN.search(clause):
                    continue
                uncovered.append(clause[:32])
        return uncovered

    def run_task(
        self,
        task: str,
        text: str | None = None,
        tags: list[str] | None = None,
        salience: float = 0.5,
        skill_id: str | None = None,
    ) -> dict[str, Any]:
        task = task.strip()
        if not task:
            raise ValueError("任务不能为空")
        input_text = (text if text is not None else task).strip()
        preview = self.preview_task(
            task,
            has_supplied_text=text is not None and bool(str(text).strip()),
        )
        started = time.perf_counter()
        selected: list[dict[str, Any]] = []
        if skill_id:
            skill = self.skills.get_skill(skill_id)
            if skill["lifecycle"] not in {"active", "experimental"}:
                raise ValueError("指定技能当前不可执行")
            selected = [
                {
                    "skill_id": skill_id,
                    "name": skill["name"],
                    "version": int(skill["active_version"]),
                    "score": 1.0,
                    "components": {"manual": 1.0},
                    "weights": {},
                    "reason": "用户明确指定",
                }
            ]
        elif preview["matches"]:
            best = preview["matches"][0]
            wants_combination = any(marker in task for marker in COMBINATION_MARKERS)
            exact_intent_matches = [
                match for match in preview["matches"]
                if match["kind"] == "atomic"
                and not match.get("fallback")
                and match.get("matched_triggers")
                and match["components"]["trigger"] >= MIN_TRIGGER_CONFIDENCE
            ]
            if best["kind"] != "workflow" and wants_combination and len(exact_intent_matches) >= 2:
                selected = exact_intent_matches
            else:
                selected = [best]

        if not preview["supported"]:
            latency = (time.perf_counter() - started) * 1000
            output = {
                "status": "unhandled",
                "text": preview["unsupported_reason"],
                "reason": "missing_capability",
            }
            experience_id = self.memory.record_experience(
                task,
                {"text": input_text},
                output,
                [],
                False,
                latency,
                tags,
                salience,
                "拒绝执行未授权能力，等待安装受信任工具或人工处理。",
            )
            return {
                "experience_id": experience_id,
                "status": "unhandled",
                "output": output,
                "selection": [],
                "alternatives": preview["matches"],
                "similar_experiences": preview["similar_experiences"],
                "latency_ms": round(latency, 3),
            }

        if not selected:
            latency = (time.perf_counter() - started) * 1000
            output = {
                "status": "unhandled",
                "text": "当前技能库没有可执行技能。请先创建或批准一个受控技能。",
                "reason": "no_skill",
            }
            experience_id = self.memory.record_experience(
                task, {"text": input_text}, output, [], False, latency, tags, salience
            )
            return {
                "experience_id": experience_id,
                "status": "unhandled",
                "output": output,
                "selection": [],
                "alternatives": [],
                "similar_experiences": preview["similar_experiences"],
                "latency_ms": round(latency, 3),
            }

        invocations: list[dict[str, Any]] | None = None
        try:
            budget = ExecutionBudget()
            executions = [
                self.executor.execute(
                    selection["skill_id"],
                    {"text": input_text, "task": task},
                    version=int(selection["version"]),
                    budget=budget,
                )
                for selection in selected
            ]
            invocations = [
                invocation for execution in executions for invocation in execution.get("invocations", [])
            ]
            technical_success = True
            status = "completed"
            if len(executions) == 1:
                output = {"status": status, **executions[0]}
            else:
                projected_size = sum(
                    len(execution["skill_name"]) + len(execution["text"]) + 5
                    for execution in executions
                ) + 2 * (len(executions) - 1)
                budget.ensure_output_size(projected_size)
                output = {
                    "status": status,
                    "text": "\n\n".join(
                        f"## {execution['skill_name']}\n{execution['text']}" for execution in executions
                    ),
                    "executions": executions,
                    "trace": [
                        {**step, "skill_id": execution["skill_id"], "skill_name": execution["skill_name"]}
                        for execution in executions
                        for step in execution["trace"]
                    ],
                }
            reflection = "执行器完成了声明式步骤；质量仍需用户或测试评价。"
        except ExecutionError as exc:
            technical_success = False
            status = "failed"
            output = {"status": status, "text": "", "error": str(exc)}
            reflection = f"执行失败：{exc}"
        latency = (time.perf_counter() - started) * 1000
        experience_id = self.memory.record_experience(
            task,
            {"text": input_text},
            output,
            selected,
            technical_success,
            latency,
            tags,
            salience,
            reflection,
            invocations,
        )
        return {
            "experience_id": experience_id,
            "status": status,
            "output": output,
            "selection": selected,
            "alternatives": [
                match for match in preview["matches"]
                if match["skill_id"] not in {item["skill_id"] for item in selected}
            ],
            "similar_experiences": preview["similar_experiences"],
            "latency_ms": round(latency, 3),
            "quality_status": "awaiting_feedback",
        }
    def add_feedback(
        self,
        experience_id: str,
        success: bool | None,
        score: float | None,
        notes: str = "",
        source: str = "user",
        confidence: float = 1.0,
        evolve_async: bool = True,
    ) -> dict[str, Any]:
        evaluation_id = self.memory.add_feedback(
            experience_id,
            success,
            score,
            notes=notes,
            source=source,
            confidence=confidence,
        )
        if evolve_async and source in {"user", "test", "tool"}:
            self._start_background_evolution()
        return {"evaluation_id": evaluation_id, "evolution_scheduled": bool(evolve_async)}

    def _start_background_evolution(self) -> None:
        def target() -> None:
            while True:
                with self._thread_lock:
                    if not self._evolution_requested:
                        self._background_thread = None
                        return
                    self._evolution_requested = False
                try:
                    self.evolution.run()
                except RuntimeError:
                    # A manual scan may briefly own the engine. Preserve the
                    # request so the worker tries again after that scan exits.
                    with self._thread_lock:
                        self._evolution_requested = True
                    time.sleep(0.05)

        with self._thread_lock:
            self._evolution_requested = True
            if self._background_thread and self._background_thread.is_alive():
                return
            self._background_thread = threading.Thread(
                target=target, name="evoagent-evolution", daemon=True
            )
            self._background_thread.start()

    def metrics(self) -> dict[str, Any]:
        with self.database.read() as connection:
            skill_total = connection.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            lifecycle_rows = connection.execute(
                "SELECT lifecycle, COUNT(*) AS count FROM skills GROUP BY lifecycle"
            ).fetchall()
            kind_rows = connection.execute("SELECT kind, COUNT(*) AS count FROM skills GROUP BY kind").fetchall()
            experience = connection.execute(
                "SELECT COUNT(*) AS count, COALESCE(AVG(latency_ms), 0) AS avg_latency FROM experiences"
            ).fetchone()
            trusted = connection.execute(
                """
                WITH ranked AS (
                    SELECT experience_id, success,
                           ROW_NUMBER() OVER (PARTITION BY experience_id ORDER BY created_at DESC, rowid DESC) rn
                    FROM evaluations
                    WHERE source IN ('user','test','tool') AND confidence >= 0.6 AND success IS NOT NULL
                )
                SELECT COUNT(*) AS count, COALESCE(SUM(success), 0) AS successes FROM ranked WHERE rn = 1
                """
            ).fetchone()
            pending = connection.execute("SELECT COUNT(*) FROM candidates WHERE status = 'pending'").fetchone()[0]
            candidate_total = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            evolution_runs = connection.execute("SELECT COUNT(*) FROM evolution_runs").fetchone()[0]
            latest_evolution = connection.execute(
                "SELECT * FROM evolution_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            evaluated = int(trusted["count"])
            successes = int(trusted["successes"])
            return {
                "skills": {
                    "total": int(skill_total),
                    "by_lifecycle": {row["lifecycle"]: int(row["count"]) for row in lifecycle_rows},
                    "by_kind": {row["kind"]: int(row["count"]) for row in kind_rows},
                },
                "experiences": {
                    "total": int(experience["count"]),
                    "avg_latency_ms": round(float(experience["avg_latency"]), 2),
                    "trusted_evaluated": evaluated,
                    "trusted_successes": successes,
                    "trusted_success_rate": round(successes / evaluated, 4) if evaluated else None,
                },
                "candidates": {"total": int(candidate_total), "pending": int(pending)},
                "evolution": {
                    "runs": int(evolution_runs),
                    "latest": (
                        {**dict(latest_evolution), "summary": json_loads(latest_evolution["summary_json"], {})}
                        if latest_evolution
                        else None
                    ),
                },
            }

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY created_at DESC, id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
            return [{**dict(row), "payload": json_loads(row["payload_json"], {})} for row in rows]
