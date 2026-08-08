from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evoagent.api import EvoAgentHTTPServer
from evoagent.service import EvoAgentService
from evoagent.skills import SkillValidationError


class EvoAgentSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "test.db"
        self.service = EvoAgentService(self.db_path, min_support=3)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def skill_by_slug(self, slug: str) -> dict:
        return next(item for item in self.service.skills.list_skills() if item["slug"] == slug)

    def test_seeded_registry_and_explainable_matching(self) -> None:
        skills = self.service.skills.list_skills()
        self.assertGreaterEqual(len(skills), 7)
        matches = self.service.skills.match("请总结这段很长的文章")
        self.assertEqual(matches[0]["name"], "文本摘要")
        self.assertIn("trigger", matches[0]["components"])
        self.assertIn("保守先验", matches[0]["reason"])

        negative = self.service.skills.match("不要总结，只提取关键词")
        self.assertEqual(negative[0]["name"], "关键词提取")

    def test_version_is_immutable_and_active_pointer_can_rollback(self) -> None:
        summary = self.skill_by_slug("text-summary")
        detail = self.service.skills.get_skill(summary["id"])
        spec = deepcopy(detail["versions"][0]["spec"])
        spec["executor"]["steps"][0]["max_sentences"] = 2
        version = self.service.skills.add_version(summary["id"], spec, "摘要缩短", activate=False)
        self.assertEqual(version, 2)
        self.assertEqual(self.service.skills.get_skill(summary["id"])["active_version"], 1)

        with self.assertRaises(sqlite3.IntegrityError):
            with self.service.database.transaction() as connection:
                connection.execute(
                    "UPDATE skill_versions SET changelog='mutated' WHERE skill_id=? AND version=1",
                    (summary["id"],),
                )

        self.service.skills.activate_version(summary["id"], 2, "人工启用测试")
        self.assertEqual(self.service.skills.get_skill(summary["id"])["active_version"], 2)
        self.service.skills.activate_version(summary["id"], 1, "回滚测试")
        detail = self.service.skills.get_skill(summary["id"])
        self.assertEqual(detail["active_version"], 1)
        self.assertEqual(len(detail["versions"]), 2)

    def test_task_run_records_exact_version_and_feedback_is_separate(self) -> None:
        result = self.service.run_task(
            "请总结下面内容",
            text="第一句介绍背景。第二句描述问题。第三句给出方案。第四句说明结果。",
            tags=["摘要测试"],
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["output"]["text"])
        experience = self.service.memory.get_experience(result["experience_id"])
        self.assertEqual(len(experience["skills"]), 1)
        self.assertEqual(experience["skills"][0]["skill_version"], 1)
        self.assertIsNone(experience["latest_evaluation"])

        self.service.add_feedback(
            result["experience_id"], True, 0.9, "断言通过", source="test", evolve_async=False
        )
        experience = self.service.memory.get_experience(result["experience_id"])
        self.assertEqual(experience["latest_evaluation"]["source"], "test")
        skill = self.service.skills.get_skill(experience["skills"][0]["skill_id"])
        self.assertEqual(skill["stats"]["successes"], 1)
        self.assertGreater(skill["stats"]["reliability"], 0.5)

    def test_meeting_workflow_locks_and_runs_four_atomic_skill_versions(self) -> None:
        meeting = self.skill_by_slug("meeting-notes-structure")
        detail = self.service.skills.get_skill(meeting["id"])
        steps = detail["versions"][0]["spec"]["executor"]["steps"]
        references = [step for step in steps if step["op"] == "skill_ref"]
        self.assertEqual(len(references), 4)
        self.assertTrue(all(step["version"] == 1 for step in references))

        result = self.service.run_task(
            "把会议记录整理为摘要、决策和待办",
            text="会议决定周五发布。张三负责接口，李四负责测试。",
        )
        self.assertEqual(result["selection"][0]["skill_id"], meeting["id"])
        self.assertIn("# 关键决策", result["output"]["text"])
        self.assertIn("会议决定周五发布", result["output"]["text"])
        self.assertEqual(len(result["output"]["trace"]), 5)
        self.assertEqual(result["output"]["trace"][0]["child_skill_id"], "skill_text_summary")
        self.assertEqual(self.service.skills.get_skill("skill_text_summary")["stats"]["uses"], 1)

    def test_unsupported_capability_is_explicitly_unhandled(self) -> None:
        for task in ("请联网搜索网页并发送邮件", "请发邮件给小王", "帮我上网查今天新闻", "证明黎曼猜想"):
            with self.subTest(task=task):
                result = self.service.run_task(task)
                self.assertEqual(result["status"], "unhandled")
                self.assertEqual(result["selection"], [])
                self.assertRegex(result["output"]["text"], "伪装|假装")
                experience = self.service.memory.get_experience(result["experience_id"])
                self.assertFalse(experience["technical_success"])

    def test_multi_intent_run_can_evolve_into_reusable_workflow(self) -> None:
        task = "请总结这段内容并提取关键词"
        for index in range(3):
            result = self.service.run_task(
                task,
                text=f"第 {index} 份材料说明版本化技能、可信反馈和安全执行。",
                tags=["摘要关键词组合"],
            )
            self.assertEqual(len(result["selection"]), 2)
            self.assertIn("文本摘要", result["output"]["text"])
            self.assertIn("关键词提取", result["output"]["text"])
            self.service.add_feedback(
                result["experience_id"], True, 0.9, source="test", evolve_async=False
            )

        evolution = self.service.evolution.run()
        candidate = self.service.evolution.get_candidate(evolution["created_candidates"][0])
        self.assertEqual(candidate["kind"], "workflow")
        self.assertEqual(candidate["proposed_spec"]["executor"]["steps"][-1]["op"], "combine_outputs")
        approved = self.service.evolution.approve(candidate["id"], "批准组合工作流")

        reused = self.service.run_task(task, text="新材料需要同时输出摘要和关键词。")
        self.assertEqual(reused["selection"][0]["skill_id"], approved["skill_id"])
        self.assertEqual(len(reused["selection"]), 1)
        self.assertIn("文本摘要", reused["output"]["text"])
        self.assertIn("关键词提取", reused["output"]["text"])

    def test_evolution_needs_trusted_feedback_and_is_idempotent(self) -> None:
        experience_ids = []
        for index in range(3):
            result = self.service.run_task(
                "把会议记录整理为摘要、决策和待办",
                text=f"会议决定完成事项 {index}，负责人在周五前提交。",
                tags=["会议纪要自动化"],
            )
            experience_ids.append(result["experience_id"])

        first = self.service.evolution.run()
        self.assertEqual(first["created_candidates"], [])

        self.service.add_feedback(
            experience_ids[0], True, 0.9, source="model", confidence=1, evolve_async=False
        )
        second = self.service.evolution.run()
        self.assertEqual(second["created_candidates"], [])

        for experience_id in experience_ids:
            self.service.add_feedback(
                experience_id, True, 0.9, "用户认可", source="user", evolve_async=False
            )
        third = self.service.evolution.run()
        self.assertEqual(len(third["created_candidates"]), 1)
        candidate = self.service.evolution.get_candidate(third["created_candidates"][0])
        self.assertEqual(candidate["status"], "pending")
        self.assertEqual(candidate["validation"]["mode"], "evidence_only")
        self.assertEqual(candidate["validation"]["holdout_count"], 0)

        fourth = self.service.evolution.run()
        self.assertEqual(fourth["created_candidates"], [])

    def test_candidate_approval_creates_experimental_skill_with_locked_reference(self) -> None:
        ids = []
        for index in range(3):
            result = self.service.run_task(
                "把会议记录整理为摘要、决策和待办",
                text=f"会议确认行动项 {index}，负责人明天完成。",
                tags=["例会闭环"],
            )
            self.service.add_feedback(
                result["experience_id"], True, 0.95, source="test", evolve_async=False
            )
            ids.append(result["experience_id"])
        run = self.service.evolution.run()
        candidate_id = run["created_candidates"][0]
        outcome = self.service.evolution.approve(candidate_id, "测试批准")
        skill = self.service.skills.get_skill(outcome["skill_id"])
        self.assertEqual(skill["lifecycle"], "experimental")
        step = skill["versions"][0]["spec"]["executor"]["steps"][0]
        self.assertIn("skill_id", step)
        self.assertEqual(step["version"], 1)
        self.assertEqual(self.service.evolution.get_candidate(candidate_id)["status"], "approved")

    def test_repeated_failures_propose_revision_not_silent_mutation(self) -> None:
        summary = self.skill_by_slug("text-summary")
        for index in range(3):
            result = self.service.run_task(
                "请总结下面内容",
                text=f"这是失败边界样本 {index}。",
                tags=["失败边界"],
            )
            self.service.add_feedback(
                result["experience_id"], False, 0.1, "不应命中", source="user", evolve_async=False
            )
        run = self.service.evolution.run()
        candidates = [self.service.evolution.get_candidate(item) for item in run["created_candidates"]]
        revision = next(item for item in candidates if item["kind"] == "revision")
        self.assertEqual(revision["target_skill_id"], summary["id"])
        self.assertEqual(self.service.skills.get_skill(summary["id"])["active_version"], 1)

        outcome = self.service.evolution.approve(revision["id"], "批准边界修订")
        self.assertEqual(outcome["version"], 2)
        detail = self.service.skills.get_skill(summary["id"])
        self.assertEqual(detail["active_version"], 2)
        self.assertEqual(len(detail["versions"]), 2)

    def test_stale_revision_candidate_cannot_overwrite_newer_active_version(self) -> None:
        summary = self.skill_by_slug("text-summary")
        for index in range(3):
            result = self.service.run_task("请总结下面内容", text=f"失败样本 {index}")
            self.service.add_feedback(
                result["experience_id"], False, 0.1, "不应命中", source="user", evolve_async=False
            )
        evolution = self.service.evolution.run()
        revision = next(
            self.service.evolution.get_candidate(item)
            for item in evolution["created_candidates"]
            if self.service.evolution.get_candidate(item)["kind"] == "revision"
        )

        detail = self.service.skills.get_skill(summary["id"])
        manual_spec = deepcopy(detail["versions"][0]["spec"])
        manual_spec["executor"]["steps"][0]["max_sentences"] = 2
        self.service.skills.add_version(summary["id"], manual_spec, "并发人工版本", activate=True)

        with self.assertRaisesRegex(ValueError, "当前活跃版本"):
            self.service.evolution.approve(revision["id"], "不应覆盖")
        self.assertEqual(self.service.skills.get_skill(summary["id"])["active_version"], 2)

    def test_skill_definition_cannot_request_side_effect_permissions(self) -> None:
        payload = {
            "name": "危险技能",
            "spec": {
                "schema_version": 1,
                "executable": True,
                "triggers": {"include": ["危险"], "examples": [], "exclude": []},
                "executor": {"type": "pipeline", "steps": [{"op": "normalize_text"}]},
                "permissions": {"filesystem": ["*"], "network": [], "commands": []},
            },
        }
        with self.assertRaises(SkillValidationError):
            self.service.skills.create_skill(payload)

        malformed = deepcopy(payload)
        malformed["name"] = "格式错误技能"
        malformed["spec"]["permissions"]["filesystem"] = []
        malformed["spec"]["triggers"] = []
        with self.assertRaises(SkillValidationError):
            self.service.skills.create_skill(malformed)

    def test_workflow_cannot_reference_missing_or_unversioned_skill(self) -> None:
        missing = {
            "name": "无效组合",
            "spec": {
                "schema_version": 1,
                "executable": True,
                "triggers": {"include": ["组合"], "examples": [], "exclude": []},
                "executor": {
                    "type": "pipeline",
                    "steps": [{"op": "skill_ref", "skill_id": "missing", "version": 1}],
                },
                "permissions": {"filesystem": [], "network": [], "commands": []},
            },
        }
        with self.assertRaises(SkillValidationError):
            self.service.skills.create_skill(missing)

        del missing["spec"]["executor"]["steps"][0]["version"]
        with self.assertRaises(SkillValidationError):
            self.service.skills.create_skill(missing)

    def test_experience_can_be_excluded_from_evolution(self) -> None:
        result = self.service.run_task("请提取关键词", text="版本 技能 经验 反馈")
        self.service.memory.set_evolution_eligibility(result["experience_id"], False)
        experience = self.service.memory.get_experience(result["experience_id"])
        self.assertFalse(experience["eligible_for_evolution"])

    def test_latest_feedback_is_deterministic_and_append_only(self) -> None:
        result = self.service.run_task("请总结下面内容", text="需要被评价的文本。")
        self.service.add_feedback(
            result["experience_id"], True, 0.9, source="user", evolve_async=False
        )
        self.service.add_feedback(
            result["experience_id"], False, 0.1, source="user", evolve_async=False
        )
        experience = self.service.memory.get_experience(result["experience_id"])
        self.assertFalse(experience["latest_evaluation"]["success"])
        skill = self.service.skills.get_skill(result["selection"][0]["skill_id"])
        self.assertEqual(skill["stats"]["failures"], 1)
        self.assertEqual(skill["stats"]["successes"], 0)

        with self.assertRaises(sqlite3.IntegrityError):
            with self.service.database.transaction() as connection:
                connection.execute(
                    "UPDATE evaluations SET success=1 WHERE experience_id=?",
                    (result["experience_id"],),
                )

    def test_quarantined_dependency_and_protected_meta_skill_are_hard_blocked(self) -> None:
        self.service.skills.set_lifecycle(
            "skill_text_summary", "quarantined", "安全测试隔离"
        )
        meeting = self.skill_by_slug("meeting-notes-structure")
        result = self.service.run_task(
            "把会议记录整理为摘要、决策和待办",
            text="会议决定明天发布。",
            skill_id=meeting["id"],
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("quarantined", result["output"]["error"])

        meta = self.skill_by_slug("meta-skill-selector")
        with self.assertRaises(PermissionError):
            self.service.skills.set_lifecycle(meta["id"], "draft", "不应允许")
        meta_detail = self.service.skills.get_skill(meta["id"])
        with self.assertRaises(PermissionError):
            self.service.skills.add_version(
                meta["id"], deepcopy(meta_detail["versions"][0]["spec"]), "不应允许"
            )

    def test_external_sources_mixed_side_effects_and_negation_are_safe(self) -> None:
        no_content = (
            "分析 https://example.com 的最新内容",
            "请总结 C盘文件",
            "查询天气并总结",
            "总结数据库里的客户记录",
        )
        for task in no_content:
            with self.subTest(task=task):
                self.assertEqual(self.service.run_task(task)["status"], "unhandled")

        external_actions = (
            ("查询天气并总结", "北京"),
            ("从数据库查询客户记录并总结", "客户表"),
            ("用浏览器打开网站并总结", "网页参数"),
            ("用 curl 获取网页并总结", "请求参数"),
            ("总结这段内容并把结果寄给小王", "需要总结的正文"),
            ("总结内容后将 C盘文件移除", "需要总结的正文"),
            ("文本摘要并备份文件", "需要总结的正文"),
            ("请做文本摘要、分享给小王", "需要总结的正文"),
            ("文本摘要和交给小王", "需要总结的正文"),
            ("文本摘要、擦除文件", "需要总结的正文"),
            ("文本摘要和复制文件", "需要总结的正文"),
            ("文本摘要、跑个脚本", "需要总结的正文"),
            ("文本摘要、查看C盘报告", "需要总结的正文"),
            ("提取关键词并不要摘要而改写全文", "需要处理的正文"),
            ("提取关键词并不要摘要而是翻译全文", "需要处理的正文"),
            ("总结并不要发邮件但通知小王", "需要总结的正文"),
            ("总结并别删除文件改为复制文件", "需要总结的正文"),
        )
        for task, text in external_actions:
            with self.subTest(task=task):
                result = self.service.run_task(task, text=text)
                self.assertEqual(result["status"], "unhandled")
                self.assertEqual(result["selection"], [])

        local_mail = self.service.run_task("请总结这封邮件", text="邮件正文介绍了项目目标和交付时间。")
        self.assertEqual(local_mail["status"], "completed")
        local_web = self.service.run_task(
            "请总结网页内容并提取关键词", text="用户已经粘贴了网页正文，内容讨论技能版本与反馈。"
        )
        self.assertEqual(local_web["status"], "completed")
        self.assertEqual(len(local_web["selection"]), 2)
        topical = self.service.run_task(
            "请总结苹果和香蕉的区别", text="苹果和香蕉都是水果，但营养与口感不同。"
        )
        self.assertEqual(topical["status"], "completed")
        for task in (
            "请温和地总结这段内容",
            "请调和后总结这段文本",
            "请概括‘然而’这个词",
            "请总结‘然而’的用法",
            "请总结但丁的人物介绍",
        ):
            with self.subTest(task=task):
                self.assertEqual(self.service.run_task(task, text="需要被总结的正文。第二句补充。")['status'], "completed")

        negated = self.service.run_task(
            "不要总结并提取关键词", text="版本 技能 经验 反馈 安全"
        )
        self.assertEqual(negated["status"], "completed")
        self.assertEqual([item["skill_id"] for item in negated["selection"]], ["skill_keyword_extraction"])

        four_intents = self.service.run_task(
            "请总结、提取关键词、制定计划并列出结论",
            text="项目决定先完成技能版本化，然后整理反馈并制定发布计划。",
        )
        self.assertEqual(four_intents["status"], "completed")
        self.assertEqual(len(four_intents["selection"]), 4)

    def test_template_and_recursive_execution_have_hard_resource_budgets(self) -> None:
        summary = self.service.skills.get_skill("skill_text_summary")
        bomb = deepcopy(summary["versions"][0]["spec"])
        bomb["executor"]["steps"] = [{"op": "format_template", "template": "{text:1000000000}"}]
        with self.assertRaises(SkillValidationError):
            self.service.skills.add_version("skill_text_summary", bomb, "格式宽度炸弹")

        bomb["executor"]["steps"] = [{"op": "format_template", "template": "{text}" * 65}]
        with self.assertRaises(SkillValidationError):
            self.service.skills.add_version("skill_text_summary", bomb, "过多占位符")

        base_spec = {
            "schema_version": 1,
            "executable": True,
            "triggers": {"include": ["预算底层"], "examples": [], "exclude": []},
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object"},
            "executor": {"type": "pipeline", "steps": [{"op": "normalize_text"}]},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        }
        base_id = self.service.skills.create_skill({"name": "预算底层", "spec": base_spec})
        child_spec = deepcopy(base_spec)
        child_spec["triggers"]["include"] = ["预算中层"]
        child_spec["executor"]["steps"] = [
            {"op": "skill_ref", "skill_id": base_id, "version": 1} for _ in range(50)
        ]
        child_id = self.service.skills.create_skill({"name": "预算中层", "kind": "workflow", "spec": child_spec})
        parent_spec = deepcopy(base_spec)
        parent_spec["triggers"]["include"] = ["预算顶层"]
        parent_spec["executor"]["steps"] = [
            {"op": "skill_ref", "skill_id": child_id, "version": 1},
            {"op": "skill_ref", "skill_id": child_id, "version": 1},
        ]
        parent_id = self.service.skills.create_skill({"name": "预算顶层", "kind": "workflow", "spec": parent_spec})
        result = self.service.run_task("预算顶层", text="短文本", skill_id=parent_id)
        self.assertEqual(result["status"], "failed")
        self.assertIn("全局预算", result["output"]["error"])

    def test_feedback_issue_type_changes_only_the_relevant_dimension(self) -> None:
        task = "请总结下面内容"
        for index in range(3):
            result = self.service.run_task(task, text=f"第 {index} 个较长摘要样本。第二句。第三句。第四句。")
            self.service.add_feedback(
                result["experience_id"], False, 0.1, "摘要太长", source="user", evolve_async=False
            )
        evolution = self.service.evolution.run()
        candidate = next(
            self.service.evolution.get_candidate(candidate_id)
            for candidate_id in evolution["created_candidates"]
            if self.service.evolution.get_candidate(candidate_id)["kind"] == "revision"
        )
        self.assertNotIn(task, candidate["proposed_spec"]["triggers"]["exclude"])
        summarize = next(
            step for step in candidate["proposed_spec"]["executor"]["steps"]
            if step["op"] == "summarize"
        )
        self.assertEqual(summarize["max_sentences"], 2)
        self.assertEqual(candidate["evidence"]["feedback_issue"], "summary_length")

        latency_service = EvoAgentService(Path(self.tempdir.name) / "latency.db", min_support=3)
        for index in range(3):
            result = latency_service.run_task("请总结下面内容", text=f"性能样本 {index}。")
            latency_service.add_feedback(
                result["experience_id"], False, 0.1, "处理时间太长", source="user", evolve_async=False
            )
        self.assertEqual(latency_service.evolution.run()["created_candidates"], [])

    def test_rejected_pattern_can_be_reconsidered_only_when_evidence_changes(self) -> None:
        service = EvoAgentService(Path(self.tempdir.name) / "reconsider.db", min_support=2)
        tag = ["可重审模式"]
        for index in range(2):
            result = service.run_task("请总结下面内容", text=f"摘要样本 {index}", tags=tag)
            service.add_feedback(result["experience_id"], True, 0.9, source="test", evolve_async=False)
        first = service.evolution.run()
        self.assertEqual(len(first["created_candidates"]), 1)
        service.evolution.reject(first["created_candidates"][0], "当前不采用")
        self.assertEqual(service.evolution.run()["created_candidates"], [])

        for index in range(3):
            result = service.run_task("请提取关键词", text=f"关键词 样本 {index}", tags=tag)
            service.add_feedback(result["experience_id"], True, 0.9, source="test", evolve_async=False)
        reconsidered = service.evolution.run()
        self.assertEqual(len(reconsidered["created_candidates"]), 1)
        new_candidate = service.evolution.get_candidate(reconsidered["created_candidates"][0])
        self.assertEqual(
            new_candidate["evidence"]["locked_skill_versions"][0]["skill_id"],
            "skill_keyword_extraction",
        )

    def test_evolution_is_atomic_and_inherits_risk_and_child_usage(self) -> None:
        high_spec = {
            "schema_version": 1,
            "executable": True,
            "triggers": {"include": ["高风险处理"], "examples": [], "exclude": []},
            "input_schema": {"type": "object", "required": ["text"]},
            "output_schema": {"type": "object"},
            "executor": {"type": "pipeline", "steps": [{"op": "normalize_text"}]},
            "permissions": {"filesystem": [], "network": [], "commands": []},
        }
        high_id = self.service.skills.create_skill(
            {"name": "高风险受控技能", "risk_tier": "high", "spec": high_spec}
        )
        for index in range(3):
            result = self.service.run_task(
                "高风险处理", text=f"样本 {index}", tags=["高风险轨迹"], skill_id=high_id
            )
            self.service.add_feedback(
                result["experience_id"], True, 0.9, source="test", evolve_async=False
            )
        evolution = self.service.evolution.run()
        candidate = self.service.evolution.get_candidate(evolution["created_candidates"][0])
        self.assertEqual(candidate["risk_tier"], "high")
        approved = self.service.evolution.approve(candidate["id"], "风险继承测试")
        wrapper_run = self.service.run_task(
            "高风险轨迹", text="新样本", skill_id=approved["skill_id"]
        )
        self.assertEqual(wrapper_run["status"], "completed")
        self.assertEqual(self.service.skills.get_skill(high_id)["stats"]["uses"], 4)

        isolated = EvoAgentService(Path(self.tempdir.name) / "atomic.db", min_support=3)
        for index in range(3):
            result = isolated.run_task(
                "请提取关键词", text=f"版本 技能 反馈 {index}", tags=["原子事务"]
            )
            isolated.add_feedback(result["experience_id"], True, 0.9, source="test", evolve_async=False)
        original = isolated.evolution._discover_revisions

        def fail_after_pattern(*_args):
            raise RuntimeError("故障注入")

        isolated.evolution._discover_revisions = fail_after_pattern
        try:
            with self.assertRaisesRegex(RuntimeError, "故障注入"):
                isolated.evolution.run()
        finally:
            isolated.evolution._discover_revisions = original
        self.assertEqual(isolated.evolution.list_candidates(), [])
        self.assertEqual(isolated.evolution.list_runs()[0]["status"], "failed")

    def test_http_feedback_cannot_forge_trusted_sources_or_use_cross_site_posts(self) -> None:
        result = self.service.run_task("请总结下面内容", text="用于 HTTP 反馈测试的正文。")
        server = EvoAgentHTTPServer(("127.0.0.1", 0), self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def post(
            path: str,
            payload: dict,
            content_type: str = "application/json",
            origin: str | None = None,
            host: str | None = None,
        ):
            headers = {"Content-Type": content_type}
            if origin:
                headers["Origin"] = origin
            if host:
                headers["Host"] = host
            request = Request(
                base + path,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            return urlopen(request, timeout=3)

        try:
            with post(
                f"/api/experiences/{result['experience_id']}/feedback",
                {
                    "success": True,
                    "score": 0.9,
                    "source": "test",
                    "confidence": 0.1,
                    "evolve_async": False,
                },
            ):
                pass
            evaluation = self.service.memory.get_experience(result["experience_id"])["latest_evaluation"]
            self.assertEqual(evaluation["source"], "user")
            self.assertEqual(evaluation["confidence"], 1.0)

            with self.assertRaises(HTTPError) as wrong_type:
                post("/api/evolution/run", {}, content_type="text/plain")
            self.assertEqual(wrong_type.exception.code, 400)
            with self.assertRaises(HTTPError) as cross_site:
                post("/api/evolution/run", {}, origin="http://evil.example")
            self.assertEqual(cross_site.exception.code, 403)
            evil_host = f"evil.example:{server.server_address[1]}"
            with self.assertRaises(HTTPError) as rebinding:
                post("/api/evolution/run", {}, origin=f"http://{evil_host}", host=evil_host)
            self.assertEqual(rebinding.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
