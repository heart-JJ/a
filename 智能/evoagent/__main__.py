from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .api import run_server
from .service import EvoAgentService


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoagent", description="可审计、可回滚的本地自适应技能运行时")
    parser.add_argument("--db", default="data/evoagent.db", help="SQLite 数据库路径")
    parser.add_argument("--version", action="version", version=f"EvoAgent {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="启动本地 Web 控制台")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)

    run = sub.add_parser("run", help="执行一个任务")
    run.add_argument("task")
    run.add_argument("--text")
    run.add_argument("--tag", action="append", default=[])
    run.add_argument("--skill-id")

    preview = sub.add_parser("preview", help="仅查看技能匹配，不执行")
    preview.add_argument("task")

    sub.add_parser("skills", help="列出技能")
    sub.add_parser("experiences", help="列出最近经验")
    sub.add_parser("metrics", help="查看进化指标")
    sub.add_parser("evolve", help="立即运行一次候选发现")

    candidate = sub.add_parser("candidate", help="管理候选")
    candidate_sub = candidate.add_subparsers(dest="candidate_command")
    candidate_sub.add_parser("list")
    approve = candidate_sub.add_parser("approve")
    approve.add_argument("candidate_id")
    approve.add_argument("--note", default="CLI 批准进入实验")
    reject = candidate_sub.add_parser("reject")
    reject.add_argument("candidate_id")
    reject.add_argument("--note", required=True)

    demo = sub.add_parser("demo", help="写入三次多技能可信经验并生成工作流候选")
    demo.add_argument("--tag", default="摘要关键词组合")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    service = EvoAgentService(args.db)
    try:
        if command == "serve":
            run_server(service, args.host if hasattr(args, "host") else "127.0.0.1", args.port if hasattr(args, "port") else 8787)
        elif command == "run":
            _print(service.run_task(args.task, text=args.text, tags=args.tag, skill_id=args.skill_id))
        elif command == "preview":
            _print(service.preview_task(args.task))
        elif command == "skills":
            _print(service.skills.list_skills())
        elif command == "experiences":
            _print(service.memory.list_experiences())
        elif command == "metrics":
            _print(service.metrics())
        elif command == "evolve":
            _print(service.evolution.run())
        elif command == "candidate":
            if args.candidate_command in {None, "list"}:
                _print(service.evolution.list_candidates())
            elif args.candidate_command == "approve":
                _print(service.evolution.approve(args.candidate_id, args.note))
            elif args.candidate_command == "reject":
                service.evolution.reject(args.candidate_id, args.note)
                _print({"candidate_id": args.candidate_id, "status": "rejected"})
        elif command == "demo":
            samples = [
                "不可变技能版本让历史运行可以复现，活跃指针负责发布与回滚。",
                "可信反馈与技术成功分开保存，模型自评不能冒充客观结果。",
                "重复多技能轨迹先生成候选，人工批准后才进入实验目录。",
            ]
            ids = []
            for sample in samples:
                result = service.run_task("请总结下面内容并提取关键词", text=sample, tags=[args.tag])
                service.add_feedback(
                    result["experience_id"], True, 0.9, "演示用可信反馈", evolve_async=False
                )
                ids.append(result["experience_id"])
            _print({"experiences": ids, "evolution": service.evolution.run()})
        return 0
    except (ValueError, KeyError, PermissionError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
