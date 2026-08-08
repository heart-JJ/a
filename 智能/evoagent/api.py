from __future__ import annotations

import json
import mimetypes
import re
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .chat import ConversationBusyError
from .service import EvoAgentService


STATIC_ROOT = Path(__file__).with_name("static")
MAX_BODY_BYTES = 2 * 1024 * 1024


class EvoAgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: EvoAgentService):
        self.service = service
        super().__init__(address, EvoAgentHandler)


class EvoAgentHandler(BaseHTTPRequestHandler):
    server: EvoAgentHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                self._validate_host()
                if method in {"POST", "PATCH", "DELETE"}:
                    self._validate_origin()
                if method == "POST" and path == "/api/chat/stream":
                    self._chat_stream(self._body())
                    return
                result = self._api(method, path, query)
                if result is not None:
                    self._json(HTTPStatus.OK, result)
                return
            if method != "GET":
                self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"})
                return
            self._static(path)
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": _error_text(exc)})
        except PermissionError as exc:
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden", "message": str(exc)})
        except ConversationBusyError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": "conversation_busy", "message": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error", "message": str(exc)})

    def _api(self, method: str, path: str, query: dict[str, list[str]]) -> Any:
        service = self.server.service
        body = self._body() if method in {"POST", "PATCH"} else {}

        if method == "GET" and path == "/api/health":
            return {"status": "ok", "product": "EvoAgent", "mode": "chat"}
        if method == "GET" and path == "/api/conversations":
            include_archived = query.get("include_archived", ["0"])[0] == "1"
            return {"items": service.chat.list_conversations(include_archived)}
        if method == "POST" and path == "/api/conversations":
            return service.chat.create_conversation(body.get("model"), str(body.get("title", "新对话")))
        if method == "GET" and path == "/api/settings":
            return service.chat.get_settings()
        if method == "PATCH" and path == "/api/settings":
            return service.chat.update_settings(body)
        if method == "GET" and path == "/api/models":
            return service.chat.list_models(query.get("refresh", ["0"])[0] == "1")
        if method == "GET" and path == "/api/memories":
            return service.chat.list_memories(int(query.get("limit", ["100"])[0]))
        if method == "GET" and path == "/api/metrics":
            return service.metrics()
        if method == "GET" and path == "/api/skills":
            include_inactive = query.get("include_inactive", ["1"])[0] != "0"
            return {"items": service.skills.list_skills(include_inactive=include_inactive)}
        if method == "POST" and path == "/api/skills":
            skill_id = service.skills.create_skill(body)
            return {"skill_id": skill_id}
        if method == "GET" and path == "/api/experiences":
            limit = int(query.get("limit", ["50"])[0])
            return {"items": service.memory.list_experiences(limit=limit)}
        if method == "GET" and path == "/api/candidates":
            status = query.get("status", [None])[0]
            return {"items": service.evolution.list_candidates(status=status)}
        if method == "GET" and path == "/api/evolution/runs":
            return {"items": service.evolution.list_runs()}
        if method == "GET" and path == "/api/audit":
            limit = int(query.get("limit", ["100"])[0])
            return {"items": service.audit_events(limit=limit)}
        if method == "POST" and path == "/api/tasks/preview":
            supplied_text = body.get("text")
            return service.preview_task(
                str(body.get("task", "")),
                int(body.get("limit", 5)),
                has_supplied_text=supplied_text is not None and bool(str(supplied_text).strip()),
            )
        if method == "POST" and path == "/api/tasks/run":
            tags = body.get("tags", [])
            if isinstance(tags, str):
                tags = [part.strip() for part in tags.split(",") if part.strip()]
            return service.run_task(
                str(body.get("task", "")),
                text=body.get("text"),
                tags=tags,
                salience=float(body.get("salience", 0.5)),
                skill_id=body.get("skill_id"),
            )
        if method == "POST" and path == "/api/evolution/run":
            return service.evolution.run()

        match = re.fullmatch(r"/api/conversations/([^/]+)", path)
        if method == "GET" and match:
            return service.chat.get_conversation(match.group(1))
        if method == "PATCH" and match:
            return service.chat.update_conversation(match.group(1), body)
        if method == "DELETE" and match:
            service.chat.delete_conversation(match.group(1))
            return {"conversation_id": match.group(1), "deleted": True}
        match = re.fullmatch(r"/api/conversations/([^/]+)/messages", path)
        if method == "GET" and match:
            return {"items": service.chat.list_messages(match.group(1))}
        match = re.fullmatch(r"/api/chat/runs/([^/]+)/cancel", path)
        if method == "POST" and match:
            return service.chat.cancel_run(match.group(1))
        match = re.fullmatch(r"/api/messages/([^/]+)/feedback", path)
        if method == "POST" and match:
            positive = _required_bool(body.get("positive"), "positive")
            return service.chat.add_message_feedback(
                match.group(1), positive, str(body.get("notes", ""))
            )

        match = re.fullmatch(r"/api/skills/([^/]+)", path)
        if method == "GET" and match:
            return service.skills.get_skill(match.group(1))
        match = re.fullmatch(r"/api/skills/([^/]+)/versions", path)
        if method == "POST" and match:
            activate = _required_bool(body.get("activate", False), "activate")
            version = service.skills.add_version(
                match.group(1),
                body.get("spec", {}),
                str(body.get("changelog", "人工创建新版本")),
                activate=activate,
            )
            return {"skill_id": match.group(1), "version": version}
        match = re.fullmatch(r"/api/skills/([^/]+)/activate", path)
        if method == "POST" and match:
            service.skills.activate_version(
                match.group(1), int(body["version"]), str(body.get("reason", "人工切换版本"))
            )
            return {"skill_id": match.group(1), "active_version": int(body["version"])}
        match = re.fullmatch(r"/api/skills/([^/]+)/lifecycle", path)
        if method == "POST" and match:
            service.skills.set_lifecycle(
                match.group(1), str(body["status"]), str(body.get("reason", "人工调整状态"))
            )
            return {"skill_id": match.group(1), "lifecycle": body["status"]}

        match = re.fullmatch(r"/api/experiences/([^/]+)", path)
        if method == "GET" and match:
            return service.memory.get_experience(match.group(1))
        match = re.fullmatch(r"/api/experiences/([^/]+)/feedback", path)
        if method == "POST" and match:
            success = body.get("success")
            if success is not None:
                success = _required_bool(success, "success")
            return service.add_feedback(
                match.group(1),
                success,
                body.get("score"),
                notes=str(body.get("notes", "")),
                source="user",
                confidence=1.0,
                evolve_async=_required_bool(body.get("evolve_async", True), "evolve_async"),
            )
        match = re.fullmatch(r"/api/experiences/([^/]+)/eligibility", path)
        if method == "POST" and match:
            eligible = _required_bool(body.get("eligible", True), "eligible")
            service.memory.set_evolution_eligibility(match.group(1), eligible)
            return {"experience_id": match.group(1), "eligible": eligible}

        match = re.fullmatch(r"/api/candidates/([^/]+)", path)
        if method == "GET" and match:
            return service.evolution.get_candidate(match.group(1))
        match = re.fullmatch(r"/api/candidates/([^/]+)/approve", path)
        if method == "POST" and match:
            return service.evolution.approve(match.group(1), str(body.get("note", "批准进入实验")))
        match = re.fullmatch(r"/api/candidates/([^/]+)/reject", path)
        if method == "POST" and match:
            service.evolution.reject(match.group(1), str(body.get("note", "")))
            return {"candidate_id": match.group(1), "status": "rejected"}

        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": "API 路由不存在"})
        return None

    def _body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("写入请求必须使用 application/json")
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON 请求体必须是对象")
        return value

    def _validate_host(self) -> None:
        host = self.headers.get("Host", "")
        try:
            parsed_host = urlparse("//" + host)
            request_hostname = (parsed_host.hostname or "").lower()
            request_port = parsed_host.port
        except ValueError as exc:
            raise PermissionError("无效 Host 请求头") from exc
        server_hostname = str(self.server.server_address[0]).lower()
        server_port = int(self.server.server_address[1])
        loopback_names = {"127.0.0.1", "localhost", "::1"}
        allowed_names = loopback_names if server_hostname in loopback_names else {server_hostname}
        if request_hostname not in allowed_names or request_port != server_port:
            raise PermissionError("拒绝非本机 Host 请求")

    def _validate_origin(self) -> None:
        host = self.headers.get("Host", "")

        origin = self.headers.get("Origin")
        if not origin:
            return
        parsed = urlparse(origin)
        if parsed.scheme != "http" or parsed.netloc.lower() != host.lower():
            raise PermissionError("拒绝跨站写入请求")

    def _chat_stream(self, body: dict[str, Any]) -> None:
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        stream = self.server.service.chat.stream_reply(
            str(body.get("message", "")),
            str(body.get("client_request_id", "")),
            conversation_id=body.get("conversation_id"),
            model=body.get("model"),
        )
        try:
            for item in stream:
                event = str(item.get("event", "message"))
                payload = {key: value for key, value in item.items() if key != "event"}
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                block = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
                self.wfile.write(block)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.error):
            stream.close()

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        if not candidate.is_file():
            candidate = STATIC_ROOT / "index.html"
        data = candidate.read_bytes()
        content_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", (content_type or "application/octet-stream") + ("; charset=utf-8" if candidate.suffix in {".html", ".js", ".css"} else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def _error_text(exc: KeyError) -> str:
    return str(exc.args[0]) if exc.args else "资源不存在"


def _required_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} 必须是 JSON 布尔值")
    return value


def run_server(service: EvoAgentService, host: str = "127.0.0.1", port: int = 8787) -> None:
    server = EvoAgentHTTPServer((host, port), service)
    print(f"EvoAgent 已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在停止……")
    finally:
        server.server_close()
