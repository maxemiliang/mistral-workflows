#!/usr/bin/env python3
"""Local server for the grok-bot GUI.

Serves the static GUI from this directory and proxies chat requests to the
deployed ``grok-bot`` workflow on Mistral's hosted API, so the API key stays
server-side (env ``MISTRAL_API_KEY`` or a ``.env`` file at the repo root)
instead of living in the browser.

Run:
    python3 gui/server.py            # http://localhost:8000
    MISTRAL_API_KEY=... python3 gui/server.py 8080

Env:
    MISTRAL_API_KEY        Mistral API key (or set it in <repo>/.env)
    GROK_BOT_EXECUTE_URL   Override the workflow execute endpoint (testing)
    PORT                   Port, default 8000
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUI_DIR.parent
DEFAULT_EXECUTE_URL = "https://api.mistral.ai/v1/workflows/grok-bot/execute"


def load_api_key() -> str | None:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if key:
        return key
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("MISTRAL_API_KEY="):
                return line.split("=", 1)[1].strip() or None
    return None


EXECUTE_URL = os.environ.get("GROK_BOT_EXECUTE_URL", DEFAULT_EXECUTE_URL)


class Handler(BaseHTTPRequestHandler):
    server_version = "grok-bot-gui/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- static ----
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            body = json.dumps({"ok": True, "has_key": bool(load_api_key()), "execute_url": EXECUTE_URL}).encode()
            self._json(200, body)
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", ""):
            path = "/index.html"
        target = (GUI_DIR / path.lstrip("/")).resolve()
        if not str(target).startswith(str(GUI_DIR)) or not target.is_file():
            self._json(404, json.dumps({"error": "not found"}).encode())
            return
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        self.wfile.write(target.read_bytes())

    # ---- proxy ----
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._json(404, json.dumps({"error": "not found"}).encode())
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, json.dumps({"message": "invalid JSON body"}).encode())
            return

        api_key = load_api_key()
        if not api_key:
            self._json(
                500,
                json.dumps({
                    "message": "No MISTRAL_API_KEY on the server. Set it in the env or in the repo .env, then restart."
                }).encode(),
            )
            return

        body = json.dumps({"input": payload, "wait_for_result": True}).encode()
        req = urllib.request.Request(
            EXECUTE_URL,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self._json(resp.status, resp.read())
        except urllib.error.HTTPError as err:
            self._json(err.code, err.read())
        except Exception as err:  # noqa: BLE001
            self._json(502, json.dumps({"message": f"upstream error: {err}"}).encode())

    def _json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"grok-bot GUI on http://127.0.0.1:{port}  (proxying to {EXECUTE_URL})")
    print(f"API key: {'set' if load_api_key() else 'MISSING — set MISTRAL_API_KEY'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
