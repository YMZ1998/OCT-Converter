from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .state import FAViewerState


class FAViewerRequestHandler(BaseHTTPRequestHandler):
    state: FAViewerState
    html_path: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route in {"/", "/index.html"}:
                self._serve_index()
                return

            if route == "/api/state":
                self._send_json(HTTPStatus.OK, self.state.build_state_payload())
                return

            if route == "/api/load":
                filepath = self._require_query_value(query, "path", "Missing path query parameter.")
                vendor_mode = query.get("vendor", ["auto"])[0].strip() or "auto"
                self._send_json(HTTPStatus.OK, self.state.load(filepath, vendor_mode=vendor_mode))
                return

            if route == "/api/pick-file":
                vendor_mode = query.get("vendor", ["auto"])[0].strip() or "auto"
                self._send_json(HTTPStatus.OK, self.state.pick_and_load_file(vendor_mode=vendor_mode))
                return

            if route == "/api/pick-directory":
                vendor_mode = query.get("vendor", ["auto"])[0].strip() or "auto"
                self._send_json(HTTPStatus.OK, self.state.pick_and_load_directory(vendor_mode=vendor_mode))
                return

            match = re.fullmatch(r"/api/frame/(\d+)\.png", route)
            if match:
                contrast = int(query.get("contrast", ["100"])[0])
                brightness = int(query.get("brightness", ["0"])[0])
                payload = self.state.get_frame_png(
                    int(match.group(1)),
                    contrast_percent=contrast,
                    brightness_offset=brightness,
                )
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_index(self) -> None:
        payload = self.html_path.read_text(encoding="utf-8").encode("utf-8")
        self._send_bytes(HTTPStatus.OK, payload, "text/html; charset=utf-8")

    def _require_query_value(
        self,
        query: dict[str, list[str]],
        key: str,
        error_message: str,
    ) -> str:
        value = query.get(key, [""])[0].strip()
        if not value:
            raise ValueError(error_message)
        return value

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, content, "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def build_handler(state: FAViewerState, html_path: Path) -> type[FAViewerRequestHandler]:
    class Handler(FAViewerRequestHandler):
        pass

    Handler.state = state
    Handler.html_path = html_path
    return Handler
