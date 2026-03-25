from __future__ import annotations

import cgi
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .state import ViewerState


class ViewerRequestHandler(BaseHTTPRequestHandler):
    state: ViewerState
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
                self._send_state_payload(self.state.build_state_payload())
                return

            if route == "/api/load":
                filepath = self._require_query_value(query, "path", "Missing path query parameter.")
                payload = self.state.load(filepath, vendor_mode=self._query_vendor_mode(query))
                self._send_state_payload(payload)
                return

            if route == "/api/pick-file":
                payload = self.state.pick_and_load(vendor_mode=self._query_vendor_mode(query))
                self._send_state_payload(payload)
                return

            match = re.fullmatch(r"/api/volume/(\d+)/slice/(\d+)\.png", route)
            if match:
                contrast = int(query.get("contrast", ["100"])[0])
                brightness = int(query.get("brightness", ["0"])[0])
                payload = self.state.get_slice_png(
                    volume_index=int(match.group(1)),
                    slice_index=int(match.group(2)),
                    contrast_percent=contrast,
                    brightness_offset=brightness,
                )
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            match = re.fullmatch(r"/api/volume/(\d+)/projection\.png", route)
            if match:
                payload = self.state.get_projection_png(int(match.group(1)))
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            match = re.fullmatch(r"/api/volume/(\d+)/contours/(\d+)\.json", route)
            if match:
                payload = self.state.get_contours(
                    volume_index=int(match.group(1)),
                    slice_index=int(match.group(2)),
                )
                self._send_json(HTTPStatus.OK, payload)
                return

            match = re.fullmatch(r"/api/fundus/(\d+)\.png", route)
            if match:
                payload = self.state.get_fundus_png(int(match.group(1)))
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            match = re.fullmatch(r"/api/volume/(\d+)/fundus-view\.png", route)
            if match:
                payload = self.state.get_volume_fundus_view_png(int(match.group(1)))
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route == "/api/upload":
                payload = self._handle_upload(vendor_mode=self._query_vendor_mode(query))
                self._send_state_payload(payload)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_index(self) -> None:
        payload = self.html_path.read_text(encoding="utf-8").encode("utf-8")
        self._send_bytes(HTTPStatus.OK, payload, "text/html; charset=utf-8")

    def _query_vendor_mode(self, query: dict[str, list[str]]) -> str:
        return query.get("vendor", ["auto"])[0].strip()

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

    def _send_state_payload(self, payload: dict[str, Any]) -> None:
        self._send_json(HTTPStatus.OK, payload)

    def _handle_upload(self, vendor_mode: str = "auto") -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Upload requests must use multipart/form-data.")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        if "file" not in form:
            raise ValueError("Missing uploaded file field.")

        file_field = form["file"]
        if isinstance(file_field, list):
            file_field = file_field[0]

        if not getattr(file_field, "filename", ""):
            raise ValueError("Missing uploaded filename.")

        file_handle = getattr(file_field, "file", None)
        if file_handle is None:
            raise ValueError("Uploaded file payload is unavailable.")

        return self.state.load_uploaded_file(file_field.filename, file_handle, vendor_mode=vendor_mode)

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


def build_handler(state: ViewerState, html_path: Path) -> type[ViewerRequestHandler]:
    class Handler(ViewerRequestHandler):
        pass

    Handler.state = state
    Handler.html_path = html_path
    return Handler
