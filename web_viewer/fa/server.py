from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .state import FAViewerState


def create_app(state: FAViewerState, html_path: Path) -> FastAPI:
    app = FastAPI(title="FA Web Viewer", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            return JSONResponse(status_code=404, content={"error": f"Unknown route: {request.url.path}"})
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    async def serve_index() -> HTMLResponse:
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/api/state")
    async def get_state() -> dict:
        return state.build_state_payload()

    @app.get("/api/load")
    async def load_data(path: str, vendor: str = "auto") -> dict:
        filepath = path.strip()
        vendor_mode = vendor.strip() or "auto"
        if not filepath:
            raise ValueError("Missing path query parameter.")
        return state.load(filepath, vendor_mode=vendor_mode)

    @app.get("/api/pick-file")
    async def pick_file(vendor: str = "auto") -> dict:
        vendor_mode = vendor.strip() or "auto"
        return state.pick_and_load_file(vendor_mode=vendor_mode)

    @app.get("/api/pick-directory")
    async def pick_directory(vendor: str = "auto") -> dict:
        vendor_mode = vendor.strip() or "auto"
        return state.pick_and_load_directory(vendor_mode=vendor_mode)

    @app.get("/api/frame/{frame_index}.png")
    async def get_frame_png(
        frame_index: int,
        contrast: int = 100,
        brightness: int = 0,
    ) -> Response:
        payload = state.get_frame_png(
            frame_index,
            contrast_percent=contrast,
            brightness_offset=brightness,
        )
        return Response(
            content=payload,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )

    return app
