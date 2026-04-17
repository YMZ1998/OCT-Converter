from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .state import ViewerState


def create_app(state: ViewerState, html_path: Path) -> FastAPI:
    app = FastAPI(title="OCT Web Viewer", docs_url=None, redoc_url=None, openapi_url=None)

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
        vendor_mode = vendor.strip()
        if not filepath:
            raise ValueError("Missing path query parameter.")
        return state.load(filepath, vendor_mode=vendor_mode)

    @app.get("/api/pick-file")
    async def pick_file(vendor: str = "auto") -> dict:
        return state.pick_and_load(vendor_mode=vendor.strip())

    @app.post("/api/upload")
    async def upload_file(
        vendor: str = "auto",
        file: UploadFile = File(...),
    ) -> dict:
        if not file.filename:
            raise ValueError("Missing uploaded filename.")
        return state.load_uploaded_file(file.filename, file.file, vendor_mode=vendor.strip())

    @app.get("/api/volume/{volume_index}/slice/{slice_index}.png")
    async def get_slice_png(
        volume_index: int,
        slice_index: int,
        contrast: int = 100,
        brightness: int = 0,
    ) -> Response:
        payload = state.get_slice_png(
            volume_index=volume_index,
            slice_index=slice_index,
            contrast_percent=contrast,
            brightness_offset=brightness,
        )
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/volume/{volume_index}/projection.png")
    async def get_projection_png(volume_index: int) -> Response:
        payload = state.get_projection_png(volume_index)
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/volume/{volume_index}/contours/{slice_index}.json")
    async def get_contours(volume_index: int, slice_index: int) -> dict:
        return state.get_contours(volume_index=volume_index, slice_index=slice_index)

    @app.get("/api/fundus/{fundus_index}.png")
    async def get_fundus_png(fundus_index: int) -> Response:
        payload = state.get_fundus_png(fundus_index)
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/volume/{volume_index}/fundus-view.png")
    async def get_volume_fundus_view_png(volume_index: int) -> Response:
        payload = state.get_volume_fundus_view_png(volume_index)
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    return app
