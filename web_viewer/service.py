"""Unified FastAPI service for the OCT and FA web viewers."""

from __future__ import annotations

import argparse
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .fa.state import FAViewerState
from .oct.state import ViewerState


def _install_error_handlers(app: FastAPI) -> None:
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


def _rewrite_html_api_prefix(html_path: Path, route_prefix: str) -> str:
    html = html_path.read_text(encoding="utf-8")
    replacements = {
        '"/api/': f'"{route_prefix}/api/',
        "'/api/": f"'{route_prefix}/api/",
        "`/api/": f"`{route_prefix}/api/",
    }
    for source, target in replacements.items():
        html = html.replace(source, target)
    return html


def _has_python_multipart() -> bool:
    return importlib.util.find_spec("multipart") is not None


def create_service_app(
    *,
    oct_state: ViewerState | None = None,
    fa_state: FAViewerState | None = None,
    oct_html_path: Path | None = None,
    fa_html_path: Path | None = None,
    cors_allow_origins: list[str] | None = None,
) -> FastAPI:
    oct_state = oct_state or ViewerState(img_rows=1024, img_cols=512, img_interlaced=False)
    fa_state = fa_state or FAViewerState()
    oct_html_path = oct_html_path or Path(__file__).with_name("oct").joinpath("viewer.html")
    fa_html_path = fa_html_path or Path(__file__).with_name("fa").joinpath("viewer.html")
    oct_html = _rewrite_html_api_prefix(oct_html_path, "/api/oct")
    fa_html = _rewrite_html_api_prefix(fa_html_path, "/api/fa")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            oct_state.close()
            fa_state.close()

    app = FastAPI(
        title="OCT Converter Web Service",
        description="Unified service for the OCT and FA web viewer APIs.",
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "infra", "description": "Service discovery and health endpoints."},
            {"name": "oct", "description": "OCT loading, metadata, slices, projections, contours, and fundus images."},
            {"name": "fa", "description": "FA loading, metadata, and frame images."},
        ],
    )
    allow_origins = cors_allow_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    _install_error_handlers(app)

    @app.get("/", tags=["infra"], summary="Service index")
    async def service_index() -> dict:
        return {
            "service": "oct-converter-web-service",
            "docs": "/docs",
            "apiBase": "/api",
            "viewers": {"oct": "/oct", "fa": "/fa"},
            "apis": {"oct": "/api/oct", "fa": "/api/fa"},
        }

    @app.get("/healthz", tags=["infra"], summary="Health check")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api", tags=["infra"], summary="API index")
    async def api_index() -> dict:
        return {
            "oct": {
                "base": "/api/oct",
                "state": "/api/oct/state",
            },
            "fa": {
                "base": "/api/fa",
                "state": "/api/fa/state",
            },
        }

    @app.get("/oct", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/oct/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/oct/index.html", response_class=HTMLResponse, include_in_schema=False)
    async def serve_oct_index() -> HTMLResponse:
        return HTMLResponse(content=oct_html)

    @app.get("/api/oct/state", tags=["oct"], summary="Get OCT state")
    @app.get("/oct/api/state", include_in_schema=False)
    async def get_oct_state() -> dict:
        return oct_state.build_state_payload()

    @app.get("/api/oct/load", tags=["oct"], summary="Load OCT dataset from local path")
    @app.get("/oct/api/load", include_in_schema=False)
    async def load_oct_data(path: str, vendor: str = "auto") -> dict:
        filepath = path.strip()
        vendor_mode = vendor.strip() or "auto"
        if not filepath:
            raise ValueError("Missing path query parameter.")
        return oct_state.load(filepath, vendor_mode=vendor_mode)

    @app.get("/api/oct/pick-file", tags=["oct"], summary="Pick and load OCT file with native file dialog")
    @app.get("/oct/api/pick-file", include_in_schema=False)
    async def pick_oct_file(vendor: str = "auto") -> dict:
        return oct_state.pick_and_load(vendor_mode=vendor.strip() or "auto")

    if _has_python_multipart():
        @app.post("/api/oct/upload", tags=["oct"], summary="Upload and load OCT file")
        @app.post("/oct/api/upload", include_in_schema=False)
        async def upload_oct_file(
            vendor: str = "auto",
            file: UploadFile = File(...),
        ) -> dict:
            if not file.filename:
                raise ValueError("Missing uploaded filename.")
            return oct_state.load_uploaded_file(file.filename, file.file, vendor_mode=vendor.strip() or "auto")
    else:
        @app.post("/api/oct/upload", tags=["oct"], summary="Upload and load OCT file")
        @app.post("/oct/api/upload", include_in_schema=False)
        async def upload_oct_file_unavailable() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "error": 'The upload endpoint requires the optional dependency "python-multipart".',
                },
            )

    @app.get("/api/oct/volume/{volume_index}/slice/{slice_index}.png", tags=["oct"], summary="Get OCT slice PNG")
    @app.get("/oct/api/volume/{volume_index}/slice/{slice_index}.png", include_in_schema=False)
    async def get_oct_slice_png(
        volume_index: int,
        slice_index: int,
        contrast: int = 100,
        brightness: int = 0,
    ) -> Response:
        payload = oct_state.get_slice_png(
            volume_index=volume_index,
            slice_index=slice_index,
            contrast_percent=contrast,
            brightness_offset=brightness,
        )
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/oct/volume/{volume_index}/projection.png", tags=["oct"], summary="Get OCT projection PNG")
    @app.get("/oct/api/volume/{volume_index}/projection.png", include_in_schema=False)
    async def get_oct_projection_png(volume_index: int) -> Response:
        payload = oct_state.get_projection_png(volume_index)
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/oct/volume/{volume_index}/contours/{slice_index}.json", tags=["oct"], summary="Get OCT contours JSON")
    @app.get("/oct/api/volume/{volume_index}/contours/{slice_index}.json", include_in_schema=False)
    async def get_oct_contours(volume_index: int, slice_index: int) -> dict:
        return oct_state.get_contours(volume_index=volume_index, slice_index=slice_index)

    @app.get("/api/oct/fundus/{fundus_index}.png", tags=["oct"], summary="Get OCT fundus PNG")
    @app.get("/oct/api/fundus/{fundus_index}.png", include_in_schema=False)
    async def get_oct_fundus_png(fundus_index: int) -> Response:
        payload = oct_state.get_fundus_png(fundus_index)
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/oct/volume/{volume_index}/fundus-view.png", tags=["oct"], summary="Get OCT fundus overlay PNG")
    @app.get("/oct/api/volume/{volume_index}/fundus-view.png", include_in_schema=False)
    async def get_oct_fundus_view_png(volume_index: int) -> Response:
        payload = oct_state.get_volume_fundus_view_png(volume_index)
        return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/fa", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/fa/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/fa/index.html", response_class=HTMLResponse, include_in_schema=False)
    async def serve_fa_index() -> HTMLResponse:
        return HTMLResponse(content=fa_html)

    @app.get("/api/fa/state", tags=["fa"], summary="Get FA state")
    @app.get("/fa/api/state", include_in_schema=False)
    async def get_fa_state() -> dict:
        return fa_state.build_state_payload()

    @app.get("/api/fa/load", tags=["fa"], summary="Load FA dataset from local path")
    @app.get("/fa/api/load", include_in_schema=False)
    async def load_fa_data(path: str, vendor: str = "auto") -> dict:
        filepath = path.strip()
        vendor_mode = vendor.strip() or "auto"
        if not filepath:
            raise ValueError("Missing path query parameter.")
        return fa_state.load(filepath, vendor_mode=vendor_mode)

    @app.get("/api/fa/pick-file", tags=["fa"], summary="Pick and load FA file with native file dialog")
    @app.get("/fa/api/pick-file", include_in_schema=False)
    async def pick_fa_file(vendor: str = "auto") -> dict:
        return fa_state.pick_and_load_file(vendor_mode=vendor.strip() or "auto")

    @app.get("/api/fa/pick-directory", tags=["fa"], summary="Pick and load FA directory with native directory dialog")
    @app.get("/fa/api/pick-directory", include_in_schema=False)
    async def pick_fa_directory(vendor: str = "auto") -> dict:
        return fa_state.pick_and_load_directory(vendor_mode=vendor.strip() or "auto")

    @app.get("/api/fa/frame/{frame_index}.png", tags=["fa"], summary="Get FA frame PNG")
    @app.get("/fa/api/frame/{frame_index}.png", include_in_schema=False)
    async def get_fa_frame_png(
        frame_index: int,
        contrast: int = 100,
        brightness: int = 0,
    ) -> Response:
        payload = fa_state.get_frame_png(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified OCT/FA web service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument(
        "--cors-allow-origin",
        action="append",
        dest="cors_allow_origins",
        help="Allowed CORS origin. Repeat this option to allow multiple origins. Defaults to '*'.",
    )
    parser.add_argument("--oct-path", help="Optional OCT file or dataset path to load at startup.")
    parser.add_argument("--oct-vendor", default="auto", help="Startup OCT vendor mode.")
    parser.add_argument("--oct-img-rows", type=int, default=1024, help="Rows for Zeiss .img OCT files.")
    parser.add_argument("--oct-img-cols", type=int, default=512, help="Cols for Zeiss .img OCT files.")
    parser.add_argument(
        "--oct-img-interlaced",
        action="store_true",
        help="Apply de-interlacing for Zeiss .img OCT files.",
    )
    parser.add_argument("--fa-path", help="Optional FA file or dataset path to load at startup.")
    parser.add_argument("--fa-vendor", default="auto", help="Startup FA vendor mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cors_allow_origins = args.cors_allow_origins or ["*"]

    oct_state = ViewerState(
        img_rows=args.oct_img_rows,
        img_cols=args.oct_img_cols,
        img_interlaced=args.oct_img_interlaced,
    )
    fa_state = FAViewerState()

    if args.oct_path:
        oct_state.load(args.oct_path, vendor_mode=args.oct_vendor)
    if args.fa_path:
        fa_state.load(args.fa_path, vendor_mode=args.fa_vendor)

    url = f"http://{args.host}:{args.port}"
    print(f"OCT/FA web service running at {url}")
    print(f"OCT viewer: {url}/oct")
    print(f"FA viewer:  {url}/fa")
    print(f"API docs:   {url}/docs")
    try:
        uvicorn.run(
            create_service_app(
                oct_state=oct_state,
                fa_state=fa_state,
                cors_allow_origins=cors_allow_origins,
            ),
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
