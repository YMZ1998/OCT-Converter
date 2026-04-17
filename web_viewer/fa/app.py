"""Application entry point for the local FA web viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .server import create_app
from .state import FAViewerState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local FA web viewer.")
    parser.add_argument("path", nargs="?", help="Optional FA dataset path to load at startup.")
    parser.add_argument("--vendor", default="auto", help="Startup vendor mode: auto/topcon/zeiss/hdb/cfp.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8766, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path = Path(__file__).with_name("viewer.html")
    state = FAViewerState()

    if args.path:
        state.load(args.path, vendor_mode=args.vendor)

    url = f"http://{args.host}:{args.port}"
    print(f"FA web viewer running at {url}")
    try:
        uvicorn.run(
            create_app(state, html_path),
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
