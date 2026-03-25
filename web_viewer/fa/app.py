"""Application entry point for the local FA web viewer."""

from __future__ import annotations

import argparse
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

from .server import build_handler
from .state import FAViewerState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local FA web viewer.")
    parser.add_argument("path", nargs="?", help="Optional FA dataset path to load at startup.")
    parser.add_argument("--vendor", default="auto", help="Startup vendor mode: auto/topcon/zeiss/hdb.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8766, help="Port to bind.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path = Path(__file__).with_name("viewer.html")
    state = FAViewerState()

    if args.path:
        state.load(args.path, vendor_mode=args.vendor)

    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, html_path))
    url = f"http://{args.host}:{args.port}"
    print(f"FA web viewer running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()
