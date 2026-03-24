"""Application entry point for the local OCT web viewer."""

from __future__ import annotations

import argparse
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

from .server import build_handler
from .state import ViewerState


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the web viewer."""

    parser = argparse.ArgumentParser(description="Local OCT web viewer.")
    parser.add_argument("path", nargs="?", help="Optional OCT file to load at startup.")
    parser.add_argument("--host", default="192.168.0.90", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument("--img-rows", type=int, default=1024, help="Rows for Zeiss .img files.")
    parser.add_argument("--img-cols", type=int, default=512, help="Cols for Zeiss .img files.")
    parser.add_argument(
        "--img-interlaced",
        action="store_true",
        help="Apply de-interlacing for Zeiss .img files.",
    )
    parser.add_argument(
        "--no-browser",
        default=True,
        help="Do not auto-open the browser.",
    )
    return parser.parse_args()


def main() -> None:
    """Starts the local OCT web viewer server."""

    args = parse_args()
    html_path = Path(__file__).with_name("oct_web_viewer.html")
    state = ViewerState(
        img_rows=args.img_rows,
        img_cols=args.img_cols,
        img_interlaced=args.img_interlaced,
    )

    if args.path:
        state.load(args.path)

    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, html_path))
    url = f"http://{args.host}:{args.port}"
    print(f"OCT web viewer running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()
