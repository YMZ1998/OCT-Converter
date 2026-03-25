from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_impl():
    module_dir = Path(__file__).resolve().parent / "fa"
    module_path = module_dir / "fa_qt_viewer.py"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("_fa_qt_viewer_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FA viewer modle from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _load_impl()
globals().update({name: getattr(_MODULE, name) for name in dir(_MODULE) if not name.startswith("__")})


if __name__ == "__main__":
    main()
