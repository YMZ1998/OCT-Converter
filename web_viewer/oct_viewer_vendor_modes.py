"""Vendor-mode definitions for the OCT web viewer."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = (
    ".fds",
    ".fda",
    ".e2e",
    ".img",
    ".oct",
    ".OCT",
    ".dcm",
    ".dicom",
)
NORMALIZED_SUPPORTED_EXTENSIONS = frozenset(
    extension.lower() for extension in SUPPORTED_EXTENSIONS
)

FILE_DIALOG_TYPES = [
    (
        "Supported OCT files",
        "*.fds *.FDS *.fda *.FDA *.e2e *.E2E *.img *.IMG *.oct *.OCT *.dcm *.DCM *.dicom *.DICOM",
    ),
    ("Topcon FDS", "*.fds *.FDS"),
    ("Topcon FDA", "*.fda *.FDA"),
    ("Heidelberg E2E", "*.e2e *.E2E"),
    ("Zeiss IMG", "*.img *.IMG"),
    ("Optovue OCT", "*.oct"),
    ("Bioptigen OCT", "*.OCT"),
    ("DICOM", "*.dcm *.DCM *.dicom *.DICOM"),
    ("All files", "*.*"),
]

VENDOR_MODE_LABELS = {
    "auto": "自动识别",
    "heidelberg": "海德堡 E2E",
    "topcon": "拓普康 FDA/FDS",
    "zeiss": "蔡司 IMG",
    "optovue": "Optovue OCT",
    "bioptigen": "Bioptigen OCT",
    "dicom": "通用 DICOM",
    "shiwei": "视微",
    "tupai": "Tupai DICOM",
}

VENDOR_MODE_ALIASES = {
    "standard": "auto",
    "generic": "auto",
}

VALID_VENDOR_MODES = frozenset(VENDOR_MODE_LABELS) | frozenset(VENDOR_MODE_ALIASES)

SHIWEI_PATH_HINT = (
    "当前已选择“视微”，请输入视微数据目录、"
    "目录中的 `.dcm` / `.dicom` 文件，或可唯一定位该数据集的上级目录：{name}"
)
SHIWEI_UPLOAD_HINT = (
    "视微模式不支持单文件上传，请直接输入视微数据目录、"
    "可唯一定位该数据集的上级目录，或输入目录中的任一配套 DICOM 文件路径后点击“加载文件”。"
)
TUPAI_PATH_HINT = (
    "当前已选择“Tupai DICOM”，请输入包含 `OCT.dcm` 和 `Fundus.dcm` 的目录、"
    "其中任一文件路径，或可唯一定位该数据集的上级目录：{name}"
)
TUPAI_UPLOAD_HINT = (
    "Tupai 模式不支持单文件上传，请直接输入数据目录、"
    "可唯一定位该数据集的上级目录，或输入 `OCT.dcm` / `Fundus.dcm` 的路径后点击“加载文件”。"
)

VENDOR_MODE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "auto": tuple(),
    "heidelberg": (".e2e",),
    "topcon": (".fda", ".fds"),
    "zeiss": (".img",),
    "optovue": (".oct",),
    "bioptigen": (".OCT",),
    "dicom": (".dcm", ".dicom"),
    "shiwei": (".dcm", ".dicom"),
    "tupai": (".dcm", ".dicom"),
}

VENDOR_MODE_FILE_DIALOG_TYPES: dict[str, list[tuple[str, str]]] = {
    "auto": FILE_DIALOG_TYPES,
    "heidelberg": [("Heidelberg E2E", "*.e2e *.E2E"), ("All files", "*.*")],
    "topcon": [("Topcon FDA/FDS", "*.fda *.FDA *.fds *.FDS"), ("All files", "*.*")],
    "zeiss": [("Zeiss IMG", "*.img *.IMG"), ("All files", "*.*")],
    "optovue": [("Optovue OCT", "*.oct"), ("All files", "*.*")],
    "bioptigen": [("Bioptigen OCT", "*.OCT"), ("All files", "*.*")],
    "dicom": [("DICOM", "*.dcm *.DCM *.dicom *.DICOM"), ("All files", "*.*")],
    "shiwei": [("DICOM", "*.dcm *.DCM *.dicom *.DICOM"), ("All files", "*.*")],
    "tupai": [("DICOM", "*.dcm *.DCM *.dicom *.DICOM"), ("All files", "*.*")],
}


def normalize_vendor_mode(vendor_mode: str | None) -> str:
    """Normalizes the viewer vendor mode string."""

    normalized = (vendor_mode or "auto").strip().lower()
    if normalized not in VALID_VENDOR_MODES:
        return "auto"
    return VENDOR_MODE_ALIASES.get(normalized, normalized)


def is_supported_suffix_for_vendor(path: Path, vendor_mode: str) -> bool:
    """Returns whether the path suffix is valid for the vendor mode."""

    normalized_mode = normalize_vendor_mode(vendor_mode)
    suffix = path.suffix
    suffix_lower = suffix.lower()

    if normalized_mode == "auto":
        return suffix_lower in NORMALIZED_SUPPORTED_EXTENSIONS
    if normalized_mode == "bioptigen":
        return suffix == ".OCT"
    if normalized_mode == "optovue":
        return suffix == ".oct"

    supported = VENDOR_MODE_EXTENSIONS.get(normalized_mode, tuple())
    return suffix_lower in {extension.lower() for extension in supported}


def build_vendor_validation_error(path: Path, vendor_mode: str) -> str:
    """Builds a human-readable validation error for the vendor mode."""

    label = VENDOR_MODE_LABELS.get(vendor_mode, vendor_mode)
    name = path.name or str(path)
    if vendor_mode == "heidelberg":
        return f"当前已选择“{label}”，请加载 `.e2e` 文件：{name}"
    if vendor_mode == "topcon":
        return f"当前已选择“{label}”，请加载 `.fda` 或 `.fds` 文件：{name}"
    if vendor_mode == "zeiss":
        return f"当前已选择“{label}”，请加载 `.img` 文件：{name}"
    if vendor_mode == "optovue":
        return f"当前已选择“{label}”，请加载小写扩展名 `.oct` 文件：{name}"
    if vendor_mode == "bioptigen":
        return f"当前已选择“{label}”，请加载大写扩展名 `.OCT` 文件：{name}"
    if vendor_mode == "dicom":
        return f"当前已选择“{label}”，请加载 `.dcm` 或 `.dicom` 文件：{name}"
    if vendor_mode == "shiwei":
        return SHIWEI_PATH_HINT.format(name=name)
    if vendor_mode == "tupai":
        return TUPAI_PATH_HINT.format(name=name)
    return f"当前加载模式与文件类型不匹配：{name}"
