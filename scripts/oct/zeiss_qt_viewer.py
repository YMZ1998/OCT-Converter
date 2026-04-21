from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pydicom

matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QShortcut,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from scripts.old.zeiss_dicom import ZEISSDicom

OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"

FUNDUS_SOURCE_PRIORITY = {
    "fundus_photo": 0,
    "enface": 1,
    "fundus": 1,
    "multiframe_localizer": 2,
}

FUNDUS_SOURCE_LABELS = {
    "fundus_photo": "眼底照片",
    "enface": "OCT 眼底投影",
    "fundus": "眼底图",
    "multiframe_localizer": "定位图",
    "projection-fallback": "体数据投影（回退）",
}

DISPLAY_MODE_LABELS = {
    "geometry-reference": "显示=定位参考图",
    "geometry-reference-fallback": "显示=定位参考图回退",
    "fundus-or-projection": "显示=眼底图/投影",
}


@dataclass
class FundusCandidate:
    image: np.ndarray
    laterality: str
    source_file: str
    source_kind: str
    width: int
    height: int
    score: float
    physical_width_mm: float | None
    physical_height_mm: float | None
    frame_of_reference_uid: str
    series_instance_uid: str


@dataclass
class VolumeViewModel:
    label: str
    source_file: str
    volume_id: str
    laterality: str
    slices: np.ndarray
    fundus: np.ndarray
    scan_segments: list
    overlay_bounds: tuple[float, float, float, float] | None
    overlay_outline: list[tuple[float, float]] | None
    overlay_spokes: list
    photo_scan_segments: list
    photo_overlay_bounds: tuple[float, float, float, float] | None
    photo_overlay_outline: list[tuple[float, float]] | None
    segmentation_surfaces: list[np.ndarray]
    fovea_point: tuple[float, float] | None
    photo_fovea_point: tuple[float, float] | None
    display_mode: str
    metadata_text: str
    fundus_source_file: str
    fundus_source_kind: str
    photo_fundus: np.ndarray | None
    photo_source_file: str | None
    photo_source_kind: str | None


@dataclass
class ExamViewModel:
    label: str
    exam_id: str
    patient_id: str
    exam_path: str
    volumes: list[VolumeViewModel]


@dataclass
class RawAnalysisData:
    source_file: str
    fovea_x_norm: float | None
    fovea_y_norm: float | None
    segmentation_surfaces: list[np.ndarray]


def parse_args():
    parser = argparse.ArgumentParser(
        description="蔡司 OCT DICOM 导出数据查看器。",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="可选：蔡司导出根目录、exam 目录、DCM 文件或 DICOMDIR 路径。",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).split("\x00", 1)[0]
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_pixel_spacing(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None

    if isinstance(value, (list, tuple)):
        items = [parse_float(item) for item in value[:2]]
        while len(items) < 2:
            items.append(None)
        return items[0], items[1]

    text = clean_text(value)
    if not text:
        return None, None

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 2:
        return None, None
    return parse_float(parts[0]), parse_float(parts[1])


def derive_physical_size_mm(rows: int | None, columns: int | None, pixel_spacing: tuple[float | None, float | None]) -> tuple[float | None, float | None]:
    if rows is None or columns is None:
        return None, None
    row_spacing, column_spacing = pixel_spacing
    physical_height = rows * row_spacing if row_spacing and row_spacing > 0 else None
    physical_width = columns * column_spacing if column_spacing and column_spacing > 0 else None
    return physical_width, physical_height


def clean_xml_text(value: Any) -> str:
    text = clean_text(value)
    start = text.find("<?xml")
    if start >= 0:
        text = text[start:]
    end = text.rfind(">")
    if end >= 0:
        text = text[: end + 1]
    return text.strip()


def extract_xml_float(root: ET.Element, path: str) -> float | None:
    element = root.find(path)
    if element is None or element.text is None:
        return None
    return parse_float(element.text)


def parse_raw_analysis_file(path: Path) -> RawAnalysisData | None:
    ds = safe_dcmread(path)
    analysis_xml = clean_xml_text(ds.get((0x0073, 0x1140), ""))
    fovea_x_norm = None
    fovea_y_norm = None
    if analysis_xml:
        try:
            xml_root = ET.fromstring(analysis_xml)
            fovea_x_norm = extract_xml_float(xml_root, ".//FoveaCenterData/FoveaPosition/X")
            fovea_y_norm = extract_xml_float(xml_root, ".//FoveaCenterData/FoveaPosition/Y")
        except ET.ParseError:
            fovea_x_norm = None
            fovea_y_norm = None

    segmentation_surfaces: list[np.ndarray] = []
    for tag in ((0x0073, 0x1150), (0x0073, 0x1155), (0x0073, 0x1160)):
        element = ds.get(tag)
        if element is None:
            continue
        try:
            surface = np.frombuffer(element.value, dtype=np.int16)
        except TypeError:
            continue
        if surface.size == 0 or np.all(surface == 0):
            continue
        segmentation_surfaces.append(surface.astype(np.float32))

    if not analysis_xml and not segmentation_surfaces:
        return None

    return RawAnalysisData(
        source_file=path.name,
        fovea_x_norm=fovea_x_norm,
        fovea_y_norm=fovea_y_norm,
        segmentation_surfaces=segmentation_surfaces,
    )


def load_exam_raw_analysis(exam_dir: Path) -> RawAnalysisData | None:
    raw_dcm_files = sorted(exam_dir.glob("*.DCM"))
    for path in raw_dcm_files:
        ds = safe_dcmread(path)
        sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
        if sop_class_uid != "1.2.840.10008.5.1.4.1.1.66":
            continue
        analysis = parse_raw_analysis_file(path)
        if analysis is not None:
            return analysis
    return None


def reshape_segmentation_surface(surface: np.ndarray, num_slices: int, slice_width: int) -> np.ndarray | None:
    if surface.size != num_slices * slice_width:
        return None
    reshaped = surface.reshape(num_slices, slice_width)
    reshaped = np.flip(reshaped, axis=0)
    reshaped = np.flip(reshaped, axis=1)
    return reshaped.copy()


def prepare_segmentation_surfaces(
    raw_analysis: RawAnalysisData | None,
    num_slices: int,
    slice_width: int,
) -> list[np.ndarray]:
    if raw_analysis is None:
        return []
    reshaped: list[np.ndarray] = []
    for surface in raw_analysis.segmentation_surfaces:
        reshaped_surface = reshape_segmentation_surface(surface, num_slices, slice_width)
        if reshaped_surface is not None:
            reshaped.append(reshaped_surface)
    reshaped.sort(key=lambda surface: float(np.nanmean(surface)))
    return reshaped


def classify_fundus_source_kind(
    ds: pydicom.dataset.FileDataset,
    classification: str,
    *,
    from_embedded_fundus: bool,
) -> str:
    sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
    number_of_frames = parse_int(getattr(ds, "NumberOfFrames", None))

    if sop_class_uid == OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID:
        return "fundus_photo"
    if classification == "multi_frame_image" or (number_of_frames is not None and number_of_frames > 1):
        return "multiframe_localizer"
    if classification == "single_frame_image" or from_embedded_fundus:
        return "enface"
    return "fundus"


def source_kind_priority(source_kind: str) -> int:
    return FUNDUS_SOURCE_PRIORITY.get(source_kind, 99)


def describe_fundus_source_kind(source_kind: str) -> str:
    return FUNDUS_SOURCE_LABELS.get(source_kind, source_kind)


def scope_rank(candidate: FundusCandidate, frame_of_reference_uid: str, series_instance_uid: str) -> int:
    if frame_of_reference_uid and candidate.frame_of_reference_uid == frame_of_reference_uid:
        return 0
    if series_instance_uid and candidate.series_instance_uid == series_instance_uid:
        return 1
    return 2


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) * (255.0 / (max_value - min_value))
    return np.clip(image, 0, 255).astype(np.uint8)


def to_display_image(image: np.ndarray) -> np.ndarray:
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_gray_image(image: np.ndarray) -> np.ndarray:
    image = to_display_image(image)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_gray_volume(volume_array: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_array)
    if volume.ndim == 2:
        return volume[np.newaxis, ...]
    if volume.ndim == 3:
        return volume
    if volume.ndim == 4 and volume.shape[-1] >= 1:
        return volume[..., 0]
    raise ValueError(f"Unsupported volume shape: {volume.shape}")


def safe_dcmread(path: Path) -> pydicom.dataset.FileDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydicom.dcmread(path, force=True)


def classify_dicom(ds: pydicom.dataset.FileDataset) -> str:
    has_pixel_data = "PixelData" in ds
    number_of_frames = None
    try:
        number_of_frames = int(clean_text(getattr(ds, "NumberOfFrames", "")) or 0)
    except ValueError:
        number_of_frames = None
    rows = getattr(ds, "Rows", None)
    columns = getattr(ds, "Columns", None)

    if not has_pixel_data:
        return "non_image_dicom"
    if number_of_frames is not None and number_of_frames >= 32:
        return "oct_volume"
    if number_of_frames is not None and number_of_frames > 1:
        return "multi_frame_image"
    if rows is not None and columns is not None:
        return "single_frame_image"
    return "unknown_image"


def resolve_exam_dirs(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.name.upper() == "DICOMDIR" or path.suffix.upper() == ".DCM":
            return [path.parent]
        raise ValueError("请选择蔡司导出目录、DICOMDIR 或 DCM 文件。")

    if (path / "DataFiles").exists():
        return sorted(child for child in (path / "DataFiles").iterdir() if child.is_dir())

    if path.name.lower() == "datafiles":
        return sorted(child for child in path.iterdir() if child.is_dir())

    if (path / "DICOMDIR").exists() or any(path.glob("*.DCM")):
        return [path]

    exam_dirs = [child for child in path.iterdir() if child.is_dir() and (child / "DICOMDIR").exists()]
    if exam_dirs:
        return sorted(exam_dirs)

    raise ValueError("未识别到蔡司导出结构。")


def is_bscan_like(volume: np.ndarray) -> bool:
    gray_volume = to_gray_volume(volume)
    height, width = gray_volume.shape[1:]
    ratio = max(width, height) / max(min(width, height), 1)
    return ratio >= 1.35


def make_projection_fundus(volume_slices: np.ndarray) -> np.ndarray:
    volume_array = to_gray_volume(volume_slices)
    projection = np.mean(volume_array, axis=1)
    return normalize_to_uint8(projection)


def build_reference_bounds(
    reference_shape: tuple[int, int],
    *,
    reference_source_kind: str,
    reference_physical_width_mm: float | None = None,
    reference_physical_height_mm: float | None = None,
    scan_width_mm: float | None = None,
    scan_height_mm: float | None = None,
    center_x_norm: float | None = None,
    center_y_norm: float | None = None,
) -> tuple[float, float, float, float]:
    height, width = reference_shape[:2]
    if reference_source_kind in {"enface", "multiframe_localizer"}:
        return 0.0, 0.0, float(max(width - 1, 1)), float(max(height - 1, 1))

    if (
        reference_physical_width_mm
        and reference_physical_height_mm
        and scan_width_mm
        and scan_height_mm
        and reference_physical_width_mm > 0
        and reference_physical_height_mm > 0
        and scan_width_mm > 0
        and scan_height_mm > 0
    ):
        box_w = min(width, width * (scan_width_mm / reference_physical_width_mm))
        box_h = min(height, height * (scan_height_mm / reference_physical_height_mm))
        if box_w < width * 0.1:
            box_w = width * 0.66
        if box_h < height * 0.1:
            box_h = height * 0.66
    else:
        box_w = width * 0.66
        box_h = height * 0.66

    center_x = float(width) * center_x_norm if center_x_norm is not None else width / 2.0
    center_y = float(height) * center_y_norm if center_y_norm is not None else height / 2.0
    x0 = max(0.0, min(width - box_w, center_x - box_w / 2.0))
    y0 = max(0.0, min(height - box_h, center_y - box_h / 2.0))
    return x0, y0, float(box_w), float(box_h)


def build_reference_point(
    reference_shape: tuple[int, int],
    *,
    center_x_norm: float | None,
    center_y_norm: float | None,
) -> tuple[float, float] | None:
    if center_x_norm is None or center_y_norm is None:
        return None
    height, width = reference_shape[:2]
    x = float(np.clip(center_x_norm, 0.0, 1.0)) * max(width - 1, 1)
    y = float(np.clip(center_y_norm, 0.0, 1.0)) * max(height - 1, 1)
    return x, y


def build_raster_segments_from_bounds(num_slices: int, bounds: tuple[float, float, float, float]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    x0, y0, box_w, box_h = bounds
    x1 = x0 + box_w
    y1 = y0 + box_h

    if num_slices <= 0:
        return []
    if num_slices == 1:
        y_values = [(y0 + y1) / 2.0]
    else:
        y_values = np.linspace(y0, y1, num_slices)
    return [((x0, float(y_val)), (x1, float(y_val))) for y_val in y_values]


def build_overlay_outline(bounds: tuple[float, float, float, float] | None) -> list[tuple[float, float]] | None:
    if bounds is None:
        return None
    x0, y0, width, height = bounds
    x1 = x0 + width
    y1 = y0 + height
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def build_raster_segments(
    num_slices: int,
    fundus_shape: tuple[int, int],
    *,
    fundus_source_kind: str = "fundus",
    fundus_physical_width_mm: float | None = None,
    fundus_physical_height_mm: float | None = None,
    scan_width_mm: float | None = None,
    scan_height_mm: float | None = None,
    center_x_norm: float | None = None,
    center_y_norm: float | None = None,
) -> tuple[list, tuple[float, float, float, float] | None]:
    if num_slices <= 0:
        return [], None

    bounds = build_reference_bounds(
        fundus_shape,
        reference_source_kind=fundus_source_kind,
        reference_physical_width_mm=fundus_physical_width_mm,
        reference_physical_height_mm=fundus_physical_height_mm,
        scan_width_mm=scan_width_mm,
        scan_height_mm=scan_height_mm,
        center_x_norm=center_x_norm,
        center_y_norm=center_y_norm,
    )
    return build_raster_segments_from_bounds(num_slices, bounds), bounds


def build_overlay_spokes(bounds: tuple[float, float, float, float] | None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if bounds is None:
        return []

    x0, y0, width, height = bounds
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    points = [
        (x0, y0),
        (x0 + width / 2.0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height / 2.0),
        (x0 + width, y0 + height),
        (x0 + width / 2.0, y0 + height),
        (x0, y0 + height),
        (x0, y0 + height / 2.0),
    ]
    return [((cx, cy), point) for point in points]


def estimate_bscan_guides(bscan_image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    gray = to_gray_image(bscan_image)
    height, width = gray.shape
    if height < 32 or width < 32:
        return None, None

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    gradient = np.diff(blurred.astype(np.float32), axis=0)

    upper_limit = max(8, int(height * 0.55))
    ilm = np.argmax(gradient[:upper_limit, :], axis=0).astype(np.float32)

    second = np.full(width, np.nan, dtype=np.float32)
    for x_pos in range(width):
        start = int(min(height - 2, ilm[x_pos] + max(5, int(height * 0.06))))
        end = int(min(height - 1, start + max(12, int(height * 0.28))))
        if end <= start:
            continue
        local_gradient = gradient[start:end, x_pos]
        if local_gradient.size == 0:
            continue
        second[x_pos] = start + int(np.argmin(local_gradient))

    kernel = np.ones(9, dtype=np.float32) / 9.0
    ilm = np.convolve(ilm, kernel, mode="same")
    valid_second = np.nan_to_num(second, nan=np.nanmean(second) if np.isfinite(np.nanmean(second)) else 0.0)
    valid_second = np.convolve(valid_second, kernel, mode="same")
    return ilm, valid_second


def score_fundus_candidate(candidate: FundusCandidate, laterality: str) -> float:
    score = candidate.score
    if candidate.laterality and laterality and candidate.laterality == laterality:
        score += 1_000_000.0
    score -= float(source_kind_priority(candidate.source_kind)) * 100_000.0
    return score


def geometry_match_penalty(
    candidate: FundusCandidate,
    *,
    scan_width_mm: float | None,
    scan_height_mm: float | None,
) -> float:
    if (
        candidate.physical_width_mm is None
        or candidate.physical_height_mm is None
        or scan_width_mm is None
        or scan_height_mm is None
        or candidate.physical_width_mm <= 0
        or candidate.physical_height_mm <= 0
        or scan_width_mm <= 0
        or scan_height_mm <= 0
    ):
        return float("inf")

    width_ratio = abs(math.log(max(scan_width_mm, 1e-6) / max(candidate.physical_width_mm, 1e-6)))
    height_ratio = abs(math.log(max(scan_height_mm, 1e-6) / max(candidate.physical_height_mm, 1e-6)))
    return width_ratio + height_ratio


def choose_best_fundus(
    candidates: list[FundusCandidate],
    laterality: str,
    *,
    frame_of_reference_uid: str,
    series_instance_uid: str,
    scan_width_mm: float | None,
    scan_height_mm: float | None,
) -> FundusCandidate | None:
    if not candidates:
        return None

    exact_laterality = [
        candidate
        for candidate in candidates
        if candidate.laterality and laterality and candidate.laterality == laterality
    ]
    pool = exact_laterality or candidates

    best_source_priority = min(source_kind_priority(candidate.source_kind) for candidate in pool)
    source_tier = [candidate for candidate in pool if source_kind_priority(candidate.source_kind) == best_source_priority]

    best_scope = min(scope_rank(candidate, frame_of_reference_uid, series_instance_uid) for candidate in source_tier)
    scoped = [
        candidate
        for candidate in source_tier
        if scope_rank(candidate, frame_of_reference_uid, series_instance_uid) == best_scope
    ]

    by_geometry = [
        (
            geometry_match_penalty(
                candidate,
                scan_width_mm=scan_width_mm,
                scan_height_mm=scan_height_mm,
            ),
            candidate,
        )
        for candidate in scoped
    ]
    finite_matches = [item for item in by_geometry if math.isfinite(item[0])]
    if finite_matches:
        best_penalty = min(item[0] for item in finite_matches)
        close_matches = [candidate for penalty, candidate in finite_matches if penalty <= best_penalty + 0.20]
        return max(close_matches, key=lambda candidate: score_fundus_candidate(candidate, laterality))

    return max(scoped, key=lambda candidate: score_fundus_candidate(candidate, laterality))


def geometry_reference_priority(candidate: FundusCandidate, display_candidate: FundusCandidate | None) -> tuple[int, int, float]:
    preferred = {
        "enface": 0,
        "multiframe_localizer": 1,
        "fundus_photo": 2,
        "fundus": 3,
    }
    same_as_display = int(
        display_candidate is not None
        and candidate.source_file == display_candidate.source_file
        and candidate.source_kind == display_candidate.source_kind
        and candidate.width == display_candidate.width
        and candidate.height == display_candidate.height
    )
    return preferred.get(candidate.source_kind, 9), same_as_display, -candidate.score


def choose_geometry_reference(
    candidates: list[FundusCandidate],
    laterality: str,
    *,
    display_candidate: FundusCandidate | None,
    frame_of_reference_uid: str,
    series_instance_uid: str,
) -> FundusCandidate | None:
    if not candidates:
        return None

    exact_laterality = [
        candidate
        for candidate in candidates
        if candidate.laterality and laterality and candidate.laterality == laterality
    ]
    pool = exact_laterality or candidates
    best_scope = min(scope_rank(candidate, frame_of_reference_uid, series_instance_uid) for candidate in pool)
    scoped = [
        candidate
        for candidate in pool
        if scope_rank(candidate, frame_of_reference_uid, series_instance_uid) == best_scope
    ]
    return min(scoped, key=lambda candidate: geometry_reference_priority(candidate, display_candidate))


def build_metadata_text(
    exam_dir: Path,
    patient_id: str,
    source_file: Path,
    volume: np.ndarray,
    fundus_candidate: FundusCandidate | None,
    geometry_candidate: FundusCandidate | None,
    display_candidate: FundusCandidate | None,
    display_mode: str,
    raw_analysis: RawAnalysisData | None,
    segmentation_surface_count: int,
    *,
    scan_width_mm: float | None,
    scan_height_mm: float | None,
) -> str:
    gray_volume = to_gray_volume(volume)
    metadata = {
        "exam_dir": str(exam_dir),
        "patient_id": patient_id,
        "source_file": str(source_file),
        "num_slices": int(gray_volume.shape[0]),
        "slice_shape": [int(gray_volume.shape[1]), int(gray_volume.shape[2])],
        "fundus_source": fundus_candidate.source_file if fundus_candidate else "projection-fallback",
        "fundus_kind": fundus_candidate.source_kind if fundus_candidate else "projection-fallback",
        "fundus_kind_label": describe_fundus_source_kind(fundus_candidate.source_kind) if fundus_candidate else describe_fundus_source_kind("projection-fallback"),
        "geometry_reference_source": geometry_candidate.source_file if geometry_candidate else None,
        "geometry_reference_kind": geometry_candidate.source_kind if geometry_candidate else None,
        "geometry_reference_kind_label": describe_fundus_source_kind(geometry_candidate.source_kind) if geometry_candidate else None,
        "display_source": display_candidate.source_file if display_candidate else "projection-fallback",
        "display_kind": display_candidate.source_kind if display_candidate else "projection-fallback",
        "display_kind_label": describe_fundus_source_kind(display_candidate.source_kind) if display_candidate else describe_fundus_source_kind("projection-fallback"),
        "display_mode": display_mode,
        "display_mode_label": DISPLAY_MODE_LABELS.get(display_mode, display_mode),
        "raw_analysis_source": raw_analysis.source_file if raw_analysis else None,
        "fovea_center_norm": (
            [raw_analysis.fovea_x_norm, raw_analysis.fovea_y_norm]
            if raw_analysis and raw_analysis.fovea_x_norm is not None and raw_analysis.fovea_y_norm is not None
            else None
        ),
        "segmentation_surface_count": segmentation_surface_count,
        "fundus_physical_width_mm": fundus_candidate.physical_width_mm if fundus_candidate else None,
        "fundus_physical_height_mm": fundus_candidate.physical_height_mm if fundus_candidate else None,
        "fundus_photo_available": fundus_candidate is not None,
        "scan_width_mm": scan_width_mm,
        "scan_height_mm": scan_height_mm,
        "bscan_guides": "优先显示 Raw Data 中解析出的真实分层；缺失时才使用启发式辅助线。",
    }
    return json.dumps(metadata, indent=2, ensure_ascii=False)


def load_exam(exam_dir: Path) -> ExamViewModel:
    dcm_files = sorted(exam_dir.glob("*.DCM"))
    if not dcm_files:
        raise RuntimeError(f"在 {exam_dir} 下没有找到 DCM 文件。")
    raw_analysis = load_exam_raw_analysis(exam_dir)

    dicomdir_path = exam_dir / "DICOMDIR"
    patient_id = ""
    if dicomdir_path.exists():
        ds = safe_dcmread(dicomdir_path)
        for record in getattr(ds, "DirectoryRecordSequence", []):
            patient_id = clean_text(getattr(record, "PatientID", ""))
            if patient_id:
                break

    fundus_candidates: list[FundusCandidate] = []
    pending_volumes: list[dict[str, Any]] = []

    for dcm_file in dcm_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = safe_dcmread(dcm_file)
            classification = classify_dicom(ds)
            if classification == "non_image_dicom":
                continue

            rows = getattr(ds, "Rows", None)
            columns = getattr(ds, "Columns", None)
            pixel_spacing = parse_pixel_spacing(getattr(ds, "PixelSpacing", None))
            physical_width_mm, physical_height_mm = derive_physical_size_mm(rows, columns, pixel_spacing)
            frame_of_reference_uid = clean_text(getattr(ds, "FrameOfReferenceUID", ""))
            series_instance_uid = clean_text(getattr(ds, "SeriesInstanceUID", ""))
            spacing_between_slices = parse_float(getattr(ds, "SpacingBetweenSlices", None))
            fundus_source_kind = classify_fundus_source_kind(
                ds,
                classification,
                from_embedded_fundus=True,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = ZEISSDicom(dcm_file)
            oct_volumes, fundus_images = reader.read_data()

        file_laterality = clean_text(getattr(ds, "Laterality", ""))

        for image in fundus_images:
            image_array = to_display_image(image.image)
            height, width = image_array.shape[:2]
            ratio = min(height, width) / max(height, width)
            score = float(height * width) * (0.5 + 0.5 * ratio)
            fundus_candidates.append(
                FundusCandidate(
                    image=image_array,
                    laterality=clean_text(getattr(image, "laterality", "")) or file_laterality,
                    source_file=dcm_file.name,
                    source_kind=fundus_source_kind,
                    width=width,
                    height=height,
                    score=score,
                    physical_width_mm=physical_width_mm,
                    physical_height_mm=physical_height_mm,
                    frame_of_reference_uid=frame_of_reference_uid,
                    series_instance_uid=series_instance_uid,
                )
            )

        for index, volume in enumerate(oct_volumes):
            gray_volume = to_gray_volume(volume.volume)
            height, width = gray_volume.shape[1:]

            if is_bscan_like(gray_volume):
                pending_volumes.append(
                    {
                        "source_file": dcm_file,
                        "volume_index": index,
                        "laterality": clean_text(getattr(volume, "laterality", "")) or file_laterality,
                        "volume": gray_volume,
                        "frame_of_reference_uid": frame_of_reference_uid,
                        "series_instance_uid": series_instance_uid,
                        "scan_width_mm": physical_width_mm,
                        "scan_height_mm": (
                            spacing_between_slices * max(gray_volume.shape[0] - 1, 1)
                            if spacing_between_slices and spacing_between_slices > 0 and gray_volume.shape[0] > 1
                            else physical_height_mm
                        ),
                    }
                )
            elif gray_volume.shape[0] > 0:
                localizer_image = normalize_to_uint8(gray_volume[0])
                local_h, local_w = localizer_image.shape[:2]
                ratio = min(local_h, local_w) / max(local_h, local_w)
                score = float(local_h * local_w) * (0.5 + 0.5 * ratio)
                fundus_candidates.append(
                    FundusCandidate(
                        image=localizer_image,
                        laterality=clean_text(getattr(volume, "laterality", "")) or file_laterality,
                        source_file=dcm_file.name,
                        source_kind=classify_fundus_source_kind(
                            ds,
                            classification,
                            from_embedded_fundus=False,
                        ),
                        width=local_w,
                        height=local_h,
                        score=score,
                        physical_width_mm=physical_width_mm,
                        physical_height_mm=physical_height_mm,
                        frame_of_reference_uid=frame_of_reference_uid,
                        series_instance_uid=series_instance_uid,
                    )
                )

    if not pending_volumes:
        raise RuntimeError(f"在 {exam_dir} 下没有找到可显示的 Zeiss B-scan 体数据。")

    models: list[VolumeViewModel] = []
    for item in pending_volumes:
        best_fundus = choose_best_fundus(
            fundus_candidates,
            item["laterality"],
            frame_of_reference_uid=item["frame_of_reference_uid"],
            series_instance_uid=item["series_instance_uid"],
            scan_width_mm=item["scan_width_mm"],
            scan_height_mm=item["scan_height_mm"],
        )
        displayed_fundus = best_fundus.image if best_fundus is not None else make_projection_fundus(item["volume"])
        geometry_candidate = choose_geometry_reference(
            fundus_candidates,
            item["laterality"],
            display_candidate=best_fundus,
            frame_of_reference_uid=item["frame_of_reference_uid"],
            series_instance_uid=item["series_instance_uid"],
        )
        geometry_image = geometry_candidate.image if geometry_candidate is not None else displayed_fundus
        display_source_candidate = best_fundus
        photo_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        photo_bounds: tuple[float, float, float, float] | None = None
        photo_outline: list[tuple[float, float]] | None = None
        photo_fovea_point: tuple[float, float] | None = None
        if best_fundus is not None:
            photo_segments, photo_bounds = build_raster_segments(
                item["volume"].shape[0],
                best_fundus.image.shape,
                fundus_source_kind=best_fundus.source_kind,
                fundus_physical_width_mm=best_fundus.physical_width_mm,
                fundus_physical_height_mm=best_fundus.physical_height_mm,
                scan_width_mm=item["scan_width_mm"],
                scan_height_mm=item["scan_height_mm"],
                center_x_norm=raw_analysis.fovea_x_norm if raw_analysis else None,
                center_y_norm=raw_analysis.fovea_y_norm if raw_analysis else None,
            )
            photo_outline = build_overlay_outline(photo_bounds)
            photo_fovea_point = build_reference_point(
                best_fundus.image.shape,
                center_x_norm=raw_analysis.fovea_x_norm if raw_analysis else None,
                center_y_norm=raw_analysis.fovea_y_norm if raw_analysis else None,
            )
        segments, bounds = build_raster_segments(
            item["volume"].shape[0],
            geometry_image.shape,
            fundus_source_kind=geometry_candidate.source_kind if geometry_candidate else (best_fundus.source_kind if best_fundus else "projection-fallback"),
            fundus_physical_width_mm=geometry_candidate.physical_width_mm if geometry_candidate else (best_fundus.physical_width_mm if best_fundus else None),
            fundus_physical_height_mm=geometry_candidate.physical_height_mm if geometry_candidate else (best_fundus.physical_height_mm if best_fundus else None),
            scan_width_mm=item["scan_width_mm"],
            scan_height_mm=item["scan_height_mm"],
            center_x_norm=raw_analysis.fovea_x_norm if raw_analysis else None,
            center_y_norm=raw_analysis.fovea_y_norm if raw_analysis else None,
        )
        outline = build_overlay_outline(bounds)
        spokes = build_overlay_spokes(bounds)
        fovea_point = build_reference_point(
            geometry_image.shape,
            center_x_norm=raw_analysis.fovea_x_norm if raw_analysis else None,
            center_y_norm=raw_analysis.fovea_y_norm if raw_analysis else None,
        )
        display_mode = "fundus-or-projection"
        fundus_image = geometry_image
        if best_fundus is not None and photo_segments:
            display_source_candidate = best_fundus
            segments = photo_segments
            bounds = photo_bounds
            outline = photo_outline
            fovea_point = photo_fovea_point
            spokes = build_overlay_spokes(bounds)
            fundus_image = displayed_fundus
            display_mode = "fundus-or-projection"
        elif geometry_candidate is not None:
            display_source_candidate = geometry_candidate
            display_mode = "geometry-reference"
            if best_fundus is not None and best_fundus.source_file != geometry_candidate.source_file:
                display_mode = "geometry-reference-fallback"

        segmentation_surfaces = prepare_segmentation_surfaces(
            raw_analysis,
            num_slices=item["volume"].shape[0],
            slice_width=item["volume"].shape[2],
        )
        label = item["source_file"].stem
        if item["laterality"]:
            label = f"{label} ({item['laterality']})"

        metadata_text = build_metadata_text(
            exam_dir=exam_dir,
            patient_id=patient_id,
            source_file=item["source_file"],
            volume=item["volume"],
            fundus_candidate=best_fundus,
            geometry_candidate=geometry_candidate,
            display_candidate=display_source_candidate,
            display_mode=display_mode,
            raw_analysis=raw_analysis,
            segmentation_surface_count=len(segmentation_surfaces),
            scan_width_mm=item["scan_width_mm"],
            scan_height_mm=item["scan_height_mm"],
        )

        models.append(
            VolumeViewModel(
                label=label,
                source_file=item["source_file"].name,
                volume_id=item["source_file"].stem,
                laterality=item["laterality"],
                slices=item["volume"],
                fundus=to_display_image(fundus_image),
                scan_segments=segments,
                overlay_bounds=bounds,
                overlay_outline=outline,
                overlay_spokes=spokes,
                photo_scan_segments=photo_segments,
                photo_overlay_bounds=photo_bounds,
                photo_overlay_outline=photo_outline,
                segmentation_surfaces=segmentation_surfaces,
                fovea_point=fovea_point,
                photo_fovea_point=photo_fovea_point,
                display_mode=display_mode,
                metadata_text=metadata_text,
                fundus_source_file=display_source_candidate.source_file if display_source_candidate else "projection-fallback",
                fundus_source_kind=display_source_candidate.source_kind if display_source_candidate else "projection-fallback",
                photo_fundus=to_display_image(best_fundus.image) if best_fundus is not None else None,
                photo_source_file=best_fundus.source_file if best_fundus is not None else None,
                photo_source_kind=best_fundus.source_kind if best_fundus is not None else None,
            )
        )

    return ExamViewModel(
        label=f"{exam_dir.name} ({patient_id or 'Unknown'})",
        exam_id=exam_dir.name,
        patient_id=patient_id,
        exam_path=str(exam_dir),
        volumes=models,
    )


def load_exams(path: str | Path) -> list[ExamViewModel]:
    exam_dirs = resolve_exam_dirs(path)
    return [load_exam(exam_dir) for exam_dir in exam_dirs]


class ViewerCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(10, 5), tight_layout=True, facecolor="#111827")
        super().__init__(self.figure)
        self.ax_fundus = self.figure.add_subplot(1, 2, 1)
        self.ax_bscan = self.figure.add_subplot(1, 2, 2)
        for axis in (self.ax_fundus, self.ax_bscan):
            axis.set_facecolor("#0B1220")


def apply_image_window(image: np.ndarray, contrast_percent: int, brightness_offset: int) -> np.ndarray:
    contrast = max(1, contrast_percent) / 100.0
    adjusted = image.astype(np.float32) * contrast + float(brightness_offset)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


class ZeissViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("蔡司 OCT 查看器")
        self.resize(1560, 920)
        self.setFont(QFont("Microsoft YaHei UI", 10))

        self.settings = QSettings("OpenAI", "ZeissOCTViewer")
        self.source_path: str | None = None
        self.exams: list[ExamViewModel] = []
        self.current_exam_index = 0
        self.current_volume_index = 0
        self.current_slice_index = 0

        self._apply_styles()
        self.canvas = ViewerCanvas()
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)
        self.setStatusBar(QStatusBar(self))

        self.open_button = QPushButton("打开目录")
        self.open_button.clicked.connect(self.open_path_dialog)

        self.snapshot_button = QPushButton("保存截图")
        self.snapshot_button.clicked.connect(self.save_snapshot)

        self.path_label = QLabel("未打开目录")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("PathLabel")

        self.exam_combo = QComboBox()
        self.exam_combo.currentIndexChanged.connect(self.on_exam_changed)

        self.volume_combo = QComboBox()
        self.volume_combo.currentIndexChanged.connect(self.on_volume_changed)

        self.fundus_view_combo = QComboBox()
        self.fundus_view_combo.currentIndexChanged.connect(self.redraw_views)

        self.prev_volume_button = QPushButton("上一组")
        self.prev_volume_button.clicked.connect(self.previous_volume)

        self.next_volume_button = QPushButton("下一组")
        self.next_volume_button.clicked.connect(self.next_volume)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(0)
        self.slice_spin.valueChanged.connect(self.on_slice_changed)

        self.prev_slice_button = QPushButton("上一层")
        self.prev_slice_button.clicked.connect(self.previous_slice)

        self.next_slice_button = QPushButton("下一层")
        self.next_slice_button.clicked.connect(self.next_slice)

        self.slice_info_label = QLabel("切片：-")
        self.slice_info_label.setObjectName("SliceInfo")

        self.summary_label = QLabel("等待导入蔡司 OCT 数据")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("SummaryLabel")

        self.show_guides_checkbox = QCheckBox("显示分层/辅助线")
        self.show_guides_checkbox.setChecked(True)
        self.show_guides_checkbox.toggled.connect(self.redraw_views)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setSingleStep(5)
        self.contrast_slider.valueChanged.connect(self.redraw_views)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-80, 80)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setSingleStep(2)
        self.brightness_slider.valueChanged.connect(self.redraw_views)

        self.reset_view_button = QPushButton("重置显示")
        self.reset_view_button.clicked.connect(self.reset_display_controls)

        self.exam_info_value = QLabel("-")
        self.volume_info_value = QLabel("-")
        self.fundus_info_value = QLabel("-")
        self.geometry_info_value = QLabel("-")

        self.legend_label = QLabel(
            "图例：绿色=当前 B-scan，红框=扫描范围，青色十字=黄斑中心，青/粉/绿线=分层结果或辅助线"
        )
        self.legend_label.setWordWrap(True)
        self.legend_label.setObjectName("LegendLabel")

        self.metadata_edit = QPlainTextEdit()
        self.metadata_edit.setReadOnly(True)
        self.metadata_edit.setFont(QFont("Microsoft YaHei UI", 10))

        self._build_toolbar()
        self._build_layout()
        self._install_shortcuts()
        self._set_empty_state()

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111827; color: #E5E7EB; }
            QToolBar { background: #0F172A; border: none; spacing: 6px; padding: 6px; }
            QPushButton {
                background: #1F2937;
                color: #E5E7EB;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QPushButton:hover { background: #273449; }
            QPushButton:pressed { background: #334155; }
            QComboBox, QSpinBox, QPlainTextEdit {
                background: #0F172A;
                color: #E5E7EB;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #334155;
                height: 6px;
                background: #0F172A;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38BDF8;
                border: 1px solid #7DD3FC;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #93C5FD;
            }
            QLabel#PathLabel, QLabel#SummaryLabel, QLabel#LegendLabel {
                background: #0F172A;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel#SliceInfo {
                background: #0369A1;
                color: white;
                border-radius: 12px;
                padding: 6px 12px;
                font-weight: 700;
            }
            QStatusBar {
                background: #0F172A;
                color: #CBD5E1;
            }
            """
        )

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self.previous_slice)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_slice)
        QShortcut(QKeySequence(Qt.Key_A), self, self.previous_slice)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_slice)
        QShortcut(QKeySequence(Qt.Key_PageUp), self, self.previous_volume)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, self.next_volume)
        QShortcut(QKeySequence(Qt.Key_Home), self, lambda: self.set_slice_index(0))
        QShortcut(QKeySequence(Qt.Key_End), self, self.jump_to_last_slice)
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_path_dialog)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_snapshot)

    def _build_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开目录", self)
        open_action.triggered.connect(self.open_path_dialog)
        toolbar.addAction(open_action)

        snapshot_action = QAction("保存截图", self)
        snapshot_action.triggered.connect(self.save_snapshot)
        toolbar.addAction(snapshot_action)

        toolbar.addSeparator()

        prev_volume_action = QAction("上一组", self)
        prev_volume_action.triggered.connect(self.previous_volume)
        toolbar.addAction(prev_volume_action)

        next_volume_action = QAction("下一组", self)
        next_volume_action.triggered.connect(self.next_volume)
        toolbar.addAction(next_volume_action)

        toolbar.addSeparator()

        prev_slice_action = QAction("上一层", self)
        prev_slice_action.triggered.connect(self.previous_slice)
        toolbar.addAction(prev_slice_action)

        next_slice_action = QAction("下一层", self)
        next_slice_action.triggered.connect(self.next_slice)
        toolbar.addAction(next_slice_action)

    def _make_info_row(self, title: str, value_widget: QLabel) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #93C5FD; font-weight: 600;")
        value_widget.setWordWrap(True)
        layout.addWidget(title_label, 0)
        layout.addWidget(value_widget, 1)
        return row

    def _build_layout(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)

        source_group = QGroupBox("数据源")
        source_layout = QVBoxLayout(source_group)
        source_layout.addWidget(self.open_button)
        source_layout.addWidget(self.snapshot_button)
        source_layout.addWidget(self.path_label)

        exam_group = QGroupBox("检查 / 体数据")
        exam_layout = QVBoxLayout(exam_group)
        exam_layout.addWidget(QLabel("检查"))
        exam_layout.addWidget(self.exam_combo)
        exam_layout.addWidget(QLabel("体数据"))
        exam_layout.addWidget(self.volume_combo)
        volume_button_row = QWidget()
        volume_button_layout = QHBoxLayout(volume_button_row)
        volume_button_layout.setContentsMargins(0, 0, 0, 0)
        volume_button_layout.addWidget(self.prev_volume_button)
        volume_button_layout.addWidget(self.next_volume_button)
        exam_layout.addWidget(volume_button_row)

        navigation_group = QGroupBox("切片导航")
        navigation_layout = QGridLayout(navigation_group)
        navigation_layout.addWidget(self.slice_info_label, 0, 0, 1, 2)
        navigation_layout.addWidget(self.prev_slice_button, 1, 0)
        navigation_layout.addWidget(self.next_slice_button, 1, 1)
        navigation_layout.addWidget(self.slice_slider, 2, 0, 1, 2)
        navigation_layout.addWidget(self.slice_spin, 3, 0, 1, 2)

        display_group = QGroupBox("显示设置")
        display_layout = QGridLayout(display_group)
        display_layout.addWidget(QLabel("左图显示"), 0, 0)
        display_layout.addWidget(self.fundus_view_combo, 0, 1)
        display_layout.addWidget(self.show_guides_checkbox, 1, 0, 1, 2)
        display_layout.addWidget(QLabel("B-scan 对比度"), 2, 0)
        display_layout.addWidget(self.contrast_slider, 2, 1)
        display_layout.addWidget(QLabel("B-scan 亮度"), 3, 0)
        display_layout.addWidget(self.brightness_slider, 3, 1)
        display_layout.addWidget(self.reset_view_button, 4, 0, 1, 2)

        info_group = QGroupBox("对应信息")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self._make_info_row("检查", self.exam_info_value))
        info_layout.addWidget(self._make_info_row("体数据", self.volume_info_value))
        info_layout.addWidget(self._make_info_row("眼底图", self.fundus_info_value))
        info_layout.addWidget(self._make_info_row("配准范围", self.geometry_info_value))
        info_layout.addWidget(self.summary_label)
        info_layout.addWidget(self.legend_label)

        control_layout.addWidget(source_group)
        control_layout.addWidget(exam_group)
        control_layout.addWidget(navigation_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(info_group)
        control_layout.addStretch(1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas, stretch=4)
        metadata_title = QLabel("元数据")
        metadata_title.setStyleSheet("font-weight: 700; color: #93C5FD; padding: 4px 0;")
        right_layout.addWidget(metadata_title)
        right_layout.addWidget(self.metadata_edit, stretch=2)

        splitter = QSplitter()
        splitter.addWidget(control_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1180])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _set_empty_state(self):
        self.exam_combo.blockSignals(True)
        self.exam_combo.clear()
        self.exam_combo.blockSignals(False)

        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        self.volume_combo.blockSignals(False)

        self.fundus_view_combo.blockSignals(True)
        self.fundus_view_combo.clear()
        self.fundus_view_combo.blockSignals(False)

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(0)
        self.slice_spin.setValue(0)
        self.slice_spin.setSuffix("")
        self.slice_spin.blockSignals(False)

        self.exam_info_value.setText("-")
        self.volume_info_value.setText("-")
        self.fundus_info_value.setText("-")
        self.geometry_info_value.setText("-")
        self.metadata_edit.setPlainText("")
        self.summary_label.setText("等待导入蔡司 OCT 数据")
        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()
        self.canvas.ax_fundus.set_title("眼底图 / 蔡司", color="#E5E7EB")
        self.canvas.ax_bscan.set_title("B-scan", color="#E5E7EB")
        self.slice_info_label.setText("切片：-")
        self.canvas.draw_idle()
        self.statusBar().showMessage("未加载数据")

    def get_last_open_dir(self) -> str:
        last_dir = self.settings.value("last_dir", "", type=str)
        if last_dir and Path(last_dir).exists():
            return last_dir
        return ""

    def remember_path(self, path: str) -> None:
        path = str(Path(path).resolve())
        remember_dir = path if Path(path).is_dir() else str(Path(path).parent)
        self.settings.setValue("last_dir", remember_dir)

    def open_path_dialog(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择蔡司导出目录 / 检查目录",
            self.get_last_open_dir(),
        )
        if directory:
            self.load_path(directory)

    def load_path(self, path: str):
        try:
            exams = load_exams(path)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.source_path = str(path)
        self.exams = exams
        self.current_exam_index = 0
        self.current_volume_index = 0
        self.current_slice_index = 0
        self.remember_path(path)

        self.path_label.setText(self.source_path)
        self.exam_combo.blockSignals(True)
        self.exam_combo.clear()
        for exam in exams:
            self.exam_combo.addItem(exam.label)
        self.exam_combo.blockSignals(False)
        self.exam_combo.setCurrentIndex(0)
        self.statusBar().showMessage(f"已加载 {len(exams)} 个 exam", 5000)
        self.refresh_exam()

    def current_exam(self) -> ExamViewModel | None:
        if not self.exams:
            return None
        return self.exams[self.current_exam_index]

    def current_model(self) -> VolumeViewModel | None:
        exam = self.current_exam()
        if exam is None or not exam.volumes:
            return None
        return exam.volumes[self.current_volume_index]

    def refresh_exam(self):
        exam = self.current_exam()
        if exam is None:
            self._set_empty_state()
            return

        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for model in exam.volumes:
            self.volume_combo.addItem(model.label)
        self.volume_combo.blockSignals(False)

        self.current_volume_index = 0
        self.current_slice_index = 0
        self.volume_combo.setCurrentIndex(0)
        self.exam_info_value.setText(f"{exam.exam_id} | 患者 ID={exam.patient_id or '-'}")
        self.refresh_volume()

    def refresh_volume(self):
        model = self.current_model()
        if model is None:
            self._set_empty_state()
            return

        max_index = max(0, len(model.slices) - 1)
        self.current_slice_index = min(self.current_slice_index, max_index)

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(max_index)
        self.slice_slider.setValue(self.current_slice_index)
        self.slice_slider.setTickInterval(max(1, (max_index + 1) // 10))
        self.slice_slider.blockSignals(False)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(max_index)
        self.slice_spin.setValue(self.current_slice_index)
        self.slice_spin.setSuffix(f" / {max_index + 1}")
        self.slice_spin.blockSignals(False)

        previous_view = self.fundus_view_combo.currentData()
        self.fundus_view_combo.blockSignals(True)
        self.fundus_view_combo.clear()
        if model.photo_fundus is not None:
            self.fundus_view_combo.addItem("真实眼底照片（推荐）", "photo")
        self.fundus_view_combo.addItem("定位参考图", "overlay")
        target_index = 0
        if previous_view == "overlay":
            target_index = self.fundus_view_combo.count() - 1
        self.fundus_view_combo.setCurrentIndex(target_index)
        self.fundus_view_combo.blockSignals(False)

        exam = self.current_exam()
        self.metadata_edit.setPlainText(model.metadata_text)
        self.summary_label.setText(
            f"检查 {exam.exam_id} | {model.source_file} | 共 {len(model.slices)} 层 | 眼别：{model.laterality or '-'} | 点击左图跳转切片，滚轮翻页"
        )
        self.volume_info_value.setText(
            f"{model.source_file} | 序列={model.volume_id} | 尺寸={tuple(model.slices.shape)}"
        )
        fundus_info_text = (
            f"当前显示={model.fundus_source_file} | {describe_fundus_source_kind(model.fundus_source_kind)} | 尺寸={tuple(model.fundus.shape)}"
        )
        if model.photo_fundus is not None and model.photo_source_file:
            fundus_info_text += (
                f"\n真实眼底={model.photo_source_file} | "
                f"{describe_fundus_source_kind(model.photo_source_kind or 'fundus_photo')} | "
                f"尺寸={tuple(model.photo_fundus.shape)}"
            )
        self.fundus_info_value.setText(fundus_info_text)
        if model.overlay_bounds is not None:
            _, _, width, height = model.overlay_bounds
            segmentation_label = f"分层={len(model.segmentation_surfaces)} 条" if model.segmentation_surfaces else "分层=未解析"
            fovea_label = "黄斑中心=已解析" if model.fovea_point is not None else "黄斑中心=未解析"
            self.geometry_info_value.setText(
                f"覆盖框：{width:.1f}px × {height:.1f}px | {segmentation_label} | {fovea_label} | "
                f"{DISPLAY_MODE_LABELS.get(model.display_mode, model.display_mode)}"
            )
        else:
            self.geometry_info_value.setText("-")
        self.redraw_views()

    def redraw_views(self):
        model = self.current_model()
        if model is None:
            return

        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()

        selected_fundus_view = self.fundus_view_combo.currentData() or "overlay"
        showing_photo = selected_fundus_view == "photo" and model.photo_fundus is not None
        fundus = model.photo_fundus if showing_photo and model.photo_fundus is not None else model.fundus
        fundus_source_file = model.photo_source_file if showing_photo and model.photo_source_file else model.fundus_source_file
        fundus_source_kind = model.photo_source_kind if showing_photo and model.photo_source_kind else model.fundus_source_kind
        active_outline = model.photo_overlay_outline if showing_photo else model.overlay_outline
        active_bounds = model.photo_overlay_bounds if showing_photo else model.overlay_bounds
        active_fovea_point = model.photo_fovea_point if showing_photo else model.fovea_point
        active_scan_segments = model.photo_scan_segments if showing_photo else model.scan_segments
        if fundus.ndim == 2:
            self.canvas.ax_fundus.imshow(fundus, cmap="gray", origin="upper")
        else:
            self.canvas.ax_fundus.imshow(fundus, origin="upper")
        self.canvas.ax_fundus.axis("off")
        self.canvas.ax_fundus.set_title(
            f"眼底图 / {describe_fundus_source_kind(fundus_source_kind)} / {Path(fundus_source_file).stem}",
            color="#E5E7EB",
            fontsize=13,
        )

        if active_outline:
            outline = np.asarray(active_outline + [active_outline[0]], dtype=np.float32)
            self.canvas.ax_fundus.plot(
                outline[:, 0],
                outline[:, 1],
                color="red",
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
            )
        elif active_bounds is not None:
            x0, y0, width, height = active_bounds
            self.canvas.ax_fundus.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.85,
                )
            )

        if active_fovea_point is not None:
            self.canvas.ax_fundus.scatter(
                [active_fovea_point[0]],
                [active_fovea_point[1]],
                s=38,
                color="#00FFFF",
                marker="+",
                linewidths=1.4,
                zorder=6,
            )

        current_center_x = None
        current_center_y = None
        if 0 <= self.current_slice_index < len(active_scan_segments):
            start_x, start_y = active_scan_segments[self.current_slice_index][0]
            end_x, end_y = active_scan_segments[self.current_slice_index][1]
            self.canvas.ax_fundus.plot(
                [start_x, end_x],
                [start_y, end_y],
                color="#66FF66",
                alpha=0.95,
                linewidth=2.2,
            )
            current_center_x = (start_x + end_x) / 2.0
            current_center_y = (start_y + end_y) / 2.0

        if current_center_x is not None and current_center_y is not None:
            self.canvas.ax_fundus.scatter(
                [current_center_x],
                [current_center_y],
                s=28,
                color="#66FF66",
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )

        bscan = normalize_to_uint8(model.slices[self.current_slice_index])
        bscan = apply_image_window(
            bscan,
            contrast_percent=self.contrast_slider.value(),
            brightness_offset=self.brightness_slider.value(),
        )
        self.canvas.ax_bscan.imshow(bscan, cmap="gray", aspect="auto", origin="upper")
        self.canvas.ax_bscan.axis("off")
        self.canvas.ax_bscan.set_title(
            f"B-scan {self.current_slice_index + 1}/{len(model.slices)}",
            color="#E5E7EB",
            fontsize=13,
        )

        if self.show_guides_checkbox.isChecked() and model.segmentation_surfaces:
            surface_colors = ["#00FFFF", "#FF66CC", "#66FF66"]
            x_coords = np.arange(bscan.shape[1])
            for index, surface in enumerate(model.segmentation_surfaces):
                if self.current_slice_index >= surface.shape[0]:
                    continue
                y_coords = surface[self.current_slice_index]
                if y_coords.shape[0] != bscan.shape[1]:
                    continue
                self.canvas.ax_bscan.plot(
                    x_coords,
                    y_coords,
                    color=surface_colors[index % len(surface_colors)],
                    alpha=0.95,
                    linewidth=1.0,
                )

        self.slice_info_label.setText(f"切片：{self.current_slice_index + 1}/{len(model.slices)}")
        selected_fundus_view = self.fundus_view_combo.currentData() or "overlay"
        status_source_kind = model.photo_source_kind if selected_fundus_view == "photo" and model.photo_source_kind else model.fundus_source_kind
        self.statusBar().showMessage(
            f"{model.source_file} | 切片 {self.current_slice_index + 1}/{len(model.slices)} | "
            f"显示={describe_fundus_source_kind(status_source_kind)} | "
            f"{DISPLAY_MODE_LABELS.get(model.display_mode, model.display_mode)}"
        )
        self.canvas.draw_idle()

    def set_slice_index(self, index: int):
        model = self.current_model()
        if model is None:
            return

        index = max(0, min(index, len(model.slices) - 1))
        self.current_slice_index = index

        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(index)
        self.slice_spin.blockSignals(False)

        self.slice_info_label.setText(f"切片：{index + 1}/{len(model.slices)}")
        self.redraw_views()

    def previous_slice(self):
        self.set_slice_index(self.current_slice_index - 1)

    def next_slice(self):
        self.set_slice_index(self.current_slice_index + 1)

    def previous_volume(self):
        exam = self.current_exam()
        if exam is None or not exam.volumes:
            return
        target = max(0, self.current_volume_index - 1)
        if target != self.current_volume_index:
            self.volume_combo.setCurrentIndex(target)

    def next_volume(self):
        exam = self.current_exam()
        if exam is None or not exam.volumes:
            return
        target = min(len(exam.volumes) - 1, self.current_volume_index + 1)
        if target != self.current_volume_index:
            self.volume_combo.setCurrentIndex(target)

    def jump_to_last_slice(self):
        model = self.current_model()
        if model is None:
            return
        self.set_slice_index(len(model.slices) - 1)

    def reset_display_controls(self):
        self.fundus_view_combo.setCurrentIndex(0)
        self.show_guides_checkbox.setChecked(True)
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)
        self.redraw_views()

    def save_snapshot(self):
        model = self.current_model()
        default_name = "zeiss_viewer_snapshot.png"
        if model is not None:
            default_name = f"{Path(model.source_file).stem}_slice_{self.current_slice_index + 1:03d}.png"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前界面截图",
            str(Path(self.get_last_open_dir() or ".") / default_name),
            "PNG Image (*.png)",
        )
        if not filename:
            return
        self.canvas.figure.savefig(filename, dpi=150, bbox_inches="tight")
        self.statusBar().showMessage(f"截图已保存到 {filename}", 5000)

    def on_exam_changed(self, index: int):
        if index < 0 or index >= len(self.exams):
            return
        self.current_exam_index = index
        self.current_volume_index = 0
        self.current_slice_index = 0
        self.refresh_exam()

    def on_volume_changed(self, index: int):
        exam = self.current_exam()
        if exam is None or index < 0 or index >= len(exam.volumes):
            return
        self.current_volume_index = index
        self.current_slice_index = 0
        self.refresh_volume()

    def on_slice_changed(self, value: int):
        self.set_slice_index(int(value))

    def on_canvas_click(self, event):
        model = self.current_model()
        if model is None or event.inaxes != self.canvas.ax_fundus:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not model.scan_segments:
            return

        target_x = float(event.xdata)
        target_y = float(event.ydata)
        nearest_index = 0
        nearest_distance = float("inf")
        for index, segment in enumerate(model.scan_segments):
            (start_x, start_y), (end_x, end_y) = segment
            center_x = (start_x + end_x) / 2.0
            center_y = (start_y + end_y) / 2.0
            distance = (target_x - center_x) ** 2 + (target_y - center_y) ** 2
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        self.set_slice_index(nearest_index)

    def on_canvas_scroll(self, event):
        if event.inaxes != self.canvas.ax_bscan:
            return
        step = 1 if event.button == "up" else -1
        self.set_slice_index(self.current_slice_index + step)


def main():
    args = parse_args()
    application = QApplication(sys.argv)
    window = ZeissViewerWindow()
    window.show()
    if args.path:
        window.load_path(args.path)
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
