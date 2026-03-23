from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")

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
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


@dataclass
class OverlayState:
    scan_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    overlay_mode: str
    localizer_mode: str
    projection_mode: str
    warning: str
    bounds: tuple[float, float, float, float] | None


@dataclass
class VolumeViewModel:
    label: str
    volume_id: str
    laterality: str
    source_kind: str
    filepath: str
    slices: list[np.ndarray]
    contours: dict[str, Any] | None
    fundus: np.ndarray
    fundus_shape: tuple[int, int]
    fundus_source_label: str
    fundus_match_mode: str
    overlay: OverlayState
    pixel_spacing: list[float] | tuple[float, ...] | None
    slice_shape: tuple[int, ...]
    metadata_text: str
    status_lines: list[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Professional Heidelberg-first OCT E2E/FDA Qt viewer.")
    parser.add_argument("filepath", nargs="?", help="Optional E2E/FDA file to open at startup.")
    return parser.parse_args()


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
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def apply_image_window(image: np.ndarray, contrast_percent: int, brightness_offset: int) -> np.ndarray:
    contrast = max(1, contrast_percent) / 100.0
    adjusted = image.astype(np.float32) * contrast + float(brightness_offset)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def safe_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def normalize_laterality_code(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"R", "OD"}:
        return "R"
    if normalized in {"L", "OS"}:
        return "L"
    return normalized


def make_projection_fundus(volume_slices: list[np.ndarray]) -> np.ndarray:
    volume_array = np.asarray(volume_slices)
    if volume_array.ndim != 3:
        return np.zeros((512, 512), dtype=np.uint8)
    projection = np.mean(volume_array, axis=1)
    return normalize_to_uint8(projection)


def partition_bscan_metadata(bscan_data: list[dict[str, Any]], slice_counts: list[int]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for count in slice_counts:
        groups.append(bscan_data[cursor : cursor + count])
        cursor += count
    return groups


def apply_affine_transform(
    points: list[tuple[float, float]],
    transform_values: list[float] | tuple[float, ...] | None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    if not transform_values or len(transform_values) < 6:
        return []

    a, b, c, d, e, f = [float(value) for value in transform_values[:6]]

    def standard(point: tuple[float, float]) -> tuple[float, float]:
        x_pos, y_pos = point
        return a * x_pos + b * y_pos + c, d * x_pos + e * y_pos + f

    def alt1(point: tuple[float, float]) -> tuple[float, float]:
        x_pos, y_pos = point
        return a * x_pos + b * y_pos + e, c * x_pos + d * y_pos + f

    def alt2(point: tuple[float, float]) -> tuple[float, float]:
        x_pos, y_pos = point
        return a * x_pos + c * y_pos + e, b * x_pos + d * y_pos + f

    variants = []
    for name, candidate in (
        ("affine-standard", standard),
        ("affine-alt1", alt1),
        ("affine-alt2", alt2),
    ):
        variants.append((name, [candidate(point) for point in points]))
    return variants


def map_e2e_point_to_fundus(
    point: tuple[float, float],
    width: int,
    height: int,
    field_size_degrees: float,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[float, float]:
    x_pos, y_pos = point
    if flip_x:
        x_pos = -x_pos
    if flip_y:
        y_pos = -y_pos

    x_scale = width / field_size_degrees
    y_scale = height / field_size_degrees
    x_pixel = width / 2.0 + x_pos * x_scale
    y_pixel = height / 2.0 + y_pos * y_scale
    return x_pixel, y_pixel


def score_projected_points(points: list[tuple[float, float]], width: int, height: int) -> float:
    if not points:
        return -1.0

    inside = 0
    outside_penalty = 0.0
    for x_pos, y_pos in points:
        if -0.2 * width <= x_pos <= 1.2 * width and -0.2 * height <= y_pos <= 1.2 * height:
            inside += 1
        else:
            dx = max(-0.2 * width - x_pos, 0.0, x_pos - 1.2 * width)
            dy = max(-0.2 * height - y_pos, 0.0, y_pos - 1.2 * height)
            outside_penalty += dx + dy

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    spread = max(xs) - min(xs) + max(ys) - min(ys)
    spread_score = min(spread / max(width + height, 1), 2.5)
    penalty_score = outside_penalty / max(width + height, 1)
    return inside + spread_score - penalty_score


def points_to_segments(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(points[index], points[index + 1]) for index in range(0, len(points), 2)]


def build_parallel_segments(
    num_slices: int,
    fundus_shape: tuple[int, int] | tuple[int, int, int],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    height, width = fundus_shape[:2]
    if num_slices <= 0:
        return []

    x_min = width * 0.18
    x_max = width * 0.82
    y_min = height * 0.18
    y_max = height * 0.82

    if num_slices == 1:
        x_pos = (x_min + x_max) / 2.0
        return [((x_pos, y_min), (x_pos, y_max))]

    segments = []
    for index in range(num_slices):
        ratio = index / (num_slices - 1)
        x_pos = x_min + ratio * (x_max - x_min)
        segments.append(((x_pos, y_min), (x_pos, y_max)))
    return segments



def compute_scan_bounds(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float, float, float] | None:
    if not segments:
        return None
    xs = [point[0] for segment in segments for point in segment]
    ys = [point[1] for segment in segments for point in segment]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def select_matching_fundus(volume: Any, fundus_images: list[Any], index: int) -> tuple[Any | None, str]:
    if not fundus_images:
        return None, "projection-fallback"

    fundus_by_id = {
        getattr(image, "image_id", None): image
        for image in fundus_images
        if getattr(image, "image_id", None)
    }
    volume_id = getattr(volume, "volume_id", None)
    if volume_id and volume_id in fundus_by_id:
        return fundus_by_id[volume_id], "matched-by-id"

    laterality = getattr(volume, "laterality", None)
    if laterality:
        same_laterality = [
            image for image in fundus_images if getattr(image, "laterality", None) == laterality
        ]
        if len(same_laterality) == 1:
            return same_laterality[0], "matched-by-laterality"

    return fundus_images[min(index, len(fundus_images) - 1)], "matched-by-index-fallback"


def choose_localizer(
    localizers: list[dict[str, Any]],
    volume_index: int,
    total_volumes: int,
) -> tuple[dict[str, Any] | None, str]:
    if not localizers:
        return None, "missing"
    if len(localizers) == total_volumes:
        return localizers[volume_index], "matched-by-count"
    if len(localizers) == 1:
        return localizers[0], "single-shared"
    return None, f"disabled-count-mismatch:{len(localizers)}/{total_volumes}"


def choose_e2e_field_size_degrees(
    eye_data_entries: list[dict[str, Any]],
    *,
    laterality: str,
    volume_index: int,
    total_volumes: int,
    default_field_size_degrees: float = 30.0,
) -> tuple[float, str]:
    laterality_code = normalize_laterality_code(laterality)
    candidates = []
    for entry in eye_data_entries:
        if not isinstance(entry, dict):
            continue
        field_size = safe_positive_float(entry.get("vfieldMean"))
        if field_size is None:
            continue
        candidates.append((normalize_laterality_code(entry.get("eyeSide")), field_size))

    if laterality_code:
        same_laterality = [field_size for eye_side, field_size in candidates if eye_side == laterality_code]
        if len(same_laterality) == 1:
            return same_laterality[0], "eye-data-laterality"

    if len(candidates) == total_volumes and 0 <= volume_index < len(candidates):
        return candidates[volume_index][1], "eye-data-count"

    if len(candidates) == 1:
        return candidates[0][1], "eye-data-single"

    if candidates:
        return candidates[0][1], "eye-data-first"

    return default_field_size_degrees, "default-30deg"


def project_e2e_overlay(
    group: list[dict[str, Any]],
    localizer_entry: dict[str, Any] | None,
    localizer_mode: str,
    fundus_shape: tuple[int, int] | tuple[int, int, int],
    field_size_degrees: float = 30.0,
) -> OverlayState:
    if not group:
        scan_segments = build_parallel_segments(0, fundus_shape)
        return OverlayState(
            scan_segments=scan_segments,
            overlay_mode="fallback-parallel",
            localizer_mode=localizer_mode,
            projection_mode="no-bscan-metadata",
            warning="B-scan 元数据缺失，定位线使用降级模式。",
            bounds=compute_scan_bounds(scan_segments),
        )

    raw_segments = []
    for item in group:
        raw_segments.append(
            (
                (float(item.get("posX1", 0.0)), float(item.get("posY1", 0.0))),
                (float(item.get("posX2", 0.0)), float(item.get("posY2", 0.0))),
            )
        )

    fundus_height, fundus_width = fundus_shape[:2]
    points = [point for segment in raw_segments for point in segment]
    field_size_degrees = safe_positive_float(field_size_degrees) or 30.0
    transform_values = (localizer_entry or {}).get("transform")

    candidate_sets: list[tuple[str, list[tuple[float, float]]]] = []
    for flip_x in (False, True):
        for flip_y in (False, True):
            label = f"raw-map/fx-{int(flip_x)}/fy-{int(flip_y)}"
            mapped_points = [
                map_e2e_point_to_fundus(
                    point=point,
                    width=fundus_width,
                    height=fundus_height,
                    field_size_degrees=field_size_degrees,
                    flip_x=flip_x,
                    flip_y=flip_y,
                )
                for point in points
            ]
            candidate_sets.append((label, mapped_points))

            if localizer_entry is not None:
                for affine_name, affine_points in apply_affine_transform(mapped_points, transform_values):
                    candidate_sets.append((f"{label}/{affine_name}", affine_points))

    best_points: list[tuple[float, float]] | None = None
    best_label = "fallback-parallel"
    best_score = float("-inf")
    for label, candidate in candidate_sets:
        score = score_projected_points(candidate, fundus_width, fundus_height)
        if score > best_score:
            best_score = score
            best_points = candidate
            best_label = label

    if best_points is None:
        scan_segments = build_parallel_segments(len(group), fundus_shape)
        return OverlayState(
            scan_segments=scan_segments,
            overlay_mode="fallback-parallel",
            localizer_mode=localizer_mode,
            projection_mode="score-failed",
            warning="定位拟合失败，已降级为平行定位线。",
            bounds=compute_scan_bounds(scan_segments),
        )

    warning = ""
    overlay_mode = "e2e-metadata"
    if localizer_entry is None and localizer_mode.startswith("disabled-count-mismatch"):
        warning = "已禁用 localizer，当前采用安全投影模式。"
    elif localizer_entry is None and localizer_mode == "missing":
        warning = "未提供 localizer，当前采用 B-scan 几何投影。"
    elif "affine" in best_label and localizer_entry is not None:
        warning = f"使用 localizer 变换优化定位：{best_label}"

    scan_segments = points_to_segments(best_points)
    return OverlayState(
        scan_segments=scan_segments,
        overlay_mode=overlay_mode,
        localizer_mode=localizer_mode,
        projection_mode=best_label,
        warning=warning,
        bounds=compute_scan_bounds(scan_segments),
    )


def build_metadata_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False, default=str)


def format_laterality(laterality: str) -> str:
    if laterality == "R":
        return "右眼"
    if laterality == "L":
        return "左眼"
    return "未标注"


def classify_display_severity(model: VolumeViewModel) -> str:
    if model.fundus_match_mode in {"projection-fallback", "matched-by-index-fallback"}:
        return "risk"
    if model.overlay.overlay_mode == "fallback-parallel":
        return "risk"
    if model.overlay.warning:
        return "degraded"
    if model.fundus_match_mode in {"matched-by-laterality", "embedded-fundus"}:
        return "degraded"
    return "normal"


def format_fundus_match_mode(mode: str) -> tuple[str, str]:
    mapping = {
        "matched-by-id": ("精确匹配", "眼底参考图已与当前 volume 按唯一标识精确对应。"),
        "matched-by-laterality": ("按眼别匹配", "眼底参考图按眼别推断匹配，建议结合图像人工确认。"),
        "matched-by-index-fallback": ("按顺序兼容匹配", "眼底参考图按读取顺序兼容对应，建议人工复核。"),
        "projection-fallback": ("投影替代", "未找到可用眼底参考图，使用 OCT 投影图替代显示。"),
        "embedded-fundus": ("内嵌眼底图", "当前文件自带眼底参考图。"),
    }
    return mapping.get(mode, ("未知匹配方式", mode))


def format_overlay_mode(mode: str) -> tuple[str, str]:
    mapping = {
        "e2e-metadata": ("元数据定位", "定位线由 E2E 的 B-scan 几何信息投影生成。"),
        "fallback-parallel": ("平行线降级", "无法稳定恢复真实定位，使用平行定位线近似显示。"),
        "parallel-approximation": ("平行线近似", "当前文件使用平行定位线作近似显示。"),
    }
    return mapping.get(mode, ("未知定位模式", mode))


def format_projection_mode(mode: str) -> tuple[str, str]:
    if mode.startswith("raw-map/"):
        return "几何直接投影", "使用 B-scan 几何坐标直接映射到眼底参考图。"
    if "affine" in mode:
        return "Localizer 优化投影", "使用 Localizer 仿射变换优化定位显示。"
    mapping = {
        "parallel-default": ("默认平行线", "使用默认平行定位线作为近似显示。"),
        "score-failed": ("拟合失败已降级", "定位拟合失败，已自动改用平行定位线。"),
        "no-bscan-metadata": ("缺少 B-scan 元数据", "未读取到 B-scan 几何元数据，无法恢复真实定位。"),
    }
    return mapping.get(mode, ("未知投影策略", mode))


def format_localizer_mode(mode: str) -> tuple[str, str]:
    if mode.startswith("disabled-count-mismatch"):
        return "已禁用", "定位参考数量与 volume 不一致，已自动禁用以避免错配。"
    mapping = {
        "matched-by-count": ("已启用", "定位参考数量与 volume 一一对应，已用于定位评估。"),
        "single-shared": ("共享定位参考", "仅发现一个定位参考，按共享模式参与定位评估。"),
        "missing": ("缺失", "未找到可用定位参考。"),
        "not-applicable": ("不适用", "当前文件类型不提供定位参考。"),
    }
    return mapping.get(mode, ("未知状态", mode))


def build_display_text(model: VolumeViewModel, slice_index: int = 0) -> dict[str, str]:
    laterality_text = format_laterality(model.laterality)
    fundus_title, fundus_reason = format_fundus_match_mode(model.fundus_match_mode)
    overlay_title, overlay_reason = format_overlay_mode(model.overlay.overlay_mode)
    projection_title, projection_reason = format_projection_mode(model.overlay.projection_mode)
    localizer_title, localizer_reason = format_localizer_mode(model.overlay.localizer_mode)
    severity = classify_display_severity(model)

    if severity == "risk":
        warning_text = (
            model.overlay.warning
            or "当前显示存在较高不确定性，建议结合原始图像与检查流程人工复核。"
        )
    elif severity == "degraded":
        warning_text = model.overlay.warning or "当前显示可用，但采用了兼容或降级策略。"
    else:
        warning_text = "当前显示可信，未触发降级策略。"

    volume_summary = (
        f"{laterality_text} | {len(model.slices)} 层 | 眼底{fundus_title} | "
        f"定位{projection_title}"
    )
    if severity == "degraded":
        volume_summary += " | 已降级"
    elif severity == "risk":
        volume_summary += " | 建议复核"

    return {
        "severity": severity,
        "volume_summary": volume_summary,
        "summary": (
            f"{model.source_kind} 检查 | {laterality_text} | {len(model.slices)} 层 | "
            f"眼底：{fundus_title} | 定位：{projection_title}"
        ),
        "volume_info": (
            f"{laterality_text} | Volume ID={model.volume_id} | B-scan 尺寸="
            f"{list(model.slice_shape) if model.slice_shape else '-'} | 像素间距={model.pixel_spacing or '-'}"
        ),
        "fundus_info": (
            f"眼底匹配={fundus_title} | 来源={model.fundus_source_label} | 图像尺寸={list(model.fundus_shape)} | {fundus_reason}"
        ),
        "geometry_info": (
            f"{overlay_title} | 投影={projection_title} | 定位参考={localizer_title} | "
            f"{projection_reason} {localizer_reason}"
        ),
        "warning_text": warning_text,
        "status_text": (
            f"{Path(model.filepath).name} | {laterality_text} | 第 {slice_index + 1}/{len(model.slices)} 层 | "
            f"眼底{fundus_title} | 定位{projection_title}"
        ),
        "fundus_title": fundus_title,
        "projection_title": projection_title,
    }


def build_status_lines(model: VolumeViewModel) -> list[str]:
    fundus_title, fundus_reason = format_fundus_match_mode(model.fundus_match_mode)
    overlay_title, _ = format_overlay_mode(model.overlay.overlay_mode)
    projection_title, projection_reason = format_projection_mode(model.overlay.projection_mode)
    localizer_title, localizer_reason = format_localizer_mode(model.overlay.localizer_mode)
    return [
        f"来源: {model.source_kind}",
        f"Volume: {model.volume_id}",
        f"眼别: {format_laterality(model.laterality)}",
        f"眼底来源: {model.fundus_source_label}",
        f"眼底匹配: {fundus_title}（{fundus_reason}）",
        f"定位模式: {overlay_title}",
        f"投影策略: {projection_title}（{projection_reason}）",
        f"定位参考: {localizer_title}（{localizer_reason}）",
    ]


def format_volume_summary(model: VolumeViewModel) -> str:
    return build_display_text(model)["volume_summary"]


def load_e2e_file(filepath: Path) -> list[VolumeViewModel]:
    from oct_converter.readers.e2e import E2E

    reader = E2E(filepath)
    volumes = reader.read_oct_volume()
    fundus_images = reader.read_fundus_image()
    try:
        metadata = reader.read_all_metadata()
    except Exception:
        metadata = {}

    if not volumes:
        raise RuntimeError("E2E 文件中未读取到 OCT 体数据。")

    slice_counts = [volume.num_slices for volume in volumes]
    bscan_groups = partition_bscan_metadata(metadata.get("bscan_data", []), slice_counts)
    localizers = metadata.get("localizer", [])
    eye_data_entries = metadata.get("eye_data", [])

    models = []
    total_volumes = len(volumes)
    for index, volume in enumerate(volumes):
        matched_fundus, fundus_match_mode = select_matching_fundus(volume, fundus_images, index)
        if matched_fundus is not None:
            fundus = to_display_image(matched_fundus.image)
            fundus_source_label = getattr(matched_fundus, "image_id", None) or "fundus-image"
        else:
            fundus = make_projection_fundus(list(volume.volume))
            fundus_match_mode = "projection-fallback"
            fundus_source_label = "projection-fallback"

        group = bscan_groups[index] if index < len(bscan_groups) else []
        localizer_entry, localizer_mode = choose_localizer(localizers, index, total_volumes)
        field_size_degrees, field_size_mode = choose_e2e_field_size_degrees(
            eye_data_entries,
            laterality=volume.laterality or "",
            volume_index=index,
            total_volumes=total_volumes,
        )
        overlay = project_e2e_overlay(
            group,
            localizer_entry,
            localizer_mode,
            fundus.shape,
            field_size_degrees=field_size_degrees,
        )
        if not group:
            overlay.scan_segments = build_parallel_segments(volume.num_slices, fundus.shape)
            overlay.bounds = compute_scan_bounds(overlay.scan_segments)

        label = volume.volume_id or f"E2E Volume {index + 1}"
        if volume.laterality:
            label = f"{label} ({volume.laterality})"

        slice_shape = tuple(np.asarray(volume.volume[0]).shape) if volume.volume else tuple()
        summary = {
            "file": str(filepath),
            "source": "E2E",
            "volume_id": volume.volume_id,
            "laterality": volume.laterality,
            "num_slices": volume.num_slices,
            "slice_shape": list(slice_shape),
            "pixel_spacing_mm": volume.pixel_spacing,
            "fundus_shape": list(fundus.shape[:2]),
            "fundus_source": fundus_source_label,
            "fundus_match_mode": fundus_match_mode,
            "overlay_mode": overlay.overlay_mode,
            "projection_mode": overlay.projection_mode,
            "localizer_mode": overlay.localizer_mode,
            "field_size_degrees": field_size_degrees,
            "field_size_mode": field_size_mode,
            "warning": overlay.warning,
        }

        model = VolumeViewModel(
            label=label,
            volume_id=volume.volume_id or f"volume_{index + 1}",
            laterality=volume.laterality or "",
            source_kind="E2E",
            filepath=str(filepath),
            slices=list(volume.volume),
            contours=volume.contours,
            fundus=fundus,
            fundus_shape=tuple(fundus.shape[:2]),
            fundus_source_label=fundus_source_label,
            fundus_match_mode=fundus_match_mode,
            overlay=overlay,
            pixel_spacing=volume.pixel_spacing,
            slice_shape=slice_shape,
            metadata_text=build_metadata_text(summary),
            status_lines=[],
        )
        model.status_lines = build_status_lines(model)
        models.append(model)

    return models


def load_fda_file(filepath: Path) -> list[VolumeViewModel]:
    from oct_converter.readers.fda import FDA

    reader = FDA(filepath)
    volume = reader.read_oct_volume()
    if volume is None or not volume.volume:
        raise RuntimeError("FDA 文件中未读取到 OCT 体数据。")

    fundus_image = reader.read_fundus_image()
    if fundus_image is None:
        fundus_image = reader.read_fundus_image_gray_scale()

    if fundus_image is not None:
        fundus = to_display_image(fundus_image.image)
        fundus_source_label = getattr(fundus_image, "image_id", None) or "fundus-image"
        fundus_match_mode = "embedded-fundus"
    else:
        fundus = make_projection_fundus(list(volume.volume))
        fundus_source_label = "projection-fallback"
        fundus_match_mode = "projection-fallback"

    scan_segments = build_parallel_segments(volume.num_slices, fundus.shape)
    overlay = OverlayState(
        scan_segments=scan_segments,
        overlay_mode="parallel-approximation",
        localizer_mode="not-applicable",
        projection_mode="parallel-default",
        warning="FDA 当前使用平行定位线近似显示。",
        bounds=compute_scan_bounds(scan_segments),
    )
    label = Path(filepath).stem
    if volume.laterality:
        label = f"{label} ({volume.laterality})"

    slice_shape = tuple(np.asarray(volume.volume[0]).shape) if volume.volume else tuple()
    summary = {
        "file": str(filepath),
        "source": "FDA",
        "volume_id": volume.volume_id,
        "laterality": volume.laterality,
        "num_slices": volume.num_slices,
        "slice_shape": list(slice_shape),
        "pixel_spacing_mm": volume.pixel_spacing,
        "fundus_shape": list(fundus.shape[:2]),
        "fundus_source": fundus_source_label,
        "fundus_match_mode": fundus_match_mode,
        "overlay_mode": overlay.overlay_mode,
        "projection_mode": overlay.projection_mode,
        "localizer_mode": overlay.localizer_mode,
        "warning": overlay.warning,
    }

    model = VolumeViewModel(
        label=label,
        volume_id=volume.volume_id or Path(filepath).stem,
        laterality=volume.laterality or "",
        source_kind="FDA",
        filepath=str(filepath),
        slices=list(volume.volume),
        contours=volume.contours,
        fundus=fundus,
        fundus_shape=tuple(fundus.shape[:2]),
        fundus_source_label=fundus_source_label,
        fundus_match_mode=fundus_match_mode,
        overlay=overlay,
        pixel_spacing=volume.pixel_spacing,
        slice_shape=slice_shape,
        metadata_text=build_metadata_text(summary),
        status_lines=[],
    )
    model.status_lines = build_status_lines(model)
    return [model]


def load_models(filepath: str | Path) -> list[VolumeViewModel]:
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".e2e":
        return load_e2e_file(path)
    if suffix == ".fda":
        return load_fda_file(path)
    raise ValueError("仅支持导入 .E2E 和 .FDA 文件。")


class FundusBscanCanvas(FigureCanvas):
    def __init__(self, parent: QWidget | None = None):
        figure = Figure(figsize=(11, 5), tight_layout=True, facecolor="#111827")
        self.ax_fundus = figure.add_subplot(1, 2, 1)
        self.ax_bscan = figure.add_subplot(1, 2, 2)
        super().__init__(figure)
        self.setParent(parent)
        for axis in (self.ax_fundus, self.ax_bscan):
            axis.set_facecolor("#0B1220")



class OCTViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海德堡 OCT 专业浏览器")
        self.resize(1560, 920)

        self.settings = QSettings("OpenAI", "OCTE2EFDAViewer")
        self.filepath: str | None = None
        self.models: list[VolumeViewModel] = []
        self.current_volume_index = 0
        self.current_slice_index = 0

        self._apply_styles()
        self.canvas = FundusBscanCanvas(self)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)
        self.setStatusBar(QStatusBar(self))

        self.open_button = QPushButton("打开文件")
        self.open_button.clicked.connect(self.open_file_dialog)
        self.snapshot_button = QPushButton("保存快照")
        self.snapshot_button.clicked.connect(self.save_snapshot)

        self.file_label = QLabel("尚未打开文件")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("PathLabel")

        self.volume_list = QListWidget()
        self.volume_list.currentRowChanged.connect(self.on_volume_changed)

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

        self.slice_info_label = QLabel("当前切片: -")
        self.slice_info_label.setObjectName("SliceInfo")

        self.show_segments_checkbox = QCheckBox("显示定位线")
        self.show_segments_checkbox.setChecked(True)
        self.show_segments_checkbox.toggled.connect(self.redraw_views)

        self.show_contours_checkbox = QCheckBox("显示分割")
        self.show_contours_checkbox.setChecked(True)
        self.show_contours_checkbox.toggled.connect(self.redraw_views)

        self.highlight_slice_checkbox = QCheckBox("高亮当前切片")
        self.highlight_slice_checkbox.setChecked(True)
        self.highlight_slice_checkbox.toggled.connect(self.redraw_views)

        self.show_bounds_checkbox = QCheckBox("显示定位范围框")
        self.show_bounds_checkbox.setChecked(True)
        self.show_bounds_checkbox.toggled.connect(self.redraw_views)

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

        self.volume_info_value = QLabel("-")
        self.fundus_info_value = QLabel("-")
        self.geometry_info_value = QLabel("-")
        self.warning_value = QLabel("-")
        self.warning_value.setWordWrap(True)
        self.warning_value.setObjectName("WarningLabel")

        self.summary_label = QLabel("等待导入 OCT 文件")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("SummaryLabel")

        self.legend_label = QLabel("绿色=当前切片定位线，黄色=其他切片，青色/粉色等=分割轮廓")
        self.legend_label.setWordWrap(True)
        self.legend_label.setObjectName("LegendLabel")

        self.metadata_edit = QPlainTextEdit()
        self.metadata_edit.setReadOnly(True)
        self.metadata_edit.setFont(QFont("Consolas", 10))

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
            QSpinBox, QPlainTextEdit, QListWidget {
                background: #0F172A;
                color: #E5E7EB;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 6px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: #1D4ED8;
                color: white;
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
            QLabel#PathLabel, QLabel#SummaryLabel, QLabel#LegendLabel, QLabel#WarningLabel {
                background: #0F172A;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel#WarningLabel { color: #FDE68A; }
            QLabel#SliceInfo {
                background: #0369A1;
                color: white;
                border-radius: 12px;
                padding: 6px 12px;
                font-weight: 700;
            }
            QStatusBar { background: #0F172A; color: #CBD5E1; }
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
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_file_dialog)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_snapshot)

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开", self)
        open_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(open_action)

        snapshot_action = QAction("快照", self)
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
        source_layout.addWidget(self.file_label)

        volume_group = QGroupBox("Volume 列表")
        volume_layout = QVBoxLayout(volume_group)
        volume_layout.addWidget(self.volume_list)
        volume_button_row = QWidget()
        volume_button_layout = QHBoxLayout(volume_button_row)
        volume_button_layout.setContentsMargins(0, 0, 0, 0)
        volume_button_layout.addWidget(self.prev_volume_button)
        volume_button_layout.addWidget(self.next_volume_button)
        volume_layout.addWidget(volume_button_row)

        navigation_group = QGroupBox("切片导航")
        navigation_layout = QGridLayout(navigation_group)
        navigation_layout.addWidget(self.slice_info_label, 0, 0, 1, 2)
        navigation_layout.addWidget(self.prev_slice_button, 1, 0)
        navigation_layout.addWidget(self.next_slice_button, 1, 1)
        navigation_layout.addWidget(self.slice_slider, 2, 0, 1, 2)
        navigation_layout.addWidget(self.slice_spin, 3, 0, 1, 2)

        display_group = QGroupBox("显示控制")
        display_layout = QGridLayout(display_group)
        display_layout.addWidget(self.show_segments_checkbox, 0, 0, 1, 2)
        display_layout.addWidget(self.show_contours_checkbox, 1, 0, 1, 2)
        display_layout.addWidget(self.highlight_slice_checkbox, 2, 0, 1, 2)
        display_layout.addWidget(self.show_bounds_checkbox, 3, 0, 1, 2)
        display_layout.addWidget(QLabel("B-scan 对比度"), 4, 0)
        display_layout.addWidget(self.contrast_slider, 4, 1)
        display_layout.addWidget(QLabel("B-scan 亮度"), 5, 0)
        display_layout.addWidget(self.brightness_slider, 5, 1)
        display_layout.addWidget(self.reset_view_button, 6, 0, 1, 2)

        info_group = QGroupBox("检查信息")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self._make_info_row("Volume", self.volume_info_value))
        info_layout.addWidget(self._make_info_row("Fundus", self.fundus_info_value))
        info_layout.addWidget(self._make_info_row("Geometry", self.geometry_info_value))
        info_layout.addWidget(self._make_info_row("提示", self.warning_value))
        info_layout.addWidget(self.summary_label)
        info_layout.addWidget(self.legend_label)

        control_layout.addWidget(source_group)
        control_layout.addWidget(volume_group)
        control_layout.addWidget(navigation_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(info_group)
        control_layout.addStretch(1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas, stretch=4)
        metadata_title = QLabel("检查元数据")
        metadata_title.setStyleSheet("font-weight: 700; color: #93C5FD; padding: 4px 0;")
        right_layout.addWidget(metadata_title)
        right_layout.addWidget(self.metadata_edit, stretch=2)

        splitter = QSplitter()
        splitter.addWidget(control_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1160])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter)
        self.setCentralWidget(container)


    def _set_empty_state(self):
        self.volume_list.blockSignals(True)
        self.volume_list.clear()
        self.volume_list.blockSignals(False)

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(0)
        self.slice_spin.setValue(0)
        self.slice_spin.setSuffix("")
        self.slice_spin.blockSignals(False)

        self.volume_info_value.setText("-")
        self.fundus_info_value.setText("-")
        self.geometry_info_value.setText("-")
        self.warning_value.setText("-")
        self.apply_warning_style("normal")
        self.metadata_edit.setPlainText("")
        self.summary_label.setText("等待导入 OCT 文件")
        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()
        self.canvas.ax_fundus.set_title("眼底参考图", color="#E5E7EB")
        self.canvas.ax_bscan.set_title("B 扫图像", color="#E5E7EB")
        self.canvas.draw_idle()
        self.statusBar().showMessage("就绪")

    def apply_warning_style(self, severity: str):
        colors = {
            "normal": ("#D1FAE5", "#064E3B"),
            "degraded": ("#FEF3C7", "#92400E"),
            "risk": ("#FECACA", "#991B1B"),
        }
        background, foreground = colors.get(severity, ("#E5E7EB", "#1F2937"))
        self.warning_value.setStyleSheet(
            "background: {bg}; color: {fg}; border: 1px solid #334155; border-radius: 8px; padding: 10px;".format(
                bg=background,
                fg=foreground,
            )
        )

    def get_last_open_dir(self) -> str:
        last_file = self.settings.value("last_file", "", type=str)
        if last_file and Path(last_file).exists():
            return str(Path(last_file).parent)
        last_dir = self.settings.value("last_dir", "", type=str)
        if last_dir and Path(last_dir).exists():
            return last_dir
        return ""

    def remember_file(self, filepath: str) -> None:
        filepath = str(Path(filepath).resolve())
        self.settings.setValue("last_file", filepath)
        self.settings.setValue("last_dir", str(Path(filepath).parent))

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 E2E/FDA 文件",
            self.get_last_open_dir(),
            "OCT Files (*.E2E *.e2e *.FDA *.fda)",
        )
        if filename:
            self.load_file(filename)

    def load_file(self, filepath: str):
        try:
            models = load_models(filepath)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return

        self.filepath = str(filepath)
        self.models = models
        self.current_volume_index = 0
        self.current_slice_index = 0
        self.remember_file(filepath)

        self.file_label.setText(self.filepath)
        self.volume_list.blockSignals(True)
        self.volume_list.clear()
        for model in self.models:
            QListWidgetItem(format_volume_summary(model), self.volume_list)
        self.volume_list.blockSignals(False)
        if self.models:
            self.volume_list.setCurrentRow(0)
        self.statusBar().showMessage(f"已加载 {len(self.models)} 个 volume", 5000)
        self.refresh_volume()

    def current_model(self) -> VolumeViewModel | None:
        if not self.models:
            return None
        return self.models[self.current_volume_index]

    def refresh_volume(self):
        model = self.current_model()
        if model is None:
            self._set_empty_state()
            return
        display = build_display_text(model, self.current_slice_index)

        max_index = max(0, len(model.slices) - 1)
        self.current_slice_index = min(self.current_slice_index, max_index)
        display = build_display_text(model, self.current_slice_index)

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

        self.metadata_edit.setPlainText(model.metadata_text)
        self.summary_label.setText(display["summary"])
        self.volume_info_value.setText(display["volume_info"])
        self.fundus_info_value.setText(display["fundus_info"])
        self.geometry_info_value.setText(display["geometry_info"])
        self.warning_value.setText(display["warning_text"])
        self.apply_warning_style(display["severity"])
        self.redraw_views()

    def compute_segment_linewidth(
        self,
        segment: tuple[tuple[float, float], tuple[float, float]],
        highlighted: bool,
        fundus_shape: tuple[int, int],
    ) -> float:
        (start_x, start_y), (end_x, end_y) = segment
        length = float(np.hypot(end_x - start_x, end_y - start_y))
        baseline = max(1.0, 0.003 * max(fundus_shape))
        length_factor = min(1.8, max(0.8, length / max(fundus_shape) * 1.4))
        if highlighted:
            return baseline * (1.8 + 0.45 * length_factor)
        return baseline * (0.9 + 0.25 * length_factor)

    def redraw_views(self):
        model = self.current_model()
        if model is None:
            return
        display = build_display_text(model, self.current_slice_index)

        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()

        fundus = model.fundus
        if fundus.ndim == 2:
            self.canvas.ax_fundus.imshow(fundus, cmap="gray", origin="upper")
        else:
            self.canvas.ax_fundus.imshow(fundus, origin="upper")
        self.canvas.ax_fundus.axis("off")
        self.canvas.ax_fundus.set_title(
            f"眼底参考图 / {model.source_kind} / {model.fundus_source_label}",
            color="#E5E7EB",
            fontsize=13,
        )

        if self.show_bounds_checkbox.isChecked() and model.overlay.bounds is not None:
            x0, y0, width, height = model.overlay.bounds
            self.canvas.ax_fundus.add_patch(
                Rectangle(
                    (x0, y0),
                    max(width, 1.0),
                    max(height, 1.0),
                    linewidth=1.4,
                    edgecolor="#EF4444",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.75,
                )
            )

        if self.show_segments_checkbox.isChecked():
            for index, segment in enumerate(model.overlay.scan_segments):
                highlighted = self.highlight_slice_checkbox.isChecked() and index == self.current_slice_index
                color = "#66ff66" if highlighted else "#ffd966"
                alpha = 0.96 if highlighted else 0.3
                linewidth = self.compute_segment_linewidth(segment, highlighted, model.fundus_shape)
                (start_x, start_y), (end_x, end_y) = segment
                self.canvas.ax_fundus.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                )

        bscan = to_display_image(model.slices[self.current_slice_index])
        bscan = apply_image_window(
            bscan,
            contrast_percent=self.contrast_slider.value(),
            brightness_offset=self.brightness_slider.value(),
        )
        self.canvas.ax_bscan.imshow(bscan, cmap="gray", aspect="auto", origin="upper")
        self.canvas.ax_bscan.axis("off")
        self.canvas.ax_bscan.set_title(
            f"B 扫图像 {self.current_slice_index + 1}/{len(model.slices)}",
            color="#E5E7EB",
            fontsize=13,
        )

        contour_colors = ["#00ffff", "#ff66cc", "#66ff66", "#ff9933", "#66b3ff", "#ffffff"]
        if self.show_contours_checkbox.isChecked() and model.contours:
            color_index = 0
            for _, values in model.contours.items():
                if self.current_slice_index >= len(values):
                    continue
                contour = values[self.current_slice_index]
                if contour is None:
                    continue
                contour = np.asarray(contour)
                if contour.size == 0 or np.isnan(contour).all():
                    continue
                x_coords = np.arange(contour.shape[0])
                self.canvas.ax_bscan.plot(
                    x_coords,
                    contour,
                    color=contour_colors[color_index % len(contour_colors)],
                    linewidth=1.2,
                    alpha=0.92,
                )
                color_index += 1

        self.slice_info_label.setText(f"当前切片: {self.current_slice_index + 1} / {len(model.slices)}")
        self.statusBar().showMessage(display["status_text"])
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

        self.redraw_views()

    def previous_slice(self):
        self.set_slice_index(self.current_slice_index - 1)

    def next_slice(self):
        self.set_slice_index(self.current_slice_index + 1)

    def previous_volume(self):
        if not self.models:
            return
        target = max(0, self.current_volume_index - 1)
        if target != self.current_volume_index:
            self.volume_list.setCurrentRow(target)

    def next_volume(self):
        if not self.models:
            return
        target = min(len(self.models) - 1, self.current_volume_index + 1)
        if target != self.current_volume_index:
            self.volume_list.setCurrentRow(target)

    def jump_to_last_slice(self):
        model = self.current_model()
        if model is None:
            return
        self.set_slice_index(len(model.slices) - 1)

    def reset_display_controls(self):
        self.show_segments_checkbox.setChecked(True)
        self.show_contours_checkbox.setChecked(True)
        self.highlight_slice_checkbox.setChecked(True)
        self.show_bounds_checkbox.setChecked(True)
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)
        self.redraw_views()

    def save_snapshot(self):
        model = self.current_model()
        default_name = "oct_viewer_snapshot.png"
        if model is not None:
            default_name = f"{Path(model.filepath).stem}_{model.volume_id}_slice_{self.current_slice_index + 1:03d}.png"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前视图",
            str(Path(self.get_last_open_dir() or ".") / default_name),
            "PNG Image (*.png)",
        )
        if not filename:
            return
        self.canvas.figure.savefig(filename, dpi=150, bbox_inches="tight")
        self.statusBar().showMessage(f"快照已保存: {filename}", 5000)

    def on_volume_changed(self, index: int):
        if index < 0 or index >= len(self.models):
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
        if not model.overlay.scan_segments:
            return

        target_x = float(event.xdata)
        target_y = float(event.ydata)
        nearest_index = 0
        nearest_distance = float("inf")
        for index, segment in enumerate(model.overlay.scan_segments):
            (start_x, start_y), (end_x, end_y) = segment
            center_x = (start_x + end_x) / 2.0
            center_y = (start_y + end_y) / 2.0
            distance = (target_x - center_x) ** 2 + (target_y - center_y) ** 2
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        self.set_slice_index(nearest_index)

    def on_canvas_scroll(self, event):
        if event.inaxes not in (self.canvas.ax_bscan, self.canvas.ax_fundus):
            return
        step = 1 if event.button == "up" else -1
        self.set_slice_index(self.current_slice_index + step)


def main():
    args = parse_args()
    application = QApplication(sys.argv)
    window = OCTViewerWindow()
    window.show()
    if args.filepath:
        window.load_file(args.filepath)
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
