from __future__ import annotations

import json
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oct_converter.image_types import FundusImageWithMetaData
from oct_converter.readers import BOCT, Dicom, E2E, FDA, FDS, IMG, POCT

from .vendor_modes import (
    FILE_DIALOG_TYPES,
    NORMALIZED_SUPPORTED_EXTENSIONS,
    SHIWEI_UPLOAD_HINT,
    SUPPORTED_EXTENSIONS,
    TUPAI_UPLOAD_HINT,
    VENDOR_MODE_FILE_DIALOG_TYPES,
    VENDOR_MODE_LABELS,
    build_vendor_validation_error,
    is_supported_suffix_for_vendor,
    normalize_vendor_mode,
)
from .shiwei_loader import load_shiwei_oct_dataset, resolve_shiwei_input_dir
from .tupai_loader import load_tupai_oct_dataset, resolve_tupai_input_dir
from .zeiss_loader import load_zeiss_oct_dataset, resolve_zeiss_exam_dirs


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array, dtype=np.uint8)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    array = (array - min_value) * (255.0 / (max_value - min_value))
    return np.clip(array, 0, 255).astype(np.uint8)


def apply_window(image: np.ndarray, contrast_percent: int, brightness_offset: int) -> np.ndarray:
    adjusted = image.astype(np.float32) * (max(1, contrast_percent) / 100.0) + float(
        brightness_offset
    )
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.ndim == 2:
        ok, payload = cv2.imencode(".png", array)
    elif array.ndim == 3 and array.shape[2] == 1:
        ok, payload = cv2.imencode(".png", array[:, :, 0])
    elif array.ndim == 3 and array.shape[2] == 3:
        ok, payload = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    elif array.ndim == 3 and array.shape[2] == 4:
        ok, payload = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA))
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    if not ok:
        raise ValueError("Failed to encode PNG")
    return payload.tobytes()


def make_label(prefix: str, index: int, identifier: str | None = None, laterality: str | None = None) -> str:
    parts = [prefix, str(index + 1)]
    if identifier:
        parts.append(str(identifier))
    if laterality:
        parts.append(str(laterality))
    return " | ".join(parts)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalized_lookup_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def coerce_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").replace("\x00", "").strip()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return str(value.item()).replace("\x00", "").strip()
    if isinstance(value, (str, int, float, bool)):
        return str(value).replace("\x00", "").strip()
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, np.integer)) for item in value):
            try:
                return bytes(int(item) for item in value).decode("utf-8", errors="replace").replace("\x00", "").strip()
            except (TypeError, ValueError):
                pass
        parts = [coerce_text_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return ""


def find_text_in_metadata(value: Any, candidate_keys: set[str], depth: int = 0) -> str:
    if value is None or depth > 4:
        return ""
    if isinstance(value, dict):
        for key, item in value.items():
            if normalized_lookup_key(key) in candidate_keys:
                text = coerce_text_value(item)
                if text:
                    return text
        for item in value.values():
            text = find_text_in_metadata(item, candidate_keys, depth + 1)
            if text:
                return text
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = find_text_in_metadata(item, candidate_keys, depth + 1)
            if text:
                return text
        return ""
    object_values = getattr(value, "__dict__", None)
    if isinstance(object_values, dict):
        return find_text_in_metadata(object_values, candidate_keys, depth + 1)
    return ""


def find_text_in_attributes(value: Any, candidate_attributes: tuple[str, ...]) -> str:
    if value is None:
        return ""
    for attribute in candidate_attributes:
        text = coerce_text_value(getattr(value, attribute, None))
        if text:
            return text
    return ""


def join_text_values(*values: Any) -> str:
    parts = [coerce_text_value(value) for value in values]
    return " ".join(part for part in parts if part).strip()


def partition_bscan_metadata(
    bscan_data: list[dict[str, Any]],
    slice_counts: list[int],
) -> list[list[dict[str, Any]]]:
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


def points_to_segments(
    points: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(points[index], points[index + 1]) for index in range(0, len(points), 2)]


def clamp_bounds_to_shape(
    bounds: tuple[float, float, float, float],
    fundus_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[float, float, float, float] | None:
    height, width = fundus_shape[:2]
    left, top, right, bottom = [float(value) for value in bounds]
    left = max(0.0, min(left, float(width - 1)))
    right = max(0.0, min(right, float(width - 1)))
    top = max(0.0, min(top, float(height - 1)))
    bottom = max(0.0, min(bottom, float(height - 1)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_horizontal_segments_from_bounds(
    num_slices: int,
    bounds: tuple[float, float, float, float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if num_slices <= 0:
        return []
    left, top, right, bottom = bounds
    if num_slices == 1:
        y_pos = (top + bottom) / 2.0
        return [((left, y_pos), (right, y_pos))]

    segments = []
    for index in range(num_slices):
        ratio = index / (num_slices - 1)
        y_pos = top + ratio * (bottom - top)
        segments.append(((left, y_pos), (right, y_pos)))
    return segments


def build_repeated_line_segments(
    num_slices: int,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if num_slices <= 0:
        return []
    return [(start, end) for _ in range(num_slices)]


def build_parallel_segments(
    num_slices: int,
    fundus_shape: tuple[int, int] | tuple[int, int, int],
    *,
    content_bounds: tuple[float, float, float, float] | None = None,
    orientation: str = "vertical",
    square_region: bool = False,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    height, width = fundus_shape[:2]
    if num_slices <= 0:
        return []

    if content_bounds is None:
        left = 0.0
        top = 0.0
        bounds_width = float(width)
        bounds_height = float(height)
    else:
        left, top, bounds_width, bounds_height = content_bounds

    if square_region:
        side = min(bounds_width, bounds_height) * 0.58
        center_x = left + bounds_width / 2.0
        center_y = top + bounds_height / 2.0
        x_min = center_x - side / 2.0
        x_max = center_x + side / 2.0
        y_min = center_y - side / 2.0
        y_max = center_y + side / 2.0
    elif content_bounds is None:
        x_min = width * 0.24
        x_max = width * 0.76
        y_min = height * 0.24
        y_max = height * 0.76
    else:
        margin_x = bounds_width * 0.12
        margin_y = bounds_height * 0.12
        x_min = left + margin_x
        x_max = left + bounds_width - margin_x
        y_min = top + margin_y
        y_max = top + bounds_height - margin_y

    x_min = max(0.0, min(float(width - 1), float(x_min)))
    x_max = max(x_min, min(float(width - 1), float(x_max)))
    y_min = max(0.0, min(float(height - 1), float(y_min)))
    y_max = max(y_min, min(float(height - 1), float(y_max)))

    if num_slices == 1:
        if orientation == "horizontal":
            y_pos = (y_min + y_max) / 2.0
            return [((x_min, y_pos), (x_max, y_pos))]
        x_pos = (x_min + x_max) / 2.0
        return [((x_pos, y_min), (x_pos, y_max))]

    segments = []
    for index in range(num_slices):
        ratio = index / (num_slices - 1)
        if orientation == "horizontal":
            y_pos = y_min + ratio * (y_max - y_min)
            segments.append(((x_min, y_pos), (x_max, y_pos)))
        else:
            x_pos = x_min + ratio * (x_max - x_min)
            segments.append(((x_pos, y_min), (x_pos, y_max)))
    return segments


def estimate_fundus_content_bounds(
    fundus_image: np.ndarray,
) -> tuple[float, float, float, float] | None:
    image = np.asarray(fundus_image)
    if image.ndim >= 3:
        gray = image[..., :3].max(axis=2)
    else:
        gray = image

    if gray.size == 0:
        return None

    max_value = float(np.max(gray))
    if not np.isfinite(max_value) or max_value <= 0:
        return None

    threshold = max(6.0, max_value * 0.08)
    mask = gray > threshold
    if not np.any(mask):
        return None

    rows = np.where(mask.sum(axis=1) >= max(8, int(gray.shape[1] * 0.03)))[0]
    cols = np.where(mask.sum(axis=0) >= max(8, int(gray.shape[0] * 0.03)))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    top = float(rows[0])
    bottom = float(rows[-1])
    left = float(cols[0])
    right = float(cols[-1])
    bounds_width = max(1.0, right - left)
    bounds_height = max(1.0, bottom - top)
    return left, top, bounds_width, bounds_height


def compute_scan_bounds(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float, float, float] | None:
    if not segments:
        return None
    xs = [point[0] for segment in segments for point in segment]
    ys = [point[1] for segment in segments for point in segment]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


@dataclass
class OverlayInfo:
    matched_fundus_index: int
    matched_fundus_label: str
    fundus_match_mode: str
    overlay_mode: str
    projection_mode: str
    localizer_mode: str
    warning: str
    scan_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    bounds: tuple[float, float, float, float] | None


@dataclass
class LoadedDataset:
    source_path: str
    source_kind: str
    recent_path: str
    reader_name: str
    volumes: list[Any]
    fundus_images: list[Any]
    overlay_infos: list[OverlayInfo]


class ViewerState:
    def __init__(self, img_rows: int, img_cols: int, img_interlaced: bool) -> None:
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_interlaced = img_interlaced
        self.dataset: LoadedDataset | None = None
        self.lock = threading.Lock()
        self.uploaded_file_path: Path | None = None

    def load(self, filepath: str, vendor_mode: str = "auto") -> dict[str, Any]:
        vendor_mode = normalize_vendor_mode(vendor_mode)
        path = self._resolve_path(filepath)
        payload = self._load_path(
            path,
            source_path=str(path),
            source_kind="path",
            recent_path=str(path),
            vendor_mode=vendor_mode,
        )
        self._cleanup_uploaded_file()
        return payload

    def load_uploaded_file(self, filename: str, fileobj: Any, vendor_mode: str = "auto") -> dict[str, Any]:
        vendor_mode = normalize_vendor_mode(vendor_mode)
        upload_name = Path(filename or "").name.strip()
        if not upload_name:
            raise ValueError("Missing uploaded filename.")

        self._validate_upload_request(upload_name, vendor_mode)
        upload_path = Path(upload_name)
        suffix = upload_path.suffix

        with tempfile.NamedTemporaryFile(delete=False, prefix="oct-web-viewer-", suffix=suffix) as handle:
            shutil.copyfileobj(fileobj, handle)
            temp_path = Path(handle.name)

        try:
            if temp_path.stat().st_size <= 0:
                raise ValueError("Uploaded file is empty.")

            payload = self._load_path(
                temp_path,
                source_path=upload_name,
                source_kind="upload",
                recent_path="",
                vendor_mode=vendor_mode,
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        previous_upload = self.uploaded_file_path
        self.uploaded_file_path = temp_path
        if previous_upload and previous_upload != temp_path:
            previous_upload.unlink(missing_ok=True)
        return payload

    def close(self) -> None:
        self._cleanup_uploaded_file()

    def _load_path(
        self,
        path: Path,
        *,
        source_path: str,
        source_kind: str,
        recent_path: str,
        vendor_mode: str = "auto",
    ) -> dict[str, Any]:
        normalized_mode = normalize_vendor_mode(vendor_mode)
        self._validate_load_request(path, normalized_mode)

        source_meta = {
            "source_path": source_path,
            "source_kind": source_kind,
            "recent_path": recent_path,
        }
        dataset = self._load_dataset_from_path(path, normalized_mode, source_meta)
        return self._set_dataset_and_build_payload(dataset)

    def pick_and_load(self, vendor_mode: str = "auto") -> dict[str, Any]:
        normalized_mode = normalize_vendor_mode(vendor_mode)
        filepath = self.pick_file(normalized_mode)
        if not filepath:
            payload = self.build_state_payload()
            payload["cancelled"] = True
            return payload
        payload = self.load(filepath, vendor_mode=normalized_mode)
        payload["cancelled"] = False
        return payload

    def pick_file(self, vendor_mode: str = "auto") -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise RuntimeError("Native file dialog is not available in this Python environment.") from exc

        normalized_mode = normalize_vendor_mode(vendor_mode)
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if normalized_mode in {"shiwei", "tupai"}:
                selected = filedialog.askdirectory(
                    title="Select an OCT dataset directory",
                    mustexist=True,
                )
            else:
                selected = filedialog.askopenfilename(
                    title="Select an OCT file",
                    filetypes=VENDOR_MODE_FILE_DIALOG_TYPES.get(normalized_mode, FILE_DIALOG_TYPES),
                )
        finally:
            root.destroy()
        return selected or ""

    def build_state_payload(self) -> dict[str, Any]:
        with self.lock:
            dataset = self.dataset

        if dataset is None:
            return {
                "loaded": False,
                "sourcePath": "",
                "reader": "",
                "volumes": [],
                "fundusImages": [],
                "supportedExtensions": list(SUPPORTED_EXTENSIONS),
            }

        fundus_descriptions = [
            self._describe_fundus(image, index) for index, image in enumerate(dataset.fundus_images)
        ]
        return {
            "loaded": True,
            "sourcePath": dataset.source_path,
            "sourceKind": dataset.source_kind,
            "recentPath": dataset.recent_path,
            "reader": dataset.reader_name,
            "volumes": [
                self._describe_volume(
                    volume,
                    index,
                    dataset.fundus_images,
                    dataset.overlay_infos[index] if index < len(dataset.overlay_infos) else None,
                )
                for index, volume in enumerate(dataset.volumes)
            ],
            "fundusImages": fundus_descriptions,
            "supportedExtensions": list(SUPPORTED_EXTENSIONS),
        }

    def get_slice_png(
        self,
        volume_index: int,
        slice_index: int,
        contrast_percent: int,
        brightness_offset: int,
    ) -> bytes:
        volume = self._get_volume(volume_index)
        image = self._get_slice(volume, slice_index)
        image = normalize_to_uint8(image)
        image = apply_window(image, contrast_percent, brightness_offset)
        return encode_png(image)

    def get_projection_png(self, volume_index: int) -> bytes:
        volume = self._get_volume(volume_index)
        projection = self._get_projection_image(volume)
        return encode_png(normalize_to_uint8(np.asarray(projection)))

    def get_fundus_png(self, fundus_index: int) -> bytes:
        with self.lock:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("No file is loaded.")
        if not 0 <= fundus_index < len(dataset.fundus_images):
            raise IndexError(f"Fundus index out of range: {fundus_index}")
        image = self._prepare_display_image(np.asarray(dataset.fundus_images[fundus_index].image))
        return encode_png(normalize_to_uint8(np.asarray(image)))

    def get_volume_fundus_view_png(self, volume_index: int) -> bytes:
        with self.lock:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("No file is loaded.")
        if not 0 <= volume_index < len(dataset.volumes):
            raise IndexError(f"Volume index out of range: {volume_index}")

        overlay_info = dataset.overlay_infos[volume_index] if volume_index < len(dataset.overlay_infos) else None
        matched_index = overlay_info.matched_fundus_index if overlay_info is not None else -1
        if 0 <= matched_index < len(dataset.fundus_images):
            image = self._prepare_display_image(np.asarray(dataset.fundus_images[matched_index].image))
        else:
            image = self._get_projection_image(dataset.volumes[volume_index])
        return encode_png(normalize_to_uint8(np.asarray(image)))

    def get_contours(self, volume_index: int, slice_index: int) -> dict[str, Any]:
        volume = self._get_volume(volume_index)
        contours = getattr(volume, "contours", None) or {}
        payload: dict[str, list[float | None]] = {}
        for name, values in contours.items():
            if slice_index >= len(values):
                continue
            contour = values[slice_index]
            if contour is None:
                continue
            array = np.asarray(contour, dtype=np.float32)
            if array.size == 0 or np.isnan(array).all():
                continue
            payload[name] = [None if not np.isfinite(item) else float(item) for item in array.tolist()]
        return {"sliceIndex": slice_index, "contours": payload}

    def _create_reader(self, path: Path) -> Any:
        if path.is_dir():
            raise ValueError(f"Unsupported directory input: {path}")
        suffix = path.suffix
        suffix_lower = suffix.lower()
        if suffix_lower == ".fds":
            return FDS(path)
        if suffix_lower == ".fda":
            return FDA(path)
        if suffix_lower == ".e2e":
            return E2E(path)
        if suffix == ".OCT":
            return BOCT(path)
        if suffix_lower == ".oct":
            return POCT(path)
        if suffix_lower == ".img":
            return IMG(path)
        if suffix_lower in {".dcm", ".dicom"}:
            return Dicom(path)
        raise ValueError(f"Unsupported file type: {suffix or path.name}")

    def _resolve_path(self, filepath: str) -> Path:
        path = Path(filepath.strip()).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def _validate_vendor_selection(self, path: Path, vendor_mode: str) -> None:
        normalized_mode = normalize_vendor_mode(vendor_mode)
        if normalized_mode == "auto":
            return
        if normalized_mode == "shiwei":
            if path.is_dir():
                return
            if is_supported_suffix_for_vendor(path, normalized_mode):
                return
            raise ValueError(build_vendor_validation_error(path, normalized_mode))
        if normalized_mode == "tupai":
            if path.is_dir():
                return
            if is_supported_suffix_for_vendor(path, normalized_mode):
                return
            raise ValueError(build_vendor_validation_error(path, normalized_mode))
        if path.is_dir():
            label = VENDOR_MODE_LABELS.get(normalized_mode, normalized_mode)
            raise ValueError(f"Selected vendor mode '{label}' requires choosing a file, not a directory.")
        if not is_supported_suffix_for_vendor(path, normalized_mode):
            raise ValueError(build_vendor_validation_error(path, normalized_mode))

    def _validate_load_request(self, path: Path, vendor_mode: str) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self._validate_vendor_selection(path, vendor_mode)

    def _validate_upload_request(self, upload_name: str, vendor_mode: str) -> None:
        if vendor_mode == "shiwei":
            raise ValueError(SHIWEI_UPLOAD_HINT)
        if vendor_mode == "tupai":
            raise ValueError(TUPAI_UPLOAD_HINT)

        upload_path = Path(upload_name)
        suffix = upload_path.suffix
        normalized_suffix = suffix.lower()
        if normalized_suffix not in NORMALIZED_SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or upload_name}")
        self._validate_vendor_selection(upload_path, vendor_mode)

    def _load_dataset_from_path(
        self,
        path: Path,
        vendor_mode: str,
        source_meta: dict[str, str],
    ) -> LoadedDataset:
        tupai_dataset = self._try_load_tupai_dataset(path, vendor_mode=vendor_mode)
        if tupai_dataset is not None:
            return self._build_tupai_loaded_dataset(tupai_dataset, source_meta)
        shiwei_dataset = self._try_load_shiwei_dataset(path, vendor_mode=vendor_mode)
        if shiwei_dataset is not None:
            return self._build_shiwei_loaded_dataset(shiwei_dataset, source_meta)
        zeiss_dataset = self._try_load_zeiss_dataset(path, vendor_mode=vendor_mode)
        if zeiss_dataset is not None:
            return self._build_zeiss_loaded_dataset(zeiss_dataset, source_meta)
        return self._build_reader_loaded_dataset(path, source_meta)

    def _build_tupai_loaded_dataset(
        self,
        tupai_dataset: dict[str, Any],
        source_meta: dict[str, str],
    ) -> LoadedDataset:
        dataset_dir = str(tupai_dataset.get("dataset_dir") or source_meta["source_path"])
        return LoadedDataset(
            source_path=dataset_dir,
            source_kind=source_meta["source_kind"],
            recent_path=dataset_dir,
            reader_name="TupaiDicomDataset",
            volumes=[tupai_dataset["volume"]],
            fundus_images=[tupai_dataset["fundus"]],
            overlay_infos=self._build_tupai_overlay_infos(tupai_dataset),
        )

    def _build_shiwei_loaded_dataset(
        self,
        shiwei_dataset: dict[str, Any],
        source_meta: dict[str, str],
    ) -> LoadedDataset:
        dataset_dir = str(shiwei_dataset.get("input_dir") or source_meta["source_path"])
        return LoadedDataset(
            source_path=dataset_dir,
            source_kind=source_meta["source_kind"],
            recent_path=dataset_dir,
            reader_name="ShiweiDicomDataset",
            volumes=[shiwei_dataset["volume"]],
            fundus_images=[shiwei_dataset["fundus"]],
            overlay_infos=self._build_shiwei_overlay_infos(shiwei_dataset),
        )

    def _build_zeiss_loaded_dataset(
        self,
        zeiss_dataset: dict[str, Any],
        source_meta: dict[str, str],
    ) -> LoadedDataset:
        exam_dirs = zeiss_dataset.get("exam_dirs") or []
        dataset_path = (
            str(zeiss_dataset.get("input_path") or source_meta["source_path"])
            if len(exam_dirs) != 1
            else str(exam_dirs[0])
        )
        return LoadedDataset(
            source_path=dataset_path,
            source_kind=source_meta["source_kind"],
            recent_path=dataset_path,
            reader_name="ZeissDicomDataset",
            volumes=list(zeiss_dataset.get("volumes") or []),
            fundus_images=list(zeiss_dataset.get("fundus_images") or []),
            overlay_infos=self._build_zeiss_overlay_infos(zeiss_dataset),
        )

    def _build_reader_loaded_dataset(
        self,
        path: Path,
        source_meta: dict[str, str],
    ) -> LoadedDataset:
        reader = self._create_reader(path)
        volumes = as_list(self._read_oct_volumes(reader))
        if not volumes:
            raise ValueError("No OCT volumes were extracted from this file.")

        fundus_images = as_list(self._safe_read_fundus(reader))
        overlay_infos = self._build_overlay_infos(reader, volumes, fundus_images)
        return LoadedDataset(
            source_path=source_meta["source_path"],
            source_kind=source_meta["source_kind"],
            recent_path=source_meta["recent_path"],
            reader_name=reader.__class__.__name__,
            volumes=volumes,
            fundus_images=fundus_images,
            overlay_infos=overlay_infos,
        )

    def _set_dataset_and_build_payload(self, dataset: LoadedDataset) -> dict[str, Any]:
        with self.lock:
            self.dataset = dataset
        return self.build_state_payload()

    def _cleanup_uploaded_file(self) -> None:
        if self.uploaded_file_path:
            self.uploaded_file_path.unlink(missing_ok=True)
            self.uploaded_file_path = None

    def _read_oct_volumes(self, reader: Any) -> Any:
        if isinstance(reader, IMG):
            return reader.read_oct_volume(
                rows=self.img_rows,
                cols=self.img_cols,
                interlaced=self.img_interlaced,
            )
        return reader.read_oct_volume()

    def _safe_read_fundus(self, reader: Any) -> Any:
        if not hasattr(reader, "read_fundus_image"):
            return []
        try:
            return reader.read_fundus_image()
        except Exception:
            return []

    def _build_overlay_infos(
        self,
        reader: Any,
        volumes: list[Any],
        fundus_images: list[Any],
    ) -> list[OverlayInfo]:
        if isinstance(reader, E2E):
            return self._build_e2e_overlay_infos(reader, volumes, fundus_images)
        if isinstance(reader, (FDA, FDS)):
            return self._build_parallel_overlay_infos(volumes, fundus_images, embedded_fundus=True)
        return self._build_parallel_overlay_infos(volumes, fundus_images, embedded_fundus=False)

    def _try_load_zeiss_dataset(
        self,
        path: Path,
        *,
        vendor_mode: str = "auto",
    ) -> dict[str, Any] | None:
        normalized_mode = normalize_vendor_mode(vendor_mode)
        if normalized_mode != "auto":
            return None
        if path.is_file() and path.suffix.lower() == ".img":
            return None

        exam_dirs = resolve_zeiss_exam_dirs(path)
        if exam_dirs:
            return load_zeiss_oct_dataset(path)
        return None

    def _try_load_shiwei_dataset(
        self,
        path: Path,
        *,
        vendor_mode: str = "auto",
    ) -> dict[str, Any] | None:
        normalized_mode = normalize_vendor_mode(vendor_mode)
        if normalized_mode not in {"auto", "shiwei"}:
            return None

        dataset_dir = self._resolve_shiwei_dataset_dir(path)
        if dataset_dir is None:
            if normalized_mode == "shiwei":
                raise ValueError(
                    "当前视微模式需要输入视微数据目录、目录中的任一配套 DICOM 文件路径，"
                    "或可唯一定位到该数据集的上级目录。"
                )
            return None
        return load_shiwei_oct_dataset(str(dataset_dir))

    def _resolve_shiwei_dataset_dir(self, path: Path) -> Path | None:
        return resolve_shiwei_input_dir(path)

    def _try_load_tupai_dataset(
        self,
        path: Path,
        *,
        vendor_mode: str = "auto",
    ) -> dict[str, Any] | None:
        normalized_mode = normalize_vendor_mode(vendor_mode)
        if normalized_mode not in {"auto", "tupai"}:
            return None

        dataset_dir = self._resolve_tupai_dataset_dir(path)
        if dataset_dir is None:
            if normalized_mode == "tupai":
                raise ValueError(
                    "当前 Tupai 模式需要输入包含 OCT.dcm 和 Fundus.dcm 的目录、"
                    "其中任一文件路径，或可唯一定位到该数据集的上级目录。"
                )
            return None
        return load_tupai_oct_dataset(dataset_dir)

    def _resolve_tupai_dataset_dir(self, path: Path) -> Path | None:
        return resolve_tupai_input_dir(path)

    def _build_shiwei_overlay_infos(self, dataset: dict[str, Any]) -> list[OverlayInfo]:
        fundus = dataset.get("fundus")
        raw_segments = [
            (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            )
            for start, end in dataset.get("segments", [])
        ]
        bounds = compute_scan_bounds(raw_segments)
        warning = ""
        if not raw_segments:
            warning = "Shiwei frame location metadata unavailable; overlay falls back to fundus only."

        return [
            OverlayInfo(
                matched_fundus_index=0 if isinstance(fundus, FundusImageWithMetaData) else -1,
                matched_fundus_label=getattr(fundus, "image_id", None) or "Shiwei fundus",
                fundus_match_mode="shiwei-directory",
                overlay_mode="shiwei-metadata",
                projection_mode="shiwei-metadata",
                localizer_mode="ophthalmic-frame-location-sequence",
                warning=warning,
                scan_segments=raw_segments,
                bounds=bounds,
            )
        ]

    def _build_zeiss_overlay_infos(self, dataset: dict[str, Any]) -> list[OverlayInfo]:
        overlays = dataset.get("overlays") or []
        overlay_infos: list[OverlayInfo] = []
        for overlay in overlays:
            raw_segments = [
                (
                    (float(start[0]), float(start[1])),
                    (float(end[0]), float(end[1])),
                )
                for start, end in overlay.get("scan_segments", [])
            ]
            raw_bounds = overlay.get("bounds")
            bounds = (
                tuple(float(value) for value in raw_bounds)
                if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 4
                else compute_scan_bounds(raw_segments)
            )
            overlay_infos.append(
                OverlayInfo(
                    matched_fundus_index=int(overlay.get("matched_fundus_index", -1)),
                    matched_fundus_label=str(overlay.get("matched_fundus_label") or "Zeiss fundus"),
                    fundus_match_mode=str(overlay.get("fundus_match_mode") or "zeiss-dicom"),
                    overlay_mode=str(overlay.get("overlay_mode") or "zeiss-dicom"),
                    projection_mode=str(overlay.get("projection_mode") or "zeiss-dicom"),
                    localizer_mode=str(overlay.get("localizer_mode") or "zeiss-dicom"),
                    warning=str(overlay.get("warning") or ""),
                    scan_segments=raw_segments,
                    bounds=bounds,
                )
            )
        return overlay_infos

    def _build_tupai_overlay_infos(self, dataset: dict[str, Any]) -> list[OverlayInfo]:
        fundus = dataset.get("fundus")
        full_segments = [
            (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            )
            for start, end in dataset.get("segments", [])
        ]
        band_segments = [
            (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            )
            for start, end in dataset.get("segment_coordinates", [])
        ]
        scan_segments = full_segments or band_segments
        bounds = compute_scan_bounds(scan_segments)
        warning = ""
        if not scan_segments:
            warning = "Tupai frame location metadata unavailable; overlay falls back to fundus only."

        return [
            OverlayInfo(
                matched_fundus_index=0 if isinstance(fundus, FundusImageWithMetaData) else -1,
                matched_fundus_label=getattr(fundus, "image_id", None) or "Tupai fundus",
                fundus_match_mode="tupai-directory",
                overlay_mode="tupai-metadata",
                projection_mode=str(dataset.get("coordinate_mode") or "tupai"),
                localizer_mode=str(dataset.get("segment_mode") or "ophthalmic-frame-location-sequence"),
                warning=warning,
                scan_segments=scan_segments,
                bounds=bounds,
            )
        ]

    def _build_parallel_overlay_infos(
        self,
        volumes: list[Any],
        fundus_images: list[Any],
        *,
        embedded_fundus: bool,
    ) -> list[OverlayInfo]:
        overlay_infos: list[OverlayInfo] = []
        for index, volume in enumerate(volumes):
            matched_index = self._match_fundus_index(volume, fundus_images)
            if matched_index >= 0 and matched_index < len(fundus_images):
                fundus_image = self._prepare_display_image(np.asarray(fundus_images[matched_index].image))
                fundus_label = getattr(fundus_images[matched_index], "image_id", None) or f"Fundus {matched_index + 1}"
                fundus_match_mode = "embedded-fundus" if embedded_fundus else "matched-fundus"
            else:
                fundus_image = self._get_projection_image(volume)
                fundus_label = "projection-fallback"
                fundus_match_mode = "projection-fallback"
            overlay_mode = "parallel-approximation"
            projection_mode = "square-raster-default" if embedded_fundus else "parallel-default"
            warning = "" if embedded_fundus else "Using approximate parallel scan lines."

            scan_segments = None
            if embedded_fundus:
                scan_segments, projection_mode = self._build_fda_metadata_segments(
                    volume=volume,
                    fundus_shape=fundus_image.shape,
                )
                if scan_segments:
                    overlay_mode = "fda-metadata"
                else:
                    content_bounds = estimate_fundus_content_bounds(fundus_image)
                    scan_segments = build_parallel_segments(
                        self._slice_count(volume),
                        fundus_image.shape,
                        content_bounds=content_bounds,
                        orientation="horizontal",
                        square_region=True,
                    )
                    warning = "FDA localization metadata unavailable; using approximate scan area."
                    projection_mode = "square-raster-fallback"
            else:
                content_bounds = estimate_fundus_content_bounds(fundus_image)
                scan_segments = build_parallel_segments(
                    self._slice_count(volume),
                    fundus_image.shape,
                    content_bounds=content_bounds,
                    orientation="vertical",
                    square_region=False,
                )

            overlay_infos.append(
                OverlayInfo(
                    matched_fundus_index=matched_index,
                    matched_fundus_label=str(fundus_label),
                    fundus_match_mode=fundus_match_mode,
                    overlay_mode=overlay_mode,
                    projection_mode=projection_mode,
                    localizer_mode="not-applicable",
                    warning=warning,
                    scan_segments=scan_segments,
                    bounds=compute_scan_bounds(scan_segments),
                )
            )
        return overlay_infos

    def _build_e2e_overlay_infos(
        self,
        reader: Any,
        volumes: list[Any],
        fundus_images: list[Any],
    ) -> list[OverlayInfo]:
        try:
            metadata = reader.read_all_metadata()
        except Exception:
            metadata = {}

        slice_counts = [self._slice_count(volume) for volume in volumes]
        bscan_groups = partition_bscan_metadata(metadata.get("bscan_data", []), slice_counts)
        localizers = metadata.get("localizer", [])
        total_volumes = len(volumes)

        overlay_infos: list[OverlayInfo] = []
        for index, volume in enumerate(volumes):
            matched_fundus, matched_index, fundus_match_mode = self._select_matching_fundus_for_e2e(
                volume,
                fundus_images,
                index,
            )
            if matched_fundus is not None:
                fundus_image = self._prepare_display_image(np.asarray(matched_fundus.image))
                fundus_label = getattr(matched_fundus, "image_id", None) or f"Fundus {matched_index + 1}"
            else:
                fundus_image = self._get_projection_image(volume)
                fundus_label = "projection-fallback"

            bscan_group = bscan_groups[index] if index < len(bscan_groups) else []
            localizer_entry, localizer_mode = self._choose_localizer(localizers, index, total_volumes)
            overlay = self._project_e2e_overlay(
                group=bscan_group,
                localizer_entry=localizer_entry,
                localizer_mode=localizer_mode,
                fundus_shape=fundus_image.shape,
            )
            if not bscan_group:
                overlay.scan_segments = build_parallel_segments(self._slice_count(volume), fundus_image.shape)
                overlay.bounds = compute_scan_bounds(overlay.scan_segments)

            overlay_infos.append(
                OverlayInfo(
                    matched_fundus_index=matched_index,
                    matched_fundus_label=str(fundus_label),
                    fundus_match_mode=fundus_match_mode,
                    overlay_mode=overlay.overlay_mode,
                    projection_mode=overlay.projection_mode,
                    localizer_mode=overlay.localizer_mode,
                    warning=overlay.warning,
                    scan_segments=overlay.scan_segments,
                    bounds=overlay.bounds,
                )
            )
        return overlay_infos

    def _get_volume(self, volume_index: int) -> Any:
        with self.lock:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("No file is loaded.")
        if not 0 <= volume_index < len(dataset.volumes):
            raise IndexError(f"Volume index out of range: {volume_index}")
        return dataset.volumes[volume_index]

    def _extract_patient_name(self, volume: Any, *metadata_sources: Any) -> str:
        patient_name = find_text_in_attributes(
            volume,
            (
                "patient_name",
                "patientName",
                "patient_full_name",
                "patientFullName",
                "patient_display_name",
                "patientDisplayName",
            ),
        )
        if patient_name:
            return patient_name

        combined_name = join_text_values(
            find_text_in_attributes(
                volume,
                (
                    "patient_first_name",
                    "patientFirstName",
                    "first_name",
                    "firstName",
                    "given_name",
                    "givenName",
                    "forename",
                    "name",
                ),
            ),
            find_text_in_attributes(volume, ("mid_name", "midName", "middle_name", "middleName")),
            find_text_in_attributes(
                volume,
                (
                    "patient_last_name",
                    "patientLastName",
                    "last_name",
                    "lastName",
                    "family_name",
                    "familyName",
                    "surname",
                ),
            ),
        )
        if combined_name:
            return combined_name

        patient_name_keys = {
            "patientname",
            "patientsname",
            "patientfullname",
            "patientdisplayname",
            "subjectname",
            "subjectfullname",
            "patientpersonname",
            "name",
        }
        first_name_keys = {"patientfirstname", "firstname", "givenname", "forename"}
        middle_name_keys = {"patientmiddlename", "middlename", "midname"}
        last_name_keys = {"patientlastname", "lastname", "familyname", "surname"}

        for source in metadata_sources:
            patient_name = find_text_in_metadata(source, patient_name_keys)
            if patient_name:
                return patient_name

            combined_name = join_text_values(
                find_text_in_metadata(source, first_name_keys),
                find_text_in_metadata(source, middle_name_keys),
                find_text_in_metadata(source, last_name_keys),
            )
            if combined_name:
                return combined_name
        return ""

    def _extract_patient_id(self, volume: Any, *metadata_sources: Any) -> str:
        patient_id = find_text_in_attributes(volume, ("patient_id", "patientId", "subject_id", "subjectId"))
        if patient_id:
            return patient_id

        patient_id_keys = {
            "patientid",
            "patientidentifier",
            "patientnumber",
            "subjectid",
            "subjectidentifier",
            "subjectnumber",
        }
        for source in metadata_sources:
            patient_id = find_text_in_metadata(source, patient_id_keys)
            if patient_id:
                return patient_id
        return ""

    def _extract_acquisition_date(self, volume: Any, *metadata_sources: Any) -> str:
        acquisition_date = coerce_text_value(getattr(volume, "acquisition_date", None))
        if acquisition_date:
            return acquisition_date

        acquisition_date = find_text_in_attributes(
            volume,
            ("exam_date", "examDate", "study_date", "studyDate", "scan_date", "scanDate"),
        )
        if acquisition_date:
            return acquisition_date

        date_keys = {
            "acquisitiondate",
            "examdate",
            "studydate",
            "seriesdate",
            "scandate",
            "capturedate",
            "contentdate",
        }
        for source in metadata_sources:
            acquisition_date = find_text_in_metadata(source, date_keys)
            if acquisition_date:
                return acquisition_date
        return ""

    def _extract_device_name(self, volume: Any, *metadata_sources: Any) -> str:
        manufacturer = find_text_in_attributes(
            volume,
            (
                "manufacturer",
                "device_manufacturer",
                "deviceManufacturer",
                "scanner_manufacturer",
                "scannerManufacturer",
            ),
        )
        model = find_text_in_attributes(
            volume,
            (
                "model",
                "model_name",
                "modelName",
                "device_model",
                "deviceModel",
                "scanner_model",
                "scannerModel",
            ),
        )
        device_name = join_text_values(manufacturer, model)
        if device_name:
            return device_name

        device_name = find_text_in_attributes(
            volume,
            (
                "device",
                "device_name",
                "deviceName",
                "instrument",
                "instrument_name",
                "instrumentName",
                "scanner",
                "scanner_name",
                "scannerName",
                "system",
                "system_name",
                "systemName",
            ),
        )
        if device_name:
            return device_name

        manufacturer_keys = {
            "manufacturer",
            "devicemanufacturer",
            "scannermanufacturer",
            "equipmentmanufacturer",
        }
        model_keys = {
            "model",
            "modelname",
            "manufacturermodelname",
            "devicemodel",
            "devicemodelname",
            "scannermodel",
            "scannermodelname",
            "equipmentmodel",
            "equipmentmodelname",
        }
        device_keys = {
            "device",
            "devicename",
            "instrument",
            "instrumentname",
            "scanner",
            "scannername",
            "machine",
            "machinename",
            "system",
            "systemname",
        }

        for source in metadata_sources:
            device_name = join_text_values(
                find_text_in_metadata(source, manufacturer_keys),
                find_text_in_metadata(source, model_keys),
            )
            if device_name:
                return device_name

            device_name = find_text_in_metadata(source, device_keys)
            if device_name:
                return device_name
        return ""

    def _describe_volume(
        self,
        volume: Any,
        index: int,
        fundus_images: list[Any],
        overlay_info: OverlayInfo | None,
    ) -> dict[str, Any]:
        slice_count = self._slice_count(volume)
        first_slice = self._get_slice(volume, 0)
        height, width = first_slice.shape[:2]
        measurement_scale = self._get_bscan_measurement_scale(
            pixel_spacing=getattr(volume, "pixel_spacing", None),
        )
        display_width_units, display_height_units = self._get_bscan_display_units(
            width=width,
            height=height,
            pixel_spacing=getattr(volume, "pixel_spacing", None),
        )
        contour_names = sorted((getattr(volume, "contours", None) or {}).keys())
        volume_id = getattr(volume, "volume_id", None)
        laterality = getattr(volume, "laterality", None)
        metadata = getattr(volume, "metadata", None)
        header = getattr(volume, "header", None)
        oct_header = getattr(volume, "oct_header", None)
        patient_name = self._extract_patient_name(volume, metadata, header, oct_header)
        patient_id = self._extract_patient_id(volume, metadata, header, oct_header)
        acquisition_date = self._extract_acquisition_date(volume, metadata, header, oct_header)
        device = self._extract_device_name(volume, metadata, header, oct_header)
        matched_fundus_index = (
            overlay_info.matched_fundus_index
            if overlay_info is not None
            else self._match_fundus_index(volume, fundus_images)
        )
        return {
            "index": index,
            "label": make_label("Volume", index, volume_id, laterality),
            "volumeId": volume_id or "",
            "laterality": laterality or "",
            "sliceCount": slice_count,
            "width": width,
            "height": height,
            "pixelSpacing": to_jsonable(getattr(volume, "pixel_spacing", None)),
            "measurementScale": measurement_scale,
            "displayWidthUnits": display_width_units,
            "displayHeightUnits": display_height_units,
            "displayAspectRatio": (
                (display_width_units / display_height_units)
                if display_width_units and display_height_units
                else (width / height if height else None)
            ),
            "patientName": patient_name,
            "patientId": patient_id or "",
            "acquisitionDate": acquisition_date,
            "device": device,
            "contourNames": contour_names,
            "contourCount": len(contour_names),
            "matchedFundusIndex": matched_fundus_index,
            "matchedFundusLabel": overlay_info.matched_fundus_label if overlay_info is not None else "",
            "fundusMatchMode": overlay_info.fundus_match_mode if overlay_info is not None else "",
            "overlayMode": overlay_info.overlay_mode if overlay_info is not None else "",
            "projectionMode": overlay_info.projection_mode if overlay_info is not None else "",
            "localizerMode": overlay_info.localizer_mode if overlay_info is not None else "",
            "overlayWarning": overlay_info.warning if overlay_info is not None else "",
            "overlayBounds": (
                [float(value) for value in overlay_info.bounds]
                if overlay_info is not None and overlay_info.bounds is not None
                else None
            ),
            "overlaySegments": self._serialize_segments(overlay_info.scan_segments if overlay_info is not None else []),
            "metadataText": self._format_metadata_text(
                label=f"Volume {index + 1}",
                details={
                    "volume_id": volume_id,
                    "laterality": laterality,
                    "slice_count": slice_count,
                    "width": width,
                    "height": height,
                    "pixel_spacing": getattr(volume, "pixel_spacing", None),
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "acquisition_date": acquisition_date,
                    "device": device,
                    "metadata": metadata,
                    "header": header,
                    "oct_header": oct_header,
                },
            ),
        }

    def _describe_fundus(self, image: Any, index: int) -> dict[str, Any]:
        array = self._prepare_display_image(np.asarray(image.image))
        height, width = array.shape[:2]
        laterality = getattr(image, "laterality", None)
        image_id = getattr(image, "image_id", None)
        return {
            "index": index,
            "label": make_label("Fundus", index, image_id, laterality),
            "imageId": image_id or "",
            "laterality": laterality or "",
            "width": width,
            "height": height,
            "metadataText": self._format_metadata_text(
                label=f"Fundus {index + 1}",
                details={
                    "image_id": image_id,
                    "laterality": laterality,
                    "width": width,
                    "height": height,
                    "pixel_spacing": getattr(image, "pixel_spacing", None),
                    "patient_id": getattr(image, "patient_id", None),
                    "metadata": getattr(image, "metadata", None),
                },
            ),
        }

    def _slice_count(self, volume: Any) -> int:
        return len(self._get_slices(volume))

    def _get_slice(self, volume: Any, slice_index: int) -> np.ndarray:
        slices = self._get_slices(volume)
        if not 0 <= slice_index < len(slices):
            raise IndexError(f"Slice index out of range: {slice_index}")
        return slices[slice_index]

    def _get_slices(self, volume: Any) -> list[np.ndarray]:
        volume_data = getattr(volume, "volume")
        if isinstance(volume_data, list):
            return [self._prepare_display_image(np.asarray(item)) for item in volume_data]

        array = np.asarray(volume_data)
        if array.ndim == 0:
            return [self._prepare_display_image(array)]
        if array.ndim == 1:
            return [self._prepare_display_image(np.expand_dims(array, axis=0))]
        if array.ndim == 2:
            return [self._prepare_display_image(array)]

        slice_axis = self._infer_slice_axis(array, volume)
        volume_array = np.moveaxis(array, slice_axis, 0)
        return [self._prepare_display_image(volume_array[index]) for index in range(volume_array.shape[0])]

    def _infer_slice_axis(self, array: np.ndarray, volume: Any) -> int:
        expected = self._expected_slice_count(volume)
        candidate_axes = list(range(array.ndim))

        if array.ndim >= 4:
            non_channel_axes = [
                axis for axis in candidate_axes
                if array.shape[axis] not in {1, 3, 4}
            ]
            if len(non_channel_axes) >= 1:
                candidate_axes = non_channel_axes

        if expected and expected > 1:
            matching_axes = [axis for axis in candidate_axes if int(array.shape[axis]) == int(expected)]
            if len(matching_axes) == 1:
                return matching_axes[0]

        return min(candidate_axes, key=lambda axis: (int(array.shape[axis]), axis))

    def _expected_slice_count(self, volume: Any) -> int | None:
        candidates = [
            getattr(volume, "oct_header", {}).get("number_slices") if getattr(volume, "oct_header", None) else None,
            getattr(volume, "oct_header", {}).get("num_slices") if getattr(volume, "oct_header", None) else None,
            getattr(volume, "header", {}).get("number_slices") if getattr(volume, "header", None) else None,
        ]
        for candidate in candidates:
            try:
                if candidate is not None and int(candidate) > 0:
                    return int(candidate)
            except (TypeError, ValueError):
                continue
        return None

    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 0:
            return np.array([[array.item()]])

        array = np.squeeze(array)
        if array.ndim == 2:
            return array

        if array.ndim == 3:
            if array.shape[-1] in {1, 3, 4}:
                return array if array.shape[-1] != 1 else array[:, :, 0]
            if array.shape[0] in {1, 3, 4}:
                moved = np.moveaxis(array, 0, -1)
                return moved if moved.shape[-1] != 1 else moved[:, :, 0]
            collapse_axis = int(np.argmin(array.shape))
            return np.mean(array, axis=collapse_axis)

        while array.ndim > 3:
            collapse_axis = int(np.argmin(array.shape))
            array = np.mean(array, axis=collapse_axis)

        return self._prepare_display_image(array)

    def _get_projection_image(self, volume: Any) -> np.ndarray:
        slices = self._get_slices(volume)
        stack = np.asarray([normalize_to_uint8(slice_image) for slice_image in slices])
        if stack.ndim == 4:
            return np.mean(stack, axis=0)
        if stack.ndim == 3:
            return np.mean(stack, axis=1)
        return stack[0]

    def _get_bscan_display_units(
        self,
        *,
        width: int,
        height: int,
        pixel_spacing: Any,
    ) -> tuple[float, float]:
        width_spacing, height_spacing = self._extract_bscan_spacing(pixel_spacing)

        if width_spacing is None:
            width_spacing = 1.0
        if height_spacing is None:
            height_spacing = 1.0

        return float(width) * width_spacing, float(height) * height_spacing

    def _get_bscan_measurement_scale(self, *, pixel_spacing: Any) -> dict[str, Any]:
        width_spacing, height_spacing = self._extract_bscan_spacing(pixel_spacing)
        return {
            "isCalibrated": bool(width_spacing is not None and height_spacing is not None),
            "xMmPerPixel": width_spacing,
            "yMmPerPixel": height_spacing,
        }

    def _extract_bscan_spacing(self, pixel_spacing: Any) -> tuple[float | None, float | None]:
        width_spacing = None
        height_spacing = None

        if isinstance(pixel_spacing, (list, tuple)):
            if len(pixel_spacing) >= 3:
                width_spacing = self._safe_positive_float(pixel_spacing[0])
                candidates = [
                    self._safe_positive_float(pixel_spacing[1]),
                    self._safe_positive_float(pixel_spacing[2]),
                ]
                positive_candidates = [value for value in candidates if value is not None]
                height_spacing = min(positive_candidates) if positive_candidates else None
            elif len(pixel_spacing) == 2:
                width_spacing = self._safe_positive_float(pixel_spacing[0])
                height_spacing = self._safe_positive_float(pixel_spacing[1])

        return width_spacing, height_spacing

    def _safe_positive_float(self, value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) and parsed > 0 else None

    def _match_fundus_index(self, volume: Any, fundus_images: list[Any]) -> int:
        if not fundus_images:
            return -1

        volume_laterality = getattr(volume, "laterality", None) or ""
        volume_patient_id = getattr(volume, "patient_id", None) or ""
        best_index = -1
        best_score = -1

        for index, image in enumerate(fundus_images):
            score = 0
            image_laterality = getattr(image, "laterality", None) or ""
            image_patient_id = getattr(image, "patient_id", None) or ""
            if volume_laterality and image_laterality and volume_laterality == image_laterality:
                score += 2
            if volume_patient_id and image_patient_id and volume_patient_id == image_patient_id:
                score += 3
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _build_fda_metadata_segments(
        self,
        *,
        volume: Any,
        fundus_shape: tuple[int, ...],
    ) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]] | None, str]:
        metadata = getattr(volume, "metadata", None) or {}
        num_slices = self._slice_count(volume)
        if not isinstance(metadata, dict) or num_slices <= 0:
            return None, "missing"

        scan_params = metadata.get("param_scan_04") or metadata.get("param_scan_02") or {}
        y_dimension_mm = float(scan_params.get("y_dimension_mm") or 0.0)

        regist_info = metadata.get("regist_info") or {}
        reg_bbox = regist_info.get("bounding_box_in_fundus_pixels")
        if isinstance(reg_bbox, (list, tuple)) and len(reg_bbox) == 4:
            x0, y0, x1_or_half, y1_or_flag = [float(value) for value in reg_bbox]
            if y1_or_flag == 0.0 and x1_or_half > 0.0:
                start = (x0 - x1_or_half, y0)
                end = (x0 + x1_or_half, y0)
                line_bounds = clamp_bounds_to_shape(
                    (start[0], min(start[1], end[1]), end[0], max(start[1], end[1]) + 1.0),
                    fundus_shape,
                )
                if line_bounds is not None:
                    left, top, right, _ = line_bounds
                    y_pos = max(0.0, min(float(fundus_shape[0] - 1), y0))
                    return build_repeated_line_segments(num_slices, (left, y_pos), (right, y_pos)), "regist-line"

            rect_bounds = clamp_bounds_to_shape((x0, y0, x1_or_half, y1_or_flag), fundus_shape)
            if rect_bounds is not None and y_dimension_mm > 0:
                return build_horizontal_segments_from_bounds(num_slices, rect_bounds), "regist-rect"

        effective_range = metadata.get("effective_scan_range") or {}
        effective_bbox = effective_range.get("bounding_box_fundus_pixel")
        if isinstance(effective_bbox, (list, tuple)) and len(effective_bbox) == 4:
            rect_bounds = clamp_bounds_to_shape(tuple(float(value) for value in effective_bbox), fundus_shape)
            if rect_bounds is not None:
                if y_dimension_mm > 0:
                    return build_horizontal_segments_from_bounds(num_slices, rect_bounds), "effective-rect"
                left, top, right, bottom = rect_bounds
                y_pos = (top + bottom) / 2.0
                return build_repeated_line_segments(num_slices, (left, y_pos), (right, y_pos)), "effective-line"

        return None, "missing"

    def _select_matching_fundus_for_e2e(
        self,
        volume: Any,
        fundus_images: list[Any],
        index: int,
    ) -> tuple[Any | None, int, str]:
        if not fundus_images:
            return None, -1, "projection-fallback"

        fundus_by_id = {
            getattr(image, "image_id", None): (image_index, image)
            for image_index, image in enumerate(fundus_images)
            if getattr(image, "image_id", None)
        }
        volume_id = getattr(volume, "volume_id", None)
        if volume_id and volume_id in fundus_by_id:
            image_index, image = fundus_by_id[volume_id]
            return image, image_index, "matched-by-id"

        laterality = getattr(volume, "laterality", None)
        if laterality:
            same_laterality = [
                (image_index, image)
                for image_index, image in enumerate(fundus_images)
                if getattr(image, "laterality", None) == laterality
            ]
            if len(same_laterality) == 1:
                image_index, image = same_laterality[0]
                return image, image_index, "matched-by-laterality"

        fallback_index = min(index, len(fundus_images) - 1)
        return fundus_images[fallback_index], fallback_index, "matched-by-index-fallback"

    def _choose_localizer(
        self,
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

    def _project_e2e_overlay(
        self,
        *,
        group: list[dict[str, Any]],
        localizer_entry: dict[str, Any] | None,
        localizer_mode: str,
        fundus_shape: tuple[int, ...],
    ) -> OverlayInfo:
        if not group:
            scan_segments = build_parallel_segments(0, fundus_shape)
            return OverlayInfo(
                matched_fundus_index=-1,
                matched_fundus_label="",
                fundus_match_mode="projection-fallback",
                overlay_mode="fallback-parallel",
                projection_mode="no-bscan-metadata",
                localizer_mode=localizer_mode,
                warning="B-scan metadata is unavailable; using fallback scan lines.",
                scan_segments=scan_segments,
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
        transform_values = (localizer_entry or {}).get("transform")
        field_size_degrees = 30.0

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
            return OverlayInfo(
                matched_fundus_index=-1,
                matched_fundus_label="",
                fundus_match_mode="projection-fallback",
                overlay_mode="fallback-parallel",
                projection_mode="score-failed",
                localizer_mode=localizer_mode,
                warning="Localization fit failed; using fallback scan lines.",
                scan_segments=scan_segments,
                bounds=compute_scan_bounds(scan_segments),
            )

        warning = ""
        if localizer_entry is None and localizer_mode.startswith("disabled-count-mismatch"):
            warning = "Localizer count mismatch; using safe projected localization."
        elif localizer_entry is None and localizer_mode == "missing":
            warning = "Localizer metadata is missing; using B-scan geometry only."
        elif "affine" in best_label and localizer_entry is not None:
            warning = f"Applied localizer affine transform: {best_label}"

        scan_segments = points_to_segments(best_points)
        return OverlayInfo(
            matched_fundus_index=-1,
            matched_fundus_label="",
            fundus_match_mode="projection-fallback",
            overlay_mode="e2e-metadata",
            projection_mode=best_label,
            localizer_mode=localizer_mode,
            warning=warning,
            scan_segments=scan_segments,
            bounds=compute_scan_bounds(scan_segments),
        )

    def _serialize_segments(
        self,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[dict[str, Any]]:
        payload = []
        for start, end in segments:
            angle = float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))
            payload.append(
                {
                    "start": [float(start[0]), float(start[1])],
                    "end": [float(end[0]), float(end[1])],
                    "angleDegrees": angle,
                }
            )
        return payload

    def _format_metadata_text(self, label: str, details: dict[str, Any]) -> str:
        pretty = json.dumps(to_jsonable(details), ensure_ascii=False, indent=2)
        return f"{label}\n{pretty}"
