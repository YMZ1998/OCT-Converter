"""Zeiss DICOM dataset loading helpers for the OCT web viewer."""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData
from scripts.old.zeiss_dicom import ZEISSDicom

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE

OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"
RAW_ANALYSIS_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66"
ZEISS_MANUFACTURER_PREFIX = "Carl Zeiss Meditec"

FUNDUS_SOURCE_PRIORITY = {
    "fundus_photo": 0,
    "enface": 1,
    "fundus": 1,
    "multiframe_localizer": 2,
}


@dataclass
class RawAnalysisData:
    source_file: str
    fovea_x_norm: float | None
    fovea_y_norm: float | None
    segmentation_surfaces: list[np.ndarray]


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
    fundus_index: int
    image_id: str


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


def clean_digits(value: Any) -> str:
    return "".join(character for character in clean_text(value) if character.isdigit())


def clean_person_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    primary = text.split("=", 1)[0]
    parts = [part for part in primary.split("^") if part]
    return " ".join(parts).strip()


def normalize_dicom_date(value: Any) -> str:
    digits = clean_digits(value)
    return digits[:8] if len(digits) >= 8 else ""


def parse_dicom_datetime(date_time_value: Any, date_value: Any, time_value: Any) -> datetime | None:
    date_time_digits = clean_digits(date_time_value)
    if len(date_time_digits) >= 14:
        try:
            return datetime.strptime(date_time_digits[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass

    date_digits = clean_digits(date_value)
    time_digits = clean_digits(time_value)
    if len(date_digits) != 8:
        return None
    time_digits = (time_digits + "000000")[:6]
    try:
        return datetime.strptime(f"{date_digits}{time_digits}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def build_device_name(manufacturer: str, model: str) -> str:
    if manufacturer and model and model.lower() not in manufacturer.lower():
        return f"{manufacturer} {model}"
    return manufacturer or model


def extract_zeiss_header_metadata(ds: pydicom.dataset.FileDataset) -> dict[str, Any]:
    manufacturer = clean_text(getattr(ds, "Manufacturer", ""))
    model_name = clean_text(getattr(ds, "ManufacturerModelName", ""))
    patient_name = clean_person_name(getattr(ds, "PatientName", ""))
    metadata = {
        "patient_name": patient_name,
        "patient_id": clean_text(getattr(ds, "PatientID", "")),
        "patient_birth_date": normalize_dicom_date(getattr(ds, "PatientBirthDate", "")),
        "patient_sex": clean_text(getattr(ds, "PatientSex", "")),
        "manufacturer": manufacturer,
        "manufacturer_model_name": model_name,
        "device_name": build_device_name(manufacturer, model_name),
        "protocol_name": clean_text(getattr(ds, "ProtocolName", "")),
        "study_description": clean_text(getattr(ds, "StudyDescription", "")),
        "series_description": clean_text(getattr(ds, "SeriesDescription", "")),
        "laterality": clean_text(getattr(ds, "Laterality", "")) or clean_text(getattr(ds, "ImageLaterality", "")),
        "acquisition_datetime": parse_dicom_datetime(
            getattr(ds, "AcquisitionDateTime", ""),
            getattr(ds, "AcquisitionDate", "") or getattr(ds, "StudyDate", ""),
            getattr(ds, "AcquisitionTime", "") or getattr(ds, "StudyTime", ""),
        ),
    }
    return {key: value for key, value in metadata.items() if value not in ("", None)}


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


def derive_physical_size_mm(
    rows: int | None,
    columns: int | None,
    pixel_spacing: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
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


def to_display_image(image: np.ndarray) -> np.ndarray:
    array = normalize_to_uint8(np.asarray(image))
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3]
    raise ValueError(f"Unsupported image shape: {array.shape}")


def to_gray_image(image: np.ndarray) -> np.ndarray:
    array = to_display_image(image)
    if array.ndim == 2:
        return array
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def to_gray_volume(volume_array: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_array)
    if volume.ndim == 2:
        return volume[np.newaxis, ...]
    if volume.ndim == 3:
        return volume
    if volume.ndim == 4 and volume.shape[-1] >= 1:
        return volume[..., 0]
    raise ValueError(f"Unsupported volume shape: {volume.shape}")


def safe_dcmread(path: Path, *, stop_before_pixels: bool = False) -> pydicom.dataset.FileDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pydicom.config.disable_value_validation():
            return pydicom.dcmread(path, force=True, stop_before_pixels=stop_before_pixels)


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


def clean_path_like_input(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_zeiss_dicom_file(path: Path) -> bool:
    if not path.is_file():
        return False
    suffix_lower = path.suffix.lower()
    if suffix_lower not in {".dcm", ".dicom"} and path.name.upper() != "DICOMDIR":
        return False
    try:
        ds = safe_dcmread(path, stop_before_pixels=True)
    except Exception:
        return False
    manufacturer = clean_text(getattr(ds, "Manufacturer", ""))
    if manufacturer.startswith(ZEISS_MANUFACTURER_PREFIX):
        return True
    sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
    return sop_class_uid == RAW_ANALYSIS_SOP_CLASS_UID


def is_zeiss_exam_dir(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    dcm_files = sorted(
        candidate
        for candidate in directory.iterdir()
        if candidate.is_file()
        and (
            candidate.suffix.lower() in {".dcm", ".dicom"}
            or candidate.name.upper() == "DICOMDIR"
        )
    )
    return any(is_zeiss_dicom_file(path) for path in dcm_files)


def _unique_exam_dirs(exam_dirs: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for exam_dir in exam_dirs:
        resolved = exam_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def resolve_zeiss_exam_dirs(path: str | Path) -> list[Path]:
    candidate_path = clean_path_like_input(path)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    if candidate_path.is_file():
        if candidate_path.name.upper() == "DICOMDIR" or candidate_path.suffix.lower() in {".dcm", ".dicom"}:
            exam_dir = candidate_path.parent.resolve()
            return [exam_dir] if is_zeiss_exam_dir(exam_dir) else []
        return []

    direct_exam_dirs: list[Path] = []

    if (candidate_path / "DataFiles").is_dir():
        direct_exam_dirs.extend(
            child.resolve()
            for child in sorted((candidate_path / "DataFiles").iterdir())
            if child.is_dir() and is_zeiss_exam_dir(child)
        )

    if candidate_path.name.lower() == "datafiles":
        direct_exam_dirs.extend(
            child.resolve()
            for child in sorted(candidate_path.iterdir())
            if child.is_dir() and is_zeiss_exam_dir(child)
        )

    if is_zeiss_exam_dir(candidate_path):
        direct_exam_dirs.append(candidate_path.resolve())

    if direct_exam_dirs:
        return _unique_exam_dirs(direct_exam_dirs)

    nested_exam_dirs = [
        child.resolve()
        for child in sorted(candidate_path.iterdir())
        if child.is_dir() and is_zeiss_exam_dir(child)
    ]
    return _unique_exam_dirs(nested_exam_dirs)


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
    raw_dcm_files = sorted(
        path
        for path in exam_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
    )
    for path in raw_dcm_files:
        ds = safe_dcmread(path, stop_before_pixels=True)
        sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
        if sop_class_uid != RAW_ANALYSIS_SOP_CLASS_UID:
            continue
        analysis = parse_raw_analysis_file(path)
        if analysis is not None:
            return analysis
    return None


def reshape_segmentation_surface(
    surface: np.ndarray,
    num_slices: int,
    slice_width: int,
) -> np.ndarray | None:
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


def scope_rank(candidate: FundusCandidate, frame_of_reference_uid: str, series_instance_uid: str) -> int:
    if frame_of_reference_uid and candidate.frame_of_reference_uid == frame_of_reference_uid:
        return 0
    if series_instance_uid and candidate.series_instance_uid == series_instance_uid:
        return 1
    return 2


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
    x_pos = float(np.clip(center_x_norm, 0.0, 1.0)) * max(width - 1, 1)
    y_pos = float(np.clip(center_y_norm, 0.0, 1.0)) * max(height - 1, 1)
    return x_pos, y_pos


def build_raster_segments_from_bounds(
    num_slices: int,
    bounds: tuple[float, float, float, float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
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


def build_overlay_outline(
    bounds: tuple[float, float, float, float] | None,
) -> list[tuple[float, float]] | None:
    if bounds is None:
        return None
    x0, y0, width, height = bounds
    x1 = x0 + width
    y1 = y0 + height
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def warp_points(points: list[tuple[float, float]], matrix: np.ndarray | None) -> list[tuple[float, float]]:
    if not points:
        return []
    if matrix is None:
        return [(float(x_pos), float(y_pos)) for x_pos, y_pos in points]
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if matrix.shape == (2, 3):
        warped = cv2.transform(point_array, matrix)
    elif matrix.shape == (3, 3):
        warped = cv2.perspectiveTransform(point_array, matrix)
    else:
        raise ValueError(f"Unsupported transform shape: {matrix.shape}")
    return [tuple(map(float, point)) for point in warped.reshape(-1, 2)]


def warp_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    matrix: np.ndarray | None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if matrix is None:
        return segments
    warped_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in segments:
        warped = warp_points([start, end], matrix)
        warped_segments.append((warped[0], warped[1]))
    return warped_segments


def preprocess_registration_image(
    image: np.ndarray,
    *,
    long_side: int = 900,
) -> tuple[np.ndarray, float]:
    gray = to_gray_image(image)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    scale = long_side / max(gray.shape)
    resized = cv2.resize(
        gray,
        (
            max(1, int(round(gray.shape[1] * scale))),
            max(1, int(round(gray.shape[0] * scale))),
        ),
    )
    return resized, float(scale)


def register_reference_to_fundus(
    reference_image: np.ndarray,
    fundus_image: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    if reference_image.shape[:2] == fundus_image.shape[:2] and np.array_equal(reference_image, fundus_image):
        return np.eye(2, 3, dtype=np.float32), 1.0

    fundus_prepared, fundus_scale = preprocess_registration_image(fundus_image)
    reference_prepared, reference_scale = preprocess_registration_image(reference_image)

    motion = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)
    try:
        correlation, motion = cv2.findTransformECC(
            fundus_prepared,
            reference_prepared,
            motion,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return None, None

    scale_reference = np.array(
        [[reference_scale, 0.0, 0.0], [0.0, reference_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    inv_scale_fundus = np.array(
        [[1.0 / fundus_scale, 0.0, 0.0], [0.0, 1.0 / fundus_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    motion_h = np.vstack([motion, [0.0, 0.0, 1.0]]).astype(np.float32)
    original_motion = inv_scale_fundus @ motion_h @ scale_reference
    return original_motion[:2], float(correlation)


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
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], tuple[float, float, float, float] | None]:
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
        close_matches = [
            candidate
            for penalty, candidate in finite_matches
            if penalty <= best_penalty + 0.20
        ]
        return max(close_matches, key=lambda candidate: score_fundus_candidate(candidate, laterality))

    return max(scoped, key=lambda candidate: score_fundus_candidate(candidate, laterality))


def geometry_reference_priority(
    candidate: FundusCandidate,
    display_candidate: FundusCandidate | None,
) -> tuple[int, int, float]:
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


def _append_fundus_candidate(
    fundus_images: list[FundusImageWithMetaData],
    candidates: list[FundusCandidate],
    *,
    image: np.ndarray,
    laterality: str,
    source_file: str,
    source_kind: str,
    physical_width_mm: float | None,
    physical_height_mm: float | None,
    frame_of_reference_uid: str,
    series_instance_uid: str,
    patient_id: str | None,
    exam_id: str,
    image_id: str,
) -> FundusCandidate:
    image_array = to_display_image(image)
    height, width = image_array.shape[:2]
    ratio = min(height, width) / max(height, width)
    score = float(height * width) * (0.5 + 0.5 * ratio)
    fundus_index = len(fundus_images)
    pixel_spacing = None
    if (
        physical_width_mm is not None
        and physical_width_mm > 0
        and physical_height_mm is not None
        and physical_height_mm > 0
        and width > 0
        and height > 0
    ):
        pixel_spacing = [
            physical_width_mm / width,
            physical_height_mm / height,
        ]
    fundus_images.append(
        FundusImageWithMetaData(
            image=image_array,
            laterality=laterality or None,
            patient_id=patient_id,
            image_id=image_id,
            metadata={
                "vendor": "zeiss-dicom",
                "exam_id": exam_id,
                "source_file": source_file,
                "source_kind": source_kind,
                "physical_width_mm": physical_width_mm,
                "physical_height_mm": physical_height_mm,
                "frame_of_reference_uid": frame_of_reference_uid,
                "series_instance_uid": series_instance_uid,
            },
            pixel_spacing=pixel_spacing,
        )
    )
    candidate = FundusCandidate(
        image=image_array,
        laterality=laterality,
        source_file=source_file,
        source_kind=source_kind,
        width=width,
        height=height,
        score=score,
        physical_width_mm=physical_width_mm,
        physical_height_mm=physical_height_mm,
        frame_of_reference_uid=frame_of_reference_uid,
        series_instance_uid=series_instance_uid,
        fundus_index=fundus_index,
        image_id=image_id,
    )
    candidates.append(candidate)
    return candidate


def load_zeiss_oct_dataset(path: str | Path) -> dict[str, Any]:
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE

    exam_dirs = resolve_zeiss_exam_dirs(path)
    if not exam_dirs:
        raise FileNotFoundError(
            "Could not locate a Zeiss DICOM export directory or DICOM file set."
        )

    all_volumes: list[OCTVolumeWithMetaData] = []
    all_fundus_images: list[FundusImageWithMetaData] = []
    overlays: list[dict[str, Any]] = []

    for exam_dir in exam_dirs:
        raw_analysis = load_exam_raw_analysis(exam_dir)
        patient_id = ""
        dicomdir_path = exam_dir / "DICOMDIR"
        if dicomdir_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with pydicom.config.disable_value_validation():
                        dicomdir_ds = safe_dcmread(dicomdir_path, stop_before_pixels=True)
                        for record in getattr(dicomdir_ds, "DirectoryRecordSequence", []):
                            patient_id = clean_text(getattr(record, "PatientID", ""))
                            if patient_id:
                                break
            except Exception:
                patient_id = ""

        fundus_candidates: list[FundusCandidate] = []
        pending_volumes: list[dict[str, Any]] = []
        image_counter = 0

        dcm_files = sorted(
            candidate
            for candidate in exam_dir.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in {".dcm", ".dicom"}
        )
        if not dcm_files:
            continue

        for dcm_file in dcm_files:
            ds = safe_dcmread(dcm_file)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pydicom.config.disable_value_validation():
                    manufacturer = clean_text(getattr(ds, "Manufacturer", ""))
                    sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
            if (
                not manufacturer.startswith(ZEISS_MANUFACTURER_PREFIX)
                and sop_class_uid != RAW_ANALYSIS_SOP_CLASS_UID
            ):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pydicom.config.disable_value_validation():
                    classification = classify_dicom(ds)
            if classification == "non_image_dicom":
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pydicom.config.disable_value_validation():
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
                    header_metadata = extract_zeiss_header_metadata(ds)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with pydicom.config.disable_value_validation():
                        reader = ZEISSDicom(dcm_file)
                        oct_volumes, fundus_images = reader.read_data()
            except Exception:
                continue

            file_laterality = header_metadata.get("laterality", "")
            file_patient_id = header_metadata.get("patient_id", "") or patient_id or None
            file_patient_name = header_metadata.get("patient_name", "")
            file_patient_birth_date = header_metadata.get("patient_birth_date", "")
            file_patient_sex = header_metadata.get("patient_sex", "")
            file_device_name = header_metadata.get("device_name", "")
            file_scan_pattern = (
                header_metadata.get("protocol_name", "")
                or header_metadata.get("study_description", "")
                or header_metadata.get("series_description", "")
            )
            file_acquisition_date = header_metadata.get("acquisition_datetime")

            for image in fundus_images:
                image_counter += 1
                image_id = f"{exam_dir.name} | {dcm_file.stem} | {fundus_source_kind} | {image_counter}"
                candidate = _append_fundus_candidate(
                    all_fundus_images,
                    fundus_candidates,
                    image=image.image,
                    laterality=clean_text(getattr(image, "laterality", "")) or file_laterality,
                    source_file=dcm_file.name,
                    source_kind=fundus_source_kind,
                    physical_width_mm=physical_width_mm,
                    physical_height_mm=physical_height_mm,
                    frame_of_reference_uid=frame_of_reference_uid,
                    series_instance_uid=series_instance_uid,
                    patient_id=file_patient_id,
                    exam_id=exam_dir.name,
                    image_id=image_id,
                )
                fundus_image = all_fundus_images[candidate.fundus_index]
                fundus_image.patient_name = file_patient_name
                fundus_image.patient_dob = file_patient_birth_date
                fundus_image.sex = file_patient_sex
                fundus_image.device_name = file_device_name
                fundus_image.scan_pattern = file_scan_pattern
                fundus_image.metadata = {
                    **(fundus_image.metadata or {}),
                    **header_metadata,
                }

            for index, volume in enumerate(oct_volumes):
                gray_volume = to_gray_volume(volume.volume)
                if is_bscan_like(gray_volume):
                    pending_volumes.append(
                        {
                            "exam_dir": exam_dir,
                            "source_file": dcm_file,
                            "volume_index": index,
                            "laterality": clean_text(getattr(volume, "laterality", "")) or file_laterality,
                            "patient_id": file_patient_id,
                            "patient_name": file_patient_name,
                            "patient_birth_date": file_patient_birth_date,
                            "patient_sex": file_patient_sex,
                            "device_name": file_device_name,
                            "scan_pattern": file_scan_pattern,
                            "header_metadata": header_metadata,
                            "acquisition_date": file_acquisition_date or getattr(volume, "acquisition_date", None),
                            "volume": gray_volume,
                            "frame_of_reference_uid": frame_of_reference_uid,
                            "series_instance_uid": series_instance_uid,
                            "scan_width_mm": physical_width_mm,
                            "scan_height_mm": (
                                spacing_between_slices * max(gray_volume.shape[0] - 1, 1)
                                if spacing_between_slices and spacing_between_slices > 0 and gray_volume.shape[0] > 1
                                else physical_height_mm
                            ),
                            "bscan_height_mm": physical_height_mm,
                        }
                    )
                elif gray_volume.shape[0] > 0:
                    image_counter += 1
                    source_kind = classify_fundus_source_kind(
                        ds,
                        classification,
                        from_embedded_fundus=False,
                    )
                    image_id = f"{exam_dir.name} | {dcm_file.stem} | {source_kind} | {image_counter}"
                    _append_fundus_candidate(
                        all_fundus_images,
                        fundus_candidates,
                        image=gray_volume[0],
                        laterality=clean_text(getattr(volume, "laterality", "")) or file_laterality,
                        source_file=dcm_file.name,
                        source_kind=source_kind,
                        physical_width_mm=physical_width_mm,
                        physical_height_mm=physical_height_mm,
                        frame_of_reference_uid=frame_of_reference_uid,
                        series_instance_uid=series_instance_uid,
                        patient_id=file_patient_id,
                        exam_id=exam_dir.name,
                        image_id=image_id,
                    )

        for item in pending_volumes:
            best_fundus = choose_best_fundus(
                fundus_candidates,
                item["laterality"],
                frame_of_reference_uid=item["frame_of_reference_uid"],
                series_instance_uid=item["series_instance_uid"],
                scan_width_mm=item["scan_width_mm"],
                scan_height_mm=item["scan_height_mm"],
            )
            geometry_candidate = choose_geometry_reference(
                fundus_candidates,
                item["laterality"],
                display_candidate=best_fundus,
                frame_of_reference_uid=item["frame_of_reference_uid"],
                series_instance_uid=item["series_instance_uid"],
            )

            display_source_candidate = best_fundus
            displayed_fundus = best_fundus.image if best_fundus is not None else make_projection_fundus(item["volume"])
            geometry_image = geometry_candidate.image if geometry_candidate is not None else displayed_fundus

            photo_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
            photo_bounds: tuple[float, float, float, float] | None = None
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
            fovea_point = build_reference_point(
                geometry_image.shape,
                center_x_norm=raw_analysis.fovea_x_norm if raw_analysis else None,
                center_y_norm=raw_analysis.fovea_y_norm if raw_analysis else None,
            )

            registration_score = 1.0
            transform = None
            overlay_mode = "zeiss-projection-line"
            projection_mode = "zeiss-raster-bounding-box" if bounds is not None else "zeiss-raster-line"
            localizer_mode = "projection-fallback"
            warning = ""
            matched_fundus_index = best_fundus.fundus_index if best_fundus is not None else -1
            matched_fundus_label = best_fundus.image_id if best_fundus is not None else "projection-fallback"

            if geometry_candidate is not None:
                localizer_mode = geometry_candidate.source_kind
            elif best_fundus is not None:
                localizer_mode = best_fundus.source_kind

            if geometry_candidate is not None and best_fundus is not None:
                same_image = (
                    geometry_candidate.source_file == best_fundus.source_file
                    and geometry_candidate.source_kind == best_fundus.source_kind
                    and geometry_candidate.image.shape == best_fundus.image.shape
                    and np.array_equal(geometry_candidate.image, best_fundus.image)
                )
                if not same_image:
                    transform, registration_score = register_reference_to_fundus(
                        geometry_candidate.image,
                        best_fundus.image,
                    )

            display_reason = None
            if transform is not None and registration_score is not None and registration_score >= 0.20:
                segments = warp_segments(segments, transform)
                if outline is not None:
                    outline = warp_points(outline, transform)
                    xs = [point[0] for point in outline]
                    ys = [point[1] for point in outline]
                    bounds = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                if fovea_point is not None:
                    fovea_point = warp_points([fovea_point], transform)[0]
                overlay_mode = "zeiss-registered-bounding-box" if bounds is not None else "zeiss-registered-line"
                projection_mode = "zeiss-registered-bounding-box" if bounds is not None else "zeiss-registered-line"
                localizer_mode = "registered-to-fundus"
                matched_fundus_index = best_fundus.fundus_index
                matched_fundus_label = best_fundus.image_id
            else:
                if best_fundus is not None and photo_segments:
                    display_source_candidate = best_fundus
                    matched_fundus_index = best_fundus.fundus_index
                    matched_fundus_label = best_fundus.image_id
                    segments = photo_segments
                    bounds = photo_bounds
                    outline = build_overlay_outline(bounds)
                    fovea_point = build_reference_point(
                        best_fundus.image.shape,
                        center_x_norm=raw_analysis.fovea_x_norm if raw_analysis else None,
                        center_y_norm=raw_analysis.fovea_y_norm if raw_analysis else None,
                    )
                    overlay_mode = "zeiss-photo-bounding-box" if bounds is not None else "zeiss-photo-line"
                    projection_mode = "zeiss-photo-bounding-box" if bounds is not None else "zeiss-photo-line"
                    localizer_mode = "fundus-photo-approx"
                    if geometry_candidate is not None and best_fundus.fundus_index != geometry_candidate.fundus_index:
                        if registration_score is None:
                            display_reason = (
                                "Automatic registration from Zeiss localizer to fundus image failed; "
                                "showing the real fundus photo with approximate scan overlay."
                            )
                        else:
                            display_reason = (
                                "Registration score is too low for reliable fundus alignment; "
                                f"showing the real fundus photo with approximate scan overlay ({registration_score:.3f} < 0.20)."
                            )
                        warning = display_reason
                elif geometry_candidate is not None:
                    display_source_candidate = geometry_candidate
                    matched_fundus_index = geometry_candidate.fundus_index
                    matched_fundus_label = geometry_candidate.image_id
                    overlay_mode = "zeiss-geometry-bounding-box" if bounds is not None else "zeiss-geometry-line"
                    projection_mode = "zeiss-geometry-bounding-box" if bounds is not None else "zeiss-geometry-line"
                    localizer_mode = geometry_candidate.source_kind
                    if best_fundus is not None and best_fundus.fundus_index != geometry_candidate.fundus_index:
                        if registration_score is None:
                            display_reason = "Automatic registration from Zeiss localizer to fundus image failed; falling back to geometry reference."
                        else:
                            display_reason = (
                                "Registration score is too low for reliable fundus alignment; "
                                f"falling back to geometry reference ({registration_score:.3f} < 0.20)."
                            )
                        warning = display_reason
                elif best_fundus is not None:
                    matched_fundus_index = best_fundus.fundus_index
                    matched_fundus_label = best_fundus.image_id
                    overlay_mode = "zeiss-fundus-bounding-box" if bounds is not None else "zeiss-fundus-line"
                    projection_mode = "zeiss-fundus-bounding-box" if bounds is not None else "zeiss-fundus-line"
                    localizer_mode = best_fundus.source_kind

            segmentation_surfaces = prepare_segmentation_surfaces(
                raw_analysis,
                num_slices=item["volume"].shape[0],
                slice_width=item["volume"].shape[2],
            )
            contours = {
                f"Surface {index + 1}": [surface[slice_index] for slice_index in range(surface.shape[0])]
                for index, surface in enumerate(segmentation_surfaces)
            }

            bscan_width = int(item["volume"].shape[2]) if item["volume"].ndim >= 3 else 0
            bscan_height = int(item["volume"].shape[1]) if item["volume"].ndim >= 3 else 0
            pixel_spacing = None
            if (
                item["scan_width_mm"] is not None
                and item["scan_width_mm"] > 0
                and item["bscan_height_mm"] is not None
                and item["bscan_height_mm"] > 0
                and bscan_width > 0
                and bscan_height > 0
            ):
                pixel_spacing = [
                    float(item["scan_width_mm"]) / float(bscan_width),
                    float(item["bscan_height_mm"]) / float(bscan_height),
                ]

            metadata = {
                "vendor": "zeiss-dicom",
                "exam_dir": str(item["exam_dir"]),
                "patient_id": item["patient_id"] or patient_id,
                "source_file": item["source_file"].name,
                "volume_index": item["volume_index"],
                "laterality": item["laterality"],
                "num_slices": int(item["volume"].shape[0]),
                "slice_shape": [int(item["volume"].shape[1]), int(item["volume"].shape[2])],
                "display_source": display_source_candidate.source_file if display_source_candidate else "projection-fallback",
                "display_source_kind": display_source_candidate.source_kind if display_source_candidate else "projection-fallback",
                "display_reason": display_reason,
                "registration_score": registration_score,
                "raw_analysis_source": raw_analysis.source_file if raw_analysis else None,
                "fovea_center_norm": (
                    [raw_analysis.fovea_x_norm, raw_analysis.fovea_y_norm]
                    if raw_analysis and raw_analysis.fovea_x_norm is not None and raw_analysis.fovea_y_norm is not None
                    else None
                ),
                "fovea_point": [float(fovea_point[0]), float(fovea_point[1])] if fovea_point is not None else None,
                "segmentation_surface_count": len(segmentation_surfaces),
                "fundus_source": best_fundus.source_file if best_fundus is not None else None,
                "fundus_source_kind": best_fundus.source_kind if best_fundus is not None else None,
                "geometry_reference_source": geometry_candidate.source_file if geometry_candidate is not None else None,
                "geometry_reference_kind": geometry_candidate.source_kind if geometry_candidate is not None else None,
                "matched_fundus_index": matched_fundus_index,
                "matched_fundus_label": matched_fundus_label,
                "scan_width_mm": item["scan_width_mm"],
                "scan_height_mm": item["scan_height_mm"],
                "bscan_height_mm": item["bscan_height_mm"],
                "overlay_bounds": [float(value) for value in bounds] if bounds is not None else None,
                "photo_overlay_bounds": [float(value) for value in photo_bounds] if photo_bounds is not None else None,
                "frame_of_reference_uid": item["frame_of_reference_uid"],
                "series_instance_uid": item["series_instance_uid"],
            }

            volume_id = item["source_file"].stem
            if len(exam_dirs) > 1:
                volume_id = f"{item['exam_dir'].name}/{volume_id}"
            if item["laterality"]:
                volume_id = f"{volume_id} ({item['laterality']})"

            all_volumes.append(
                OCTVolumeWithMetaData(
                    volume=item["volume"],
                    patient_id=item["patient_id"] or patient_id or None,
                    volume_id=volume_id,
                    acquisition_date=item["acquisition_date"],
                    laterality=item["laterality"] or None,
                    contours=contours or None,
                    pixel_spacing=pixel_spacing,
                    metadata=metadata,
                    header=item["header_metadata"],
                )
            )
            current_volume = all_volumes[-1]
            current_volume.patient_name = item["patient_name"]
            current_volume.patient_dob = item["patient_birth_date"]
            current_volume.sex = item["patient_sex"]
            current_volume.device_name = item["device_name"]
            current_volume.scan_pattern = item["scan_pattern"]
            overlays.append(
                {
                    "matched_fundus_index": matched_fundus_index,
                    "matched_fundus_label": matched_fundus_label,
                    "fundus_match_mode": "zeiss-dicom",
                    "overlay_mode": overlay_mode,
                    "projection_mode": projection_mode,
                    "localizer_mode": localizer_mode,
                    "warning": warning,
                    "scan_segments": segments,
                    "bounds": bounds,
                }
            )

    if not all_volumes:
        raise ValueError("No Zeiss B-scan volumes were extracted from the selected input.")

    return {
        "input_path": str(clean_path_like_input(path)),
        "exam_dirs": [str(exam_dir) for exam_dir in exam_dirs],
        "volumes": all_volumes,
        "fundus_images": all_fundus_images,
        "overlays": overlays,
    }


__all__ = [
    "is_zeiss_dicom_file",
    "is_zeiss_exam_dir",
    "load_zeiss_oct_dataset",
    "resolve_zeiss_exam_dirs",
]
