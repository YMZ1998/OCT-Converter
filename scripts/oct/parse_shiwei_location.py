from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData


DEFAULT_INPUT_DIR = (
    "E:\\Data\\OCT2\\视微OCT\\00538_20260228005_A-031001AMD_OS_2026-03-06_11-47-06Cube 6x6 512x512\\Dicom"
)

FORMAT_ALIASES = {
    "tif": "tiff",
    "tiff": "tiff",
    "gif": "gif",
}

SEGMENTATION_COLORS = [
    "#00ffff",
    "#ff66cc",
    "#66ff66",
    "#ff9933",
    "#66b3ff",
    "#ffffff",
    "#ff5555",
    "#aa88ff",
    "#00cc99",
    "#ffcc00",
    "#ff99ff",
    "#99ccff",
    "#aaffaa",
]
SEGMENTATION_STORAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"
DICOM_FILE_SUFFIXES = {".dcm", ".dicom"}


def _as_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]

    converted: list[float] = []
    for value in values:
        try:
            converted.append(float(value))
        except (TypeError, ValueError):
            return None
    return converted or None


def _extract_shared_pixel_spacing(dataset) -> list[float] | None:
    shared = getattr(dataset, "SharedFunctionalGroupsSequence", None)
    if shared and hasattr(shared[0], "PixelMeasuresSequence"):
        return _as_float_list(shared[0].PixelMeasuresSequence[0].PixelSpacing)
    return _as_float_list(getattr(dataset, "PixelSpacing", None))


def extract_volume_pixel_spacing(dataset) -> list[float] | None:
    spacing = _extract_shared_pixel_spacing(dataset)
    if spacing is None:
        return None

    row_spacing = float(spacing[0])
    col_spacing = float(spacing[1] if len(spacing) > 1 else spacing[0])
    try:
        slice_spacing = float(getattr(dataset, "SpacingBetweenSlices", row_spacing))
    except (TypeError, ValueError):
        slice_spacing = row_spacing
    return [col_spacing, row_spacing, slice_spacing]


def extract_fundus_pixel_spacing(dataset) -> list[float] | None:
    spacing = _extract_shared_pixel_spacing(dataset)
    if spacing is None:
        return None

    row_spacing = float(spacing[0])
    col_spacing = float(spacing[1] if len(spacing) > 1 else spacing[0])
    return [col_spacing, row_spacing]


def _extract_laterality(dataset) -> str | None:
    return getattr(dataset, "ImageLaterality", None) or getattr(dataset, "Laterality", None)


def _extract_acquisition_datetime(dataset) -> datetime | None:
    date_value = (
        getattr(dataset, "AcquisitionDate", None)
        or getattr(dataset, "ContentDate", None)
        or getattr(dataset, "StudyDate", None)
    )
    time_value = (
        getattr(dataset, "AcquisitionTime", None)
        or getattr(dataset, "ContentTime", None)
        or getattr(dataset, "StudyTime", None)
    )
    if not date_value:
        return None

    date_text = str(date_value).strip()
    time_text = str(time_value or "").split(".")[0].strip()
    for pattern in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(f"{date_text}{time_text}", pattern)
        except ValueError:
            continue
    return None


def _parse_dicom_date(value: Any) -> datetime | None:
    text = "".join(character for character in str(value or "").strip() if character.isdigit())
    if len(text) != 8:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return None


def extract_valid_patient_birth_date(dataset) -> str | None:
    raw_birth_date = getattr(dataset, "PatientBirthDate", None)
    birth_date = _parse_dicom_date(raw_birth_date)
    if birth_date is None:
        return raw_birth_date or None

    exam_dates = [
        _parse_dicom_date(getattr(dataset, "AcquisitionDate", None)),
        _parse_dicom_date(getattr(dataset, "ContentDate", None)),
        _parse_dicom_date(getattr(dataset, "StudyDate", None)),
    ]
    exam_dates = [item for item in exam_dates if item is not None]
    if exam_dates and birth_date > min(exam_dates):
        return None

    if birth_date > datetime.now():
        return None

    if birth_date.year < 1900:
        return None

    return birth_date.strftime("%Y%m%d")


def extract_basic_dicom_metadata(dataset, *, include_paths: dict[str, str] | None = None) -> dict[str, Any]:
    metadata = {
        "manufacturer": getattr(dataset, "Manufacturer", None),
        "manufacturer_model_name": getattr(dataset, "ManufacturerModelName", None),
        "patient_id": getattr(dataset, "PatientID", None),
        "patient_name": str(getattr(dataset, "PatientName", "")) or None,
        "patient_sex": getattr(dataset, "PatientSex", None),
        "patient_birth_date": extract_valid_patient_birth_date(dataset),
        "study_instance_uid": getattr(dataset, "StudyInstanceUID", None),
        "series_instance_uid": getattr(dataset, "SeriesInstanceUID", None),
        "sop_instance_uid": getattr(dataset, "SOPInstanceUID", None),
        "laterality": _extract_laterality(dataset),
        "modality": getattr(dataset, "Modality", None),
        "rows": getattr(dataset, "Rows", None),
        "columns": getattr(dataset, "Columns", None),
        "number_of_frames": getattr(dataset, "NumberOfFrames", None),
        "pixel_spacing": _extract_shared_pixel_spacing(dataset),
        "paths": include_paths or None,
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", {})}


def build_shiwei_contours(
    segmentation_surfaces,
    coordinates,
    segmentation_orientation,
    *,
    slice_count: int,
    target_width: int,
) -> dict[str, list[np.ndarray | None]]:
    if segmentation_surfaces is None:
        return {}

    layer_count = int(segmentation_surfaces.shape[0])
    contours = {f"Layer {layer_index + 1}": [] for layer_index in range(layer_count)}

    for slice_idx in range(slice_count):
        curves = get_segmentation_curves(
            segmentation_surfaces,
            coordinates,
            slice_idx,
            target_width,
            orientation=segmentation_orientation,
        )
        for layer_index in range(layer_count):
            key = f"Layer {layer_index + 1}"
            if layer_index < len(curves):
                contours[key].append(np.asarray(curves[layer_index], dtype=np.float32))
            else:
                contours[key].append(None)

    return contours


def build_shiwei_scan_segments(
    coordinates,
    fundus_shape,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    height, width = fundus_shape[:2]
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for coords in coordinates:
        values = np.asarray(coords, dtype=float).reshape(-1)
        if values.size < 4 or not np.isfinite(values[:4]).all() or not np.any(values[:4]):
            continue
        x0, y0, x1, y1 = values[:4]
        segments.append(
            (
                (
                    float(np.clip(x0, 0.0, max(width - 1, 0))),
                    float(np.clip(y0, 0.0, max(height - 1, 0))),
                ),
                (
                    float(np.clip(x1, 0.0, max(width - 1, 0))),
                    float(np.clip(y1, 0.0, max(height - 1, 0))),
                ),
            )
        )
    return segments


def parse_args():
    parser = argparse.ArgumentParser(
        description="View Shiwei OCT location data and export the synchronized visualization.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the Shiwei DICOM files.",
    )
    parser.add_argument(
        "--bscan-file",
        default=None,
        help="Path to the structural B-scan DICOM file.",
    )
    parser.add_argument(
        "--fundus-file",
        default=None,
        help="Path to the fundus/CSSO DICOM file.",
    )
    parser.add_argument(
        "--seg-file",
        default=None,
        help="Path to the segmentation DICOM file.",
    )
    parser.add_argument(
        "--formats",
        default="",
        help="Comma-separated export formats: tiff,gif.",
    )
    parser.add_argument(
        "--output-dir",
        default="E:\\Data\\OCT2\\result",
        help="Export directory. Defaults to <input-dir>/exports when --formats is provided.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Export file basename. Defaults to the B-scan filename stem.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Frames per second for GIF export.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive Matplotlib viewer.",
    )
    return parser.parse_args()


def normalize_formats(formats_text):
    if not formats_text:
        return []

    normalized = []
    for item in formats_text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in FORMAT_ALIASES:
            raise ValueError(
                f"Unsupported export format: {item}. Supported values are avi, tiff, gif."
            )
        value = FORMAT_ALIASES[key]
        if value not in normalized:
            normalized.append(value)
    return normalized


def find_candidate(input_dir, keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    for path in iter_dicom_files(input_dir):
        name = path.name.lower()
        if all(keyword in name for keyword in keywords) and not any(
            keyword in name for keyword in exclude_keywords
        ):
            return str(path)
    return None


def find_by_suffix(input_dir, suffixes):
    suffixes = [suffix.lower() for suffix in suffixes]
    for path in iter_dicom_files(input_dir):
        name = path.name.lower()
        for suffix in suffixes:
            if name.endswith(suffix):
                return str(path)
    return None


def iter_dicom_files(input_dir):
    input_path = Path(input_dir)
    candidates = [
        path for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in DICOM_FILE_SUFFIXES
    ]
    return candidates


def read_dicom_header(path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None


def find_segmentation_file_by_header(input_dir):
    for path in iter_dicom_files(input_dir):
        dataset = read_dicom_header(path)
        if dataset is None:
            continue
        modality = str(getattr(dataset, "Modality", "") or "").upper()
        sop_class_uid = str(getattr(dataset, "SOPClassUID", "") or "")
        series_description = str(getattr(dataset, "SeriesDescription", "") or "").lower()
        if (
            modality == "SEG"
            or sop_class_uid == SEGMENTATION_STORAGE_SOP_CLASS_UID
            or "seg" in series_description
            or "segment" in series_description
        ):
            return str(path)
    return None


def resolve_input_files(input_dir, bscan_file=None, fundus_file=None, seg_file=None):
    if bscan_file is None:
        bscan_file = find_by_suffix(
            input_dir,
            [
                "_rotatedstructural_structural.dcm",
                "_structural.dcm",
            ],
        )
        if bscan_file is None:
            bscan_file = find_candidate(input_dir, ["structural"], ["csso", "segmentation"])
        if bscan_file is None:
            bscan_file = find_candidate(input_dir, ["oct"])

    if fundus_file is None:
        fundus_file = find_by_suffix(
            input_dir,
            [
                "_rotatedstructural_csso.dcm",
                "_csso.dcm",
            ],
        )
        if fundus_file is None:
            fundus_file = find_candidate(input_dir, ["csso"])
        if fundus_file is None:
            fundus_file = find_candidate(input_dir, ["fundus"])

    if seg_file is None:
        seg_file = find_by_suffix(
            input_dir,
            [
                "_segmentation.dcm",
            ],
        )
        if seg_file is None:
            seg_file = find_candidate(input_dir, ["segmentation"])
        if seg_file is None:
            seg_file = find_segmentation_file_by_header(input_dir)

    if bscan_file is None or not os.path.exists(bscan_file):
        raise FileNotFoundError("Could not locate the structural B-scan DICOM file.")
    if fundus_file is None or not os.path.exists(fundus_file):
        raise FileNotFoundError("Could not locate the fundus/CSSO DICOM file.")

    return bscan_file, fundus_file, seg_file


def normalize_to_uint8(image):
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


def ensure_rgb(image):
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def pad_to_height(image, target_height, fill_value=0):
    if image.shape[0] == target_height:
        return image

    pad_total = target_height - image.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return np.pad(
        image,
        ((pad_top, pad_bottom), (0, 0), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )


def draw_line_rgb(image, start, end, color=(255, 0, 0), thickness=2):
    x0, y0 = start
    x1, y1 = end
    length = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.round(np.linspace(x0, x1, length)).astype(int)
    ys = np.round(np.linspace(y0, y1, length)).astype(int)
    radius = max(0, thickness // 2)

    for x, y in zip(xs, ys):
        x_min = max(0, x - radius)
        x_max = min(image.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(image.shape[0], y + radius + 1)
        image[y_min:y_max, x_min:x_max] = color


def parse_segmentation_surfaces(seg_file, num_slices, source_depth_limit):
    if seg_file is None or not os.path.exists(seg_file):
        return None

    ds_seg = pydicom.dcmread(seg_file)
    seg_array = ds_seg.pixel_array

    if seg_array.dtype == np.uint32:
        seg_array = seg_array.view(np.float32)
    else:
        seg_array = seg_array.astype(np.float32)

    if seg_array.ndim == 2:
        seg_array = seg_array[np.newaxis, ...]

    if seg_array.ndim != 3:
        return None

    valid_mask = (seg_array >= 0.0) & (seg_array <= source_depth_limit - 1)
    seg_array = np.where(valid_mask, seg_array, np.nan)

    valid_layers = []
    for layer in seg_array:
        finite_ratio = np.isfinite(layer).sum() / layer.size
        if finite_ratio < 0.95:
            continue
        if np.nanmax(layer) - np.nanmin(layer) < 1e-3:
            continue
        is_duplicate = False
        for existing in valid_layers:
            mean_abs_diff = float(np.nanmean(np.abs(layer - existing)))
            if mean_abs_diff < 1e-3:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        valid_layers.append(layer)

    if not valid_layers:
        return None

    valid_layers.sort(key=lambda layer: float(np.nanmean(layer)))
    return np.stack(valid_layers, axis=0)


def get_segmentation_curves(
    segmentation_surfaces,
    coordinates,
    slice_idx,
    target_width,
    orientation=None,
):
    if segmentation_surfaces is None:
        return []

    slice_idx = int(np.clip(slice_idx, 0, segmentation_surfaces.shape[1] - 1))
    x_source = np.arange(segmentation_surfaces.shape[2], dtype=float)
    x_target = np.linspace(0, segmentation_surfaces.shape[2] - 1, target_width, dtype=float)
    curves = []

    for layer in segmentation_surfaces:
        curve = np.asarray(layer[slice_idx], dtype=float)
        valid = np.isfinite(curve)
        if np.count_nonzero(valid) < 2:
            continue
        curve_interp = np.interp(x_target, x_source[valid], curve[valid])
        if orientation is not None and "vertical_scale" in orientation:
            curve_interp = curve_interp * float(orientation["vertical_scale"])
        if orientation is not None and "vertical_offset" in orientation:
            curve_interp = curve_interp + float(orientation["vertical_offset"])
        curves.append(curve_interp)

    return curves


def load_shiwei_data(
    input_dir,
    bscan_file=None,
    fundus_file=None,
    seg_file=None,
):
    bscan_file, fundus_file, seg_file = resolve_input_files(
        input_dir, bscan_file, fundus_file, seg_file
    )

    ds_bscan = pydicom.dcmread(bscan_file)
    volume = ds_bscan.pixel_array
    num_slices = volume.shape[0]

    ds_fundus = pydicom.dcmread(fundus_file)
    fundus = ds_fundus.pixel_array

    angles = []
    coordinates = []

    if hasattr(ds_bscan, "PerFrameFunctionalGroupsSequence"):
        for frame in ds_bscan.PerFrameFunctionalGroupsSequence:
            if hasattr(frame, "OphthalmicFrameLocationSequence"):
                location = frame.OphthalmicFrameLocationSequence[0]
                coords = np.array(location.ReferenceCoordinates, dtype=float)
            else:
                coords = np.array([0, 0, 0, 0], dtype=float)

            coordinates.append(coords)
            angle = np.arctan2(coords[3] - coords[1], coords[2] - coords[0])
            angles.append(float(np.degrees(angle)))
    else:
        coordinates = [np.array([0, 0, 0, 0], dtype=float) for _ in range(num_slices)]
        angles = [0.0] * num_slices

    segmentation_surfaces = None
    segmentation_orientation = None
    if seg_file is not None and os.path.exists(seg_file):
        ds_seg = pydicom.dcmread(seg_file, stop_before_pixels=True)

        bscan_shared = getattr(ds_bscan, "SharedFunctionalGroupsSequence", None)
        seg_shared = getattr(ds_seg, "SharedFunctionalGroupsSequence", None)

        bscan_row_spacing = None
        seg_depth_spacing = None
        if bscan_shared and hasattr(bscan_shared[0], "PixelMeasuresSequence"):
            bscan_row_spacing = float(bscan_shared[0].PixelMeasuresSequence[0].PixelSpacing[0])
        if seg_shared and hasattr(seg_shared[0], "PixelMeasuresSequence"):
            seg_depth_spacing = float(seg_shared[0].PixelMeasuresSequence[0].PixelSpacing[1])

        if bscan_row_spacing and seg_depth_spacing:
            source_depth_limit = int(round(volume.shape[1] * bscan_row_spacing / seg_depth_spacing))
            vertical_scale = seg_depth_spacing / bscan_row_spacing
        else:
            source_depth_limit = volume.shape[1]
            vertical_scale = 1.0

        segmentation_surfaces = parse_segmentation_surfaces(
            seg_file=seg_file,
            num_slices=num_slices,
            source_depth_limit=source_depth_limit,
        )
        segmentation_orientation = {
            "vertical_scale": vertical_scale,
            "vertical_offset": 0.0,
        }

    return (
        volume,
        fundus,
        coordinates,
        angles,
        segmentation_surfaces,
        segmentation_orientation,
        bscan_file,
        fundus_file,
        seg_file,
    )


def load_shiwei_oct_dataset(
    input_path,
    bscan_file=None,
    fundus_file=None,
    seg_file=None,
):
    input_root = Path(input_path)
    input_dir = input_root if input_root.is_dir() else input_root.parent

    (
        volume,
        fundus,
        coordinates,
        angles,
        segmentation_surfaces,
        segmentation_orientation,
        bscan_file,
        fundus_file,
        seg_file,
    ) = load_shiwei_data(
        input_dir=str(input_dir),
        bscan_file=bscan_file,
        fundus_file=fundus_file,
        seg_file=seg_file,
    )

    ds_bscan = pydicom.dcmread(bscan_file, stop_before_pixels=True)
    ds_fundus = pydicom.dcmread(fundus_file, stop_before_pixels=True)

    file_paths = {
        "input_dir": str(input_dir),
        "bscan_file": str(bscan_file),
        "fundus_file": str(fundus_file),
    }
    if seg_file:
        file_paths["seg_file"] = str(seg_file)

    contours = build_shiwei_contours(
        segmentation_surfaces,
        coordinates,
        segmentation_orientation,
        slice_count=int(volume.shape[0]),
        target_width=int(volume.shape[-1]),
    )

    volume_object = OCTVolumeWithMetaData(
        volume=volume,
        patient_id=getattr(ds_bscan, "PatientID", None),
        patient_dob=extract_valid_patient_birth_date(ds_bscan),
        volume_id=Path(bscan_file).stem,
        acquisition_date=_extract_acquisition_datetime(ds_bscan),
        laterality=_extract_laterality(ds_bscan),
        contours=contours or None,
        pixel_spacing=extract_volume_pixel_spacing(ds_bscan),
        metadata={
            "source": "shiwei",
            "angles_degrees": [float(angle) for angle in angles],
            "coordinate_count": len(coordinates),
            "segmentation_file_present": bool(seg_file),
            "paths": file_paths,
        },
        header=extract_basic_dicom_metadata(ds_bscan, include_paths=file_paths),
    )

    fundus_object = FundusImageWithMetaData(
        image=fundus,
        laterality=_extract_laterality(ds_fundus),
        patient_id=getattr(ds_fundus, "PatientID", None),
        image_id=Path(fundus_file).stem,
        patient_dob=extract_valid_patient_birth_date(ds_fundus),
        metadata=extract_basic_dicom_metadata(ds_fundus, include_paths=file_paths),
        pixel_spacing=extract_fundus_pixel_spacing(ds_fundus),
    )

    return {
        "volume": volume_object,
        "fundus": fundus_object,
        "coordinates": [np.asarray(item, dtype=float) for item in coordinates],
        "angles": [float(angle) for angle in angles],
        "segments": build_shiwei_scan_segments(coordinates, fundus.shape),
        "segmentation_surfaces": segmentation_surfaces,
        "segmentation_orientation": segmentation_orientation,
        "input_dir": str(input_dir),
        "bscan_file": str(bscan_file),
        "fundus_file": str(fundus_file),
        "seg_file": str(seg_file) if seg_file else None,
    }


def render_frame(fundus, bscan_slice, coords, angle, slice_idx, total_slices):
    fundus_rgb = ensure_rgb(fundus).copy()
    bscan_rgb = ensure_rgb(bscan_slice)

    coords = np.asarray(coords, dtype=float)
    if coords.size == 4 and np.any(coords):
        x0, y0, x1, y1 = np.round(coords).astype(int)
        draw_line_rgb(fundus_rgb, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)

    panel_height = max(fundus_rgb.shape[0], bscan_rgb.shape[0])
    fundus_rgb = pad_to_height(fundus_rgb, panel_height, fill_value=0)
    bscan_rgb = pad_to_height(bscan_rgb, panel_height, fill_value=0)

    spacer = np.full((panel_height, 24, 3), 20, dtype=np.uint8)
    body = np.concatenate([fundus_rgb, spacer, bscan_rgb], axis=1)
    footer = np.full((32, body.shape[1], 3), 20, dtype=np.uint8)

    progress = int(body.shape[1] * (slice_idx + 1) / max(1, total_slices))
    footer[:, :progress] = (64, 160, 255)

    return np.concatenate([body, footer], axis=0)


def export_visualization(
    volume,
    fundus,
    coordinates,
    angles,
    segmentation_surfaces,
    segmentation_orientation,
    formats,
    output_dir,
    basename,
    fps,
):
    import imageio
    import tifffile

    os.makedirs(output_dir, exist_ok=True)

    gif_writer = None
    tiff_frames = []

    try:
        if "gif" in formats:
            gif_path = os.path.join(output_dir, f"{basename}.gif")
            gif_writer = imageio.get_writer(gif_path, mode="I", duration=1.0 / fps, loop=0)
            print(f"[OK] GIF export enabled: {gif_path}")

        for index, bscan_slice in enumerate(volume):
            curves = get_segmentation_curves(
                segmentation_surfaces,
                coordinates,
                index,
                bscan_slice.shape[1],
                orientation=segmentation_orientation,
            )
            frame = create_export_frame(
                fundus=fundus,
                bscan_slice=bscan_slice,
                coords=coordinates[index],
                angle=angles[index],
                curves=curves,
                slice_idx=index,
                total_slices=len(volume),
            )
            if gif_writer is not None:
                gif_writer.append_data(frame)
            if "tiff" in formats:
                tiff_frames.append(frame)

        if "tiff" in formats:
            tiff_path = os.path.join(output_dir, f"{basename}.tiff")
            tifffile.imwrite(tiff_path, np.stack(tiff_frames, axis=0), photometric="rgb")
            print(f"[OK] TIFF exported: {tiff_path}")
    finally:
        if gif_writer is not None:
            gif_writer.close()


def create_export_frame(fundus, bscan_slice, coords, angle, curves, slice_idx, total_slices):
    import matplotlib.pyplot as plt

    fig, (ax_fundus, ax_bscan) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.03, right=0.97)

    display_fundus = ensure_rgb(fundus)
    if fundus.ndim == 2:
        ax_fundus.imshow(display_fundus[..., 0], cmap="gray", origin="upper")
    else:
        ax_fundus.imshow(display_fundus, origin="upper")
    ax_fundus.axis("off")
    ax_fundus.set_title("Fundus")

    coords = np.asarray(coords, dtype=float)
    if coords.size == 4 and np.any(coords):
        ax_fundus.plot(
            [coords[0], coords[2]],
            [coords[1], coords[3]],
            color="red",
            linewidth=2,
        )

    display_bscan = normalize_to_uint8(bscan_slice)
    ax_bscan.imshow(display_bscan, cmap="gray", origin="upper")
    ax_bscan.axis("off")
    ax_bscan.set_title(f"B-scan slice {slice_idx}, angle {angle:.2f} deg")

    for layer_index, curve in enumerate(curves):
        ax_bscan.plot(
            np.arange(len(curve)),
            curve,
            color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
            linewidth=1.1,
            alpha=0.9,
        )

    fig.suptitle(f"Slice {slice_idx + 1}/{total_slices}")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def show_interactive_viewer(
    volume,
    fundus,
    coordinates,
    angles,
    segmentation_surfaces,
    segmentation_orientation,
):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, (ax_fundus, ax_bscan) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    display_fundus = ensure_rgb(fundus)
    if fundus.ndim == 2:
        ax_fundus.imshow(display_fundus[..., 0], cmap="gray", origin="upper")
    else:
        ax_fundus.imshow(display_fundus, origin="upper")
    ax_fundus.axis("off")

    slice_idx = 0
    display_bscan = normalize_to_uint8(volume[slice_idx])
    image_bscan = ax_bscan.imshow(display_bscan, cmap="gray", origin="upper")
    ax_bscan.set_title(f"B-scan slice {slice_idx}, angle {angles[slice_idx]:.2f} deg")
    ax_bscan.axis("off")

    initial_curves = get_segmentation_curves(
        segmentation_surfaces,
        coordinates,
        slice_idx,
        volume[slice_idx].shape[1],
        orientation=segmentation_orientation,
    )
    contour_lines = []
    for layer_index, curve in enumerate(initial_curves):
        contour_line = ax_bscan.plot(
            np.arange(len(curve)),
            curve,
            color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
            linewidth=1.1,
            alpha=0.9,
        )[0]
        contour_lines.append(contour_line)

    initial_coords = coordinates[slice_idx]
    line = ax_fundus.plot(
        [initial_coords[0], initial_coords[2]],
        [initial_coords[1], initial_coords[3]],
        color="red",
        linewidth=2,
    )[0]

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, len(volume) - 1, valinit=slice_idx, valstep=1)

    def update(_):
        index = int(slider.val)
        image_bscan.set_data(normalize_to_uint8(volume[index]))
        ax_bscan.set_title(f"B-scan slice {index}, angle {angles[index]:.2f} deg")
        line.set_data(
            [coordinates[index][0], coordinates[index][2]],
            [coordinates[index][1], coordinates[index][3]],
        )

        curves = get_segmentation_curves(
            segmentation_surfaces,
            coordinates,
            index,
            volume[index].shape[1],
            orientation=segmentation_orientation,
        )
        while len(contour_lines) < len(curves):
            layer_index = len(contour_lines)
            contour_line = ax_bscan.plot(
                [],
                [],
                color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
                linewidth=1.1,
                alpha=0.9,
            )[0]
            contour_lines.append(contour_line)

        for layer_index, contour_line in enumerate(contour_lines):
            if layer_index < len(curves):
                curve = curves[layer_index]
                contour_line.set_data(np.arange(len(curve)), curve)
                contour_line.set_visible(True)
            else:
                contour_line.set_visible(False)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    args = parse_args()
    formats = normalize_formats(args.formats)

    (
        volume,
        fundus,
        coordinates,
        angles,
        segmentation_surfaces,
        segmentation_orientation,
        bscan_file,
        _,
        _,
    ) = load_shiwei_data(
        input_dir=args.input_dir,
        bscan_file=args.bscan_file,
        fundus_file=args.fundus_file,
        seg_file=args.seg_file,
    )

    basename = args.basename or Path(bscan_file).stem

    if formats:
        output_dir = args.output_dir or os.path.join(args.input_dir, "exports")
        export_visualization(
            volume=volume,
            fundus=fundus,
            coordinates=coordinates,
            angles=angles,
            segmentation_surfaces=segmentation_surfaces,
            segmentation_orientation=segmentation_orientation,
            formats=formats,
            output_dir=output_dir,
            basename=basename,
            fps=args.fps,
        )

    if not args.no_show:
        show_interactive_viewer(
            volume,
            fundus,
            coordinates,
            angles,
            segmentation_surfaces,
            segmentation_orientation,
        )


if __name__ == "__main__":
    main()
