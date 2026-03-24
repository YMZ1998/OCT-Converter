"""Tupai dataset loading helpers for the OCT web viewer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData
from scripts.parse_tupai_location import get_pixel_spacing, load_tupai_data

_TUPAI_OCT_FILENAME = "OCT.dcm"
_TUPAI_FUNDUS_FILENAME = "Fundus.dcm"


def _find_case_insensitive_file(directory: Path, target_name: str) -> Path | None:
    """Finds a file in a directory using a case-insensitive filename match."""

    target_lower = target_name.lower()
    for child in directory.iterdir():
        if child.is_file() and child.name.lower() == target_lower:
            return child
    return None


def _is_tupai_dataset_dir(directory: Path) -> bool:
    """Returns whether the directory contains the required Tupai files."""

    oct_file = _find_case_insensitive_file(directory, _TUPAI_OCT_FILENAME)
    fundus_file = _find_case_insensitive_file(directory, _TUPAI_FUNDUS_FILENAME)
    return oct_file is not None and fundus_file is not None


def _find_nested_tupai_dirs(root_dir: Path) -> list[Path]:
    """Finds nested Tupai dataset directories under a root directory."""

    matches: list[Path] = []
    for candidate in sorted(path for path in root_dir.rglob("*") if path.is_dir()):
        if _is_tupai_dataset_dir(candidate):
            matches.append(candidate)
    return matches


def resolve_tupai_input_dir(path: str | Path) -> Path | None:
    """Returns a valid Tupai dataset directory for the given input path."""

    candidate_path = Path(path)
    candidates: list[Path] = []
    if candidate_path.is_dir():
        candidates.append(candidate_path)
    elif candidate_path.suffix.lower() in {".dcm", ".dicom"}:
        candidates.append(candidate_path.parent)

    for candidate in candidates:
        if _is_tupai_dataset_dir(candidate):
            return candidate

        nested_matches = _find_nested_tupai_dirs(candidate)
        if len(nested_matches) == 1:
            return nested_matches[0]
        if len(nested_matches) > 1:
            match_list = ", ".join(str(match) for match in nested_matches[:3])
            if len(nested_matches) > 3:
                match_list = f"{match_list}, ..."
            raise ValueError(
                "Found multiple Tupai dataset directories under "
                f"'{candidate}': {match_list}. Please choose a more specific directory."
            )
    return None


def _extract_laterality(dataset: Any) -> str | None:
    """Extracts laterality from a DICOM dataset."""

    for attr in ("ImageLaterality", "Laterality"):
        value = getattr(dataset, attr, None)
        if value:
            return str(value)
    return None


def _parse_acquisition_datetime(dataset: Any) -> datetime | None:
    """Parses acquisition date and time from a DICOM dataset."""

    date_value = getattr(dataset, "AcquisitionDate", None) or getattr(
        dataset, "ContentDate", None
    )
    time_value = getattr(dataset, "AcquisitionTime", None) or getattr(
        dataset, "ContentTime", None
    )
    if not date_value:
        return None

    date_text = str(date_value).strip()
    time_text = str(time_value or "000000").strip().split(".")[0]
    time_text = "".join(character for character in time_text if character.isdigit())
    if len(date_text) != 8:
        return None
    time_text = (time_text + "000000")[:6]
    try:
        return datetime.strptime(f"{date_text}{time_text}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _sanitize_segments(
    coordinates: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Converts raw coordinate arrays into line segments."""

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in np.asarray(coordinates, dtype=float):
        if row.shape[0] < 4 or not np.isfinite(row[:4]).all():
            continue
        segments.append(
            (
                (float(row[0]), float(row[1])),
                (float(row[2]), float(row[3])),
            )
        )
    return segments


def load_tupai_oct_dataset(path: str | Path) -> dict[str, Any]:
    """Loads a Tupai dataset into viewer-friendly OCT and fundus objects."""

    dataset_dir = resolve_tupai_input_dir(path)
    if dataset_dir is None:
        raise FileNotFoundError(
            "Could not locate a Tupai dataset directory containing OCT.dcm "
            "and Fundus.dcm."
        )

    (
        volume,
        fundus,
        full_coordinates,
        coordinate_mode,
        segment_coordinates,
        segment_mode,
        scan_band_width_pixels,
    ) = load_tupai_data(str(dataset_dir))

    oct_file = _find_case_insensitive_file(dataset_dir, _TUPAI_OCT_FILENAME)
    fundus_file = _find_case_insensitive_file(dataset_dir, _TUPAI_FUNDUS_FILENAME)
    if oct_file is None or fundus_file is None:
        raise FileNotFoundError(
            "Missing OCT.dcm or Fundus.dcm in Tupai dataset directory."
        )

    ds_bscan = pydicom.dcmread(str(oct_file), stop_before_pixels=True)
    ds_fundus = pydicom.dcmread(str(fundus_file), stop_before_pixels=True)

    volume_array = np.asarray(volume)
    if volume_array.ndim < 3:
        raise ValueError("Unexpected Tupai OCT volume shape.")
    volume_slices = [
        np.asarray(volume_array[index]) for index in range(volume_array.shape[0])
    ]

    bscan_spacing = get_pixel_spacing(ds_bscan)
    fundus_spacing = get_pixel_spacing(ds_fundus)
    pixel_spacing = None
    if bscan_spacing is not None:
        pixel_spacing = [
            float(bscan_spacing[1]),
            float(bscan_spacing[0]),
            float(bscan_spacing[0]),
        ]

    laterality = _extract_laterality(ds_bscan) or _extract_laterality(ds_fundus)
    patient_id = getattr(ds_bscan, "PatientID", None) or getattr(
        ds_fundus, "PatientID", None
    )
    acquisition_date = _parse_acquisition_datetime(ds_bscan)

    oct_volume = OCTVolumeWithMetaData(
        volume=volume_slices,
        patient_id=patient_id,
        acquisition_date=acquisition_date,
        laterality=laterality,
        pixel_spacing=pixel_spacing,
        metadata={
            "dataset_dir": str(dataset_dir),
            "coordinate_mode": coordinate_mode,
            "segment_mode": segment_mode,
            "scan_band_width_pixels": float(scan_band_width_pixels),
            "dicom": {
                "oct_file": str(oct_file),
                "fundus_file": str(fundus_file),
            },
        },
        header={"source": "Tupai"},
        oct_header={
            "number_slices": int(volume_array.shape[0]),
            "width": int(volume_array.shape[2]),
            "height": int(volume_array.shape[1]),
        },
    )

    fundus_image = FundusImageWithMetaData(
        image=np.asarray(fundus),
        laterality=laterality,
        patient_id=patient_id,
        image_id=fundus_file.stem,
        metadata={
            "dataset_dir": str(dataset_dir),
            "source": "Tupai",
            "oct_file": str(oct_file),
            "fundus_file": str(fundus_file),
        },
        pixel_spacing=(
            [float(fundus_spacing[1]), float(fundus_spacing[0])]
            if fundus_spacing is not None
            else None
        ),
    )

    return {
        "dataset_dir": str(dataset_dir),
        "volume": oct_volume,
        "fundus": fundus_image,
        "segments": _sanitize_segments(full_coordinates),
        "segment_coordinates": _sanitize_segments(segment_coordinates),
        "coordinate_mode": coordinate_mode,
        "segment_mode": segment_mode,
        "scan_band_width_pixels": float(scan_band_width_pixels),
    }


__all__ = ["load_tupai_oct_dataset", "resolve_tupai_input_dir"]
