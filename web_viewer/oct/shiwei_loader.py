"""Shiwei dataset loading helpers for the OCT web viewer."""

from __future__ import annotations

from pathlib import Path

from scripts.oct.parse_shiwei_location import (
    load_shiwei_oct_dataset as _load_shiwei_oct_dataset_from_script,
    resolve_input_files as _resolve_input_files,
)

from .dataset_types import ScanSegment, ShiweiOCTDataset, VendorOverlayData


def _is_shiwei_dataset_dir(directory: Path) -> bool:
    """Returns whether the directory contains a valid Shiwei dataset."""

    try:
        _resolve_input_files(str(directory))
    except FileNotFoundError:
        return False
    return True


def _find_nested_shiwei_dirs(root_dir: Path) -> list[Path]:
    """Finds nested Shiwei dataset directories under a root directory."""

    matches: list[Path] = []
    for candidate in sorted(path for path in root_dir.rglob("*") if path.is_dir()):
        if _is_shiwei_dataset_dir(candidate):
            matches.append(candidate)
    return matches


def resolve_shiwei_input_dir(path: str | Path) -> Path | None:
    """Returns a valid Shiwei dataset directory for the given input path."""

    candidate_path = Path(path)
    candidates: list[Path] = []
    if candidate_path.is_dir():
        candidates.append(candidate_path)
    elif candidate_path.suffix.lower() in {".dcm", ".dicom"}:
        candidates.append(candidate_path.parent)

    for candidate in candidates:
        if _is_shiwei_dataset_dir(candidate):
            return candidate

        nested_matches = _find_nested_shiwei_dirs(candidate)
        if len(nested_matches) == 1:
            return nested_matches[0]
        if len(nested_matches) > 1:
            match_list = ", ".join(str(match) for match in nested_matches[:3])
            if len(nested_matches) > 3:
                match_list = f"{match_list}, ..."
            raise ValueError(
                "Found multiple Shiwei dataset directories under "
                f"'{candidate}': {match_list}. Please choose a more specific directory."
            )
    return None


def _compute_bounds(
    segments: list[ScanSegment],
) -> tuple[float, float, float, float] | None:
    if not segments:
        return None
    xs = [point[0] for segment in segments for point in segment]
    ys = [point[1] for segment in segments for point in segment]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def load_shiwei_oct_dataset(path: str | Path) -> ShiweiOCTDataset:
    """Loads a Shiwei dataset and converts it into a typed dataset object."""

    dataset_dir = resolve_shiwei_input_dir(path)
    if dataset_dir is None:
        raise FileNotFoundError(
            "Could not locate a Shiwei dataset directory containing the "
            "structural and fundus DICOM files."
        )

    dataset = _load_shiwei_oct_dataset_from_script(str(dataset_dir))
    segments = list(dataset.get("segments") or [])
    warning = ""
    if not segments:
        warning = "Shiwei frame location metadata unavailable; overlay falls back to fundus only."

    return ShiweiOCTDataset(
        input_dir=str(dataset_dir),
        volume=dataset["volume"],
        fundus=dataset["fundus"],
        coordinates=list(dataset.get("coordinates") or []),
        angles=[float(angle) for angle in (dataset.get("angles") or [])],
        segments=segments,
        segmentation_surfaces=dataset.get("segmentation_surfaces"),
        segmentation_orientation=dataset.get("segmentation_orientation"),
        bscan_file=dataset.get("bscan_file"),
        fundus_file=dataset.get("fundus_file"),
        seg_file=dataset.get("seg_file"),
        overlay_entries=[
            VendorOverlayData(
                matched_fundus_index=0,
                matched_fundus_label=getattr(dataset["fundus"], "image_id", None)
                or "Shiwei fundus",
                fundus_match_mode="shiwei-directory",
                overlay_mode="shiwei-metadata",
                projection_mode="shiwei-metadata",
                localizer_mode="ophthalmic-frame-location-sequence",
                warning=warning,
                scan_segments=segments,
                bounds=_compute_bounds(segments),
            )
        ],
    )


__all__ = ["load_shiwei_oct_dataset", "resolve_shiwei_input_dir"]
