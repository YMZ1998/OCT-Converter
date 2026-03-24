"""Shiwei dataset loading helpers for the OCT web viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.oct.parse_shiwei_location import (
    load_shiwei_oct_dataset as _load_shiwei_oct_dataset_from_script,
    resolve_input_files as _resolve_input_files,
)


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


def load_shiwei_oct_dataset(path: str | Path) -> dict[str, Any]:
    """Loads a Shiwei dataset and annotates the resolved input directory."""

    dataset_dir = resolve_shiwei_input_dir(path)
    if dataset_dir is None:
        raise FileNotFoundError(
            "Could not locate a Shiwei dataset directory containing the "
            "structural and fundus DICOM files."
        )

    dataset = _load_shiwei_oct_dataset_from_script(str(dataset_dir))
    dataset["input_dir"] = str(dataset_dir)
    return dataset


__all__ = ["load_shiwei_oct_dataset", "resolve_shiwei_input_dir"]
