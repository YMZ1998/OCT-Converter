from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from topcon_fa_qt import (
    DEFAULT_INPUT_DIR as TOPCON_DEFAULT_INPUT_DIR,
    load_topcon_fa_dataset,
)
from hdb_fa_parser import (
    laterality_to_chinese as hdb_laterality_display_text,
    load_heidelberg_fa_dataset,
)
from zeiss_fa_parser import (
    DEFAULT_INPUT_PATH as ZEISS_DEFAULT_INPUT_PATH,
    DICOM_SUFFIXES,
    discover_candidate_files,
    safe_dcmread,
)
from zeiss_fa_qt import (
    build_fa_viewer_tracks,
    clean_text,
    laterality_display_text,
    load_zeiss_fa_series,
    normalize_to_uint8,
    parse_iso_datetime,
    viewer_frame_sort_key,
)


VENDOR_AUTO = "auto"
VENDOR_TOPCON = "topcon"
VENDOR_ZEISS = "zeiss"
VENDOR_HDB = "hdb"
VENDOR_CFP = "cfp"
TOPCON_IMAGE_GLOB = "IM*.JPG"
HDB_E2E_GLOB = "*.E2E"
CFP_IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CFP_FRAME_PATTERN = re.compile(r"(?:^|[-_])(?P<label>RF|F(?P<number>\d{1,2}))$", re.IGNORECASE)
CFP_LATERALITY_PATTERN = re.compile(r"(?:^|[-_ ])(?P<eye>OD|OS|R|L)(?=$|[-_ .])", re.IGNORECASE)
CFP_DATE_INLINE_PATTERN = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
CFP_DATE_SEPARATOR_PATTERN = re.compile(r"(?<!\d)(20\d{2})[-_/\.](\d{1,2})[-_/\.](\d{1,2})(?!\d)")


@dataclass
class UnifiedFAStudyInfo:
    vendor: str
    vendor_label: str
    patient_name: str = ""
    patient_id: str = ""
    sex: str = ""
    birth_date: str = ""
    exam_date: str = ""
    laterality: str = ""
    study_code: str = ""
    device_model: str = ""


@dataclass
class UnifiedFAFrame:
    order_index: int
    vendor: str
    filename: str
    source_path: Path | None
    image_array: np.ndarray | None
    group_key: str
    group_label: str
    laterality_key: str
    laterality_label: str
    label: str
    source_detail: str
    acquisition_datetime: datetime | None
    acquisition_display: str
    elapsed_seconds: float | None
    width: int | None
    height: int | None
    elapsed_display_override: str | None = None
    is_proofsheet: bool = False

    @property
    def elapsed_display(self) -> str:
        if self.elapsed_display_override:
            return self.elapsed_display_override
        if self.elapsed_seconds is None:
            return "-"
        return format_elapsed_clock(self.elapsed_seconds)

    @property
    def size_display(self) -> str:
        if self.width and self.height:
            return f"{self.width} x {self.height}"
        return "-"


@dataclass
class UnifiedFADataset:
    vendor: str
    vendor_label: str
    input_path: Path
    study_info: UnifiedFAStudyInfo
    frames: list[UnifiedFAFrame]


def default_input_path() -> Path | None:
    if TOPCON_DEFAULT_INPUT_DIR.exists():
        return TOPCON_DEFAULT_INPUT_DIR
    if ZEISS_DEFAULT_INPUT_PATH.exists():
        return ZEISS_DEFAULT_INPUT_PATH
    if HDB_DEFAULT_INPUT_PATH.exists():
        return HDB_DEFAULT_INPUT_PATH
    return None


def parse_args() -> argparse.Namespace:
    default_path = default_input_path()
    parser = argparse.ArgumentParser(description="Unified Topcon / Zeiss / HDB / CFP FA Qt viewer.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(default_path) if default_path else None,
        help="Topcon FA folder, Zeiss FA folder/DICOM file, HDB/Heidelberg E2E file, or CFP image folder.",
    )
    parser.add_argument(
        "--vendor",
        choices=(VENDOR_AUTO, VENDOR_TOPCON, VENDOR_ZEISS, VENDOR_HDB, VENDOR_CFP),
        default=VENDOR_AUTO,
        help="Force the dataset vendor instead of auto-detecting it.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Parse and print frame info without opening the Qt window.",
    )
    return parser.parse_args()


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def summarize_values(values: list[str], *, fallback: str = "-", max_items: int = 4) -> str:
    cleaned = unique_preserve_order([value.strip() for value in values if value and value.strip()])
    if not cleaned:
        return fallback
    if len(cleaned) <= max_items:
        return " / ".join(cleaned)
    return f"{' / '.join(cleaned[:max_items])} / +{len(cleaned) - max_items} more"


def format_elapsed_clock(seconds: float | None) -> str:
    if seconds is None:
        return "-"

    total_centiseconds = max(0, int(round(float(seconds) * 100)))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def format_time_range(frames: list[UnifiedFAFrame]) -> str:
    datetimes = [frame.acquisition_datetime for frame in frames if frame.acquisition_datetime is not None]
    if datetimes:
        earliest = min(datetimes)
        latest = max(datetimes)
        return f"{earliest:%Y-%m-%d %H:%M:%S} ~ {latest:%Y-%m-%d %H:%M:%S}"

    elapsed_seconds = [frame.elapsed_seconds for frame in frames if frame.elapsed_seconds is not None]
    if elapsed_seconds:
        return f"{format_elapsed_clock(min(elapsed_seconds))} ~ {format_elapsed_clock(max(elapsed_seconds))}"
    return "-"


def summarize_groups(frames: list[UnifiedFAFrame]) -> str:
    counts = Counter(frame.group_label for frame in frames)
    return ", ".join(f"{group} {count}" for group, count in counts.items()) or "-"


def rebase_elapsed_seconds(frames: list[UnifiedFAFrame]) -> None:
    first_datetime = min(
        (frame.acquisition_datetime for frame in frames if frame.acquisition_datetime is not None),
        default=None,
    )
    first_elapsed = min(
        (frame.elapsed_seconds for frame in frames if frame.elapsed_seconds is not None),
        default=None,
    )

    for frame in frames:
        elapsed_seconds: float | None = None
        if frame.acquisition_datetime is not None and first_datetime is not None:
            elapsed_seconds = (frame.acquisition_datetime - first_datetime).total_seconds()
        elif frame.elapsed_seconds is not None and first_elapsed is not None:
            elapsed_seconds = frame.elapsed_seconds - first_elapsed

        frame.elapsed_seconds = None if elapsed_seconds is None else max(0.0, elapsed_seconds)


def _is_cfp_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in CFP_IMAGE_SUFFIXES


def _find_cfp_images(path: Path) -> list[Path]:
    if _is_cfp_image(path):
        return [path.resolve()]
    if not path.exists() or not path.is_dir():
        return []
    images = [candidate for candidate in path.rglob("*") if _is_cfp_image(candidate)]
    return sorted(images, key=lambda candidate: str(candidate).lower())


def _extract_cfp_laterality(*parts: str) -> str:
    for part in parts:
        match = CFP_LATERALITY_PATTERN.search(part)
        if not match:
            continue
        eye = match.group("eye").upper()
        if eye in {"OD", "R"}:
            return "OD"
        if eye in {"OS", "L"}:
            return "OS"
    return "UNKNOWN"


def _parse_cfp_frame_label(stem: str) -> tuple[str, tuple[int, int]]:
    match = CFP_FRAME_PATTERN.search(stem)
    if not match:
        return "-", (2, 0)

    label = match.group("label").upper()
    number_text = match.group("number")
    if number_text:
        return label, (0, int(number_text))
    return label, (1, 0)


def _extract_cfp_study_code(stem: str) -> str:
    without_frame = re.sub(r"[-_](RF|F\d{1,2})$", "", stem, flags=re.IGNORECASE)
    without_eye = re.sub(r"[-_](OD|OS|R|L)$", "", without_frame, flags=re.IGNORECASE)
    return without_eye.strip("-_ ")


def _read_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception:
        return None, None


def _parse_cfp_datetime_text(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None

    for format_text in (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y:%m:%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, format_text)
        except ValueError:
            continue
    return None


def _extract_cfp_datetime_from_metadata(image_path: Path) -> datetime | None:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif() if hasattr(image, "getexif") else None
            if exif:
                for tag_code in (36867, 36868, 306):
                    raw_value = exif.get(tag_code)
                    if raw_value is None:
                        continue
                    parsed = _parse_cfp_datetime_text(str(raw_value))
                    if parsed is not None:
                        return parsed

            for key in ("DateTimeOriginal", "DateTime", "date:create", "date:modify"):
                raw_value = image.info.get(key)
                if raw_value is None:
                    continue
                parsed = _parse_cfp_datetime_text(str(raw_value))
                if parsed is not None:
                    return parsed
    except Exception:
        return None

    return None


def _extract_cfp_datetime_from_path(path: Path) -> datetime | None:
    candidates = [path.stem] + [part for part in path.parts]
    for text in candidates:
        match = CFP_DATE_SEPARATOR_PATTERN.search(text)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                pass

        match = CFP_DATE_INLINE_PATTERN.search(text)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                pass
    return None


def _extract_cfp_acquisition_datetime(image_path: Path) -> datetime | None:
    metadata_datetime = _extract_cfp_datetime_from_metadata(image_path)
    if metadata_datetime is not None:
        return metadata_datetime
    return _extract_cfp_datetime_from_path(image_path)


def detect_vendor(path: Path) -> str | None:
    container = path if path.is_dir() else path.parent

    if path.is_file():
        if path.name.upper() == "DATAFILE":
            return VENDOR_TOPCON
        if path.match(TOPCON_IMAGE_GLOB) and (container / "DATAFILE").exists():
            return VENDOR_TOPCON
        if path.suffix.lower() == ".e2e":
            return VENDOR_HDB
        if path.suffix.lower() in DICOM_SUFFIXES or path.name.upper() == "DICOMDIR":
            return VENDOR_ZEISS
        if _is_cfp_image(path):
            return VENDOR_CFP

    if container.exists():
        if (container / "DATAFILE").exists():
            return VENDOR_TOPCON
        if any(container.glob(TOPCON_IMAGE_GLOB)):
            return VENDOR_TOPCON
        if any(container.glob(HDB_E2E_GLOB)) or any(container.glob("*.e2e")):
            return VENDOR_HDB

        search_root = container / "DataFiles" if (container / "DataFiles").is_dir() else container
        for candidate in search_root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() == ".e2e":
                return VENDOR_HDB
            if candidate.is_file() and (
                candidate.suffix.lower() in DICOM_SUFFIXES or candidate.name.upper() == "DICOMDIR"
            ):
                return VENDOR_ZEISS
        if _find_cfp_images(container):
            return VENDOR_CFP

    return None


def build_zeiss_device_label(input_path: Path) -> str:
    try:
        for candidate in discover_candidate_files(input_path):
            try:
                dataset = safe_dcmread(candidate, stop_before_pixels=True)
            except Exception:
                continue

            parts = [
                clean_text(getattr(dataset, "Manufacturer", "")),
                clean_text(getattr(dataset, "ManufacturerModelName", "")),
                clean_text(getattr(dataset, "StationName", "")),
            ]
            label = " / ".join(part for part in parts if part)
            if label:
                return label
    except Exception:
        pass

    return "Zeiss DICOM"


def build_zeiss_acquisition_display(frame) -> str:
    if frame.acquisition_datetime_iso:
        return frame.acquisition_datetime_iso.replace("T", " ")
    if frame.acquisition_time_iso:
        return frame.acquisition_time_iso
    if frame.acquisition_datetime_raw:
        return frame.acquisition_datetime_raw
    if frame.relative_time_seconds is not None:
        return f"+{frame.relative_time_seconds:.3f} s"
    return "-"


def load_topcon_dataset(input_path: Path) -> UnifiedFADataset:
    input_dir, study_info, frames = load_topcon_fa_dataset(str(input_path))
    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError(input_path)
    if not frames:
        raise RuntimeError(f"No Topcon FA frames found in {input_dir}")

    unified_frames = [
        UnifiedFAFrame(
            order_index=frame.order_index,
            vendor=VENDOR_TOPCON,
            filename=frame.filename,
            source_path=frame.image_path,
            image_array=None,
            group_key=frame.modality or "Unknown",
            group_label=frame.modality or "Unknown",
            laterality_key=clean_text(study_info.laterality).upper() or "UNKNOWN",
            laterality_label=study_info.laterality_display,
            label=frame.label or "-",
            source_detail=frame.device or study_info.device_model or "Topcon FA",
            acquisition_datetime=frame.acquisition_datetime,
            acquisition_display=frame.acquisition_display,
            elapsed_seconds=frame.elapsed_seconds,
            width=frame.width,
            height=frame.height,
            is_proofsheet=frame.is_proofsheet,
        )
        for frame in frames
    ]

    normalized_study = UnifiedFAStudyInfo(
        vendor=VENDOR_TOPCON,
        vendor_label="Topcon",
        patient_name=study_info.patient_name,
        patient_id=study_info.patient_id,
        sex=study_info.sex_display,
        birth_date=study_info.birth_date,
        exam_date=study_info.exam_date_display,
        laterality=study_info.laterality_display,
        study_code=study_info.study_code or study_info.extra_value,
        device_model=study_info.device_model or "Topcon FA",
    )

    rebase_elapsed_seconds(unified_frames)

    return UnifiedFADataset(
        vendor=VENDOR_TOPCON,
        vendor_label="Topcon",
        input_path=input_dir.resolve(),
        study_info=normalized_study,
        frames=unified_frames,
    )


def _compute_hdb_elapsed_seconds(
    frame,
    *,
    first_datetime: datetime | None,
    first_time_of_day_ms: int | None,
    first_raw_elapsed_ms: int | None,
    first_raw_elapsed_ms_alt: int | None,
    prefer_relative_time: bool = False,
) -> float | None:
    if prefer_relative_time and frame.relative_time_seconds is not None:
        return frame.relative_time_seconds
    if frame.acquisition_datetime_local is not None and first_datetime is not None:
        return (frame.acquisition_datetime_local - first_datetime).total_seconds()
    if frame.time_of_day_ms is not None and first_time_of_day_ms is not None:
        return (frame.time_of_day_ms - first_time_of_day_ms) / 1000.0
    if frame.raw_elapsed_ms is not None and first_raw_elapsed_ms is not None:
        return (frame.raw_elapsed_ms - first_raw_elapsed_ms) / 1000.0
    if frame.raw_elapsed_ms_alt is not None and first_raw_elapsed_ms_alt is not None:
        return (frame.raw_elapsed_ms_alt - first_raw_elapsed_ms_alt) / 1000.0
    return None


def load_hdb_dataset(input_path: Path, *, relative_time_mode: str = "default") -> UnifiedFADataset:
    input_file, study_info, frames = load_heidelberg_fa_dataset(str(input_path))
    if input_file is None or not input_file.exists():
        raise FileNotFoundError(input_path)
    if not frames:
        raise RuntimeError(f"No HDB / Heidelberg FA frames found in {input_file}")

    first_datetime = min(
        (frame.acquisition_datetime_local for frame in frames if frame.acquisition_datetime_local is not None),
        default=None,
    )
    first_time_of_day_ms = min(
        (frame.time_of_day_ms for frame in frames if frame.time_of_day_ms is not None),
        default=None,
    )
    first_raw_elapsed_ms = min(
        (frame.raw_elapsed_ms for frame in frames if frame.raw_elapsed_ms is not None),
        default=None,
    )
    first_raw_elapsed_ms_alt = min(
        (frame.raw_elapsed_ms_alt for frame in frames if frame.raw_elapsed_ms_alt is not None),
        default=None,
    )
    use_regex_relative_time = relative_time_mode == "parser3" and any(
        frame.relative_time_seconds is not None for frame in frames
    )

    unified_frames: list[UnifiedFAFrame] = []
    for frame in frames:
        modality_text = clean_text(frame.modality) or "Unknown"
        laterality_text = frame.laterality_display or hdb_laterality_display_text(frame.laterality)
        group_label = f"{modality_text} | {laterality_text}"
        label_parts = unique_preserve_order(
            [
                frame.modality_display,
                frame.structure_display,
            ]
        )
        label = " | ".join(label_parts) if label_parts else group_label
        source_parts = [
            f"Series {frame.series_id}",
            f"Slice {frame.slice_id}" if frame.slice_id >= 0 else "Slice -",
            frame.acquisition_source or "",
        ]
        # print(frame.time_display)
        unified_frames.append(
            UnifiedFAFrame(
                order_index=frame.order_index,
                vendor=VENDOR_HDB,
                filename=frame.image_id,
                source_path=input_file.resolve(),
                image_array=np.asarray(frame.image),
                group_key=f"{modality_text}:{clean_text(frame.laterality).upper() or 'UNKNOWN'}",
                group_label=group_label,
                laterality_key=clean_text(frame.laterality).upper() or "UNKNOWN",
                laterality_label=laterality_text,
                label=label,
                source_detail=" | ".join(part for part in source_parts if part),
                acquisition_datetime=frame.acquisition_datetime_local,
                acquisition_display=frame.time_display,
                elapsed_seconds=_compute_hdb_elapsed_seconds(
                    frame,
                    first_datetime=first_datetime,
                    first_time_of_day_ms=first_time_of_day_ms,
                    first_raw_elapsed_ms=first_raw_elapsed_ms,
                    first_raw_elapsed_ms_alt=first_raw_elapsed_ms_alt,
                    prefer_relative_time=use_regex_relative_time,
                ),
                width=frame.width,
                height=frame.height,
                elapsed_display_override=frame.relative_time_display if use_regex_relative_time else None,
            )
        )

    exam_date = "-"
    if study_info.study_datetime_local is not None:
        exam_date = study_info.study_datetime_local.strftime("%Y-%m-%d")
    elif first_datetime is not None:
        exam_date = first_datetime.strftime("%Y-%m-%d")

    if not use_regex_relative_time:
        rebase_elapsed_seconds(unified_frames)

    normalized_study = UnifiedFAStudyInfo(
        vendor=VENDOR_HDB,
        vendor_label="HDB",
        patient_name=study_info.patient_name,
        patient_id=study_info.patient_id,
        sex=study_info.sex_display,
        birth_date=study_info.birth_date,
        exam_date=exam_date,
        laterality=study_info.laterality_summary or summarize_values(
            [frame.laterality_display for frame in frames],
            fallback="-",
        ),
        study_code=study_info.modality_summary or "HDB FA E2E",
        device_model=study_info.device_display,
    )

    return UnifiedFADataset(
        vendor=VENDOR_HDB,
        vendor_label="HDB",
        input_path=input_file.resolve(),
        study_info=normalized_study,
        frames=unified_frames,
    )


def load_zeiss_dataset(input_path: Path) -> UnifiedFADataset:
    series_list = load_zeiss_fa_series(input_path, prefer_fa_only=True)
    if not series_list:
        raise RuntimeError(f"No Zeiss FA series found in {input_path}")

    elapsed_lookup: dict[tuple[str, str, int, str], float | None] = {}
    for track in build_fa_viewer_tracks(series_list):
        if clean_text(track.key).upper() == "ALL":
            continue
        for frame in track.frames:
            key = (
                clean_text(track.laterality).upper() or "UNKNOWN",
                frame.source_file,
                frame.frame_index,
                frame.series_uid,
            )
            elapsed_lookup[key] = frame.elapsed_seconds

    flattened_items: list[tuple[object, object]] = []
    for series in series_list:
        for frame in series.frames:
            flattened_items.append((series, frame))

    flattened_items.sort(key=lambda item: viewer_frame_sort_key(item[0], item[1]))

    device_label = build_zeiss_device_label(input_path)
    unified_frames: list[UnifiedFAFrame] = []

    for order_index, (series, frame) in enumerate(flattened_items):
        laterality_key = clean_text(series.laterality).upper() or "UNKNOWN"
        group_label = laterality_display_text(laterality_key)
        file_lookup = {path.name: path for path in series.files}
        source_path = file_lookup.get(frame.source_file)
        label_parts = unique_preserve_order(
            [
                clean_text(series.series_description),
                clean_text(series.protocol_name),
                clean_text(series.study_description),
            ]
        )
        label = " | ".join(label_parts) if label_parts else (
            f"Series {series.series_number}" if series.series_number is not None else "Zeiss FA"
        )

        image = np.asarray(frame.image)
        height = int(image.shape[0]) if image.ndim >= 2 else None
        width = int(image.shape[1]) if image.ndim >= 2 else None
        elapsed_key = (
            laterality_key,
            frame.source_file,
            frame.frame_index,
            series.series_uid,
        )

        unified_frames.append(
            UnifiedFAFrame(
                order_index=order_index,
                vendor=VENDOR_ZEISS,
                filename=frame.source_file,
                source_path=source_path.resolve() if source_path else None,
                image_array=image,
                group_key=laterality_key,
                group_label=group_label,
                laterality_key=laterality_key,
                laterality_label=group_label,
                label=label,
                source_detail=(
                    f"{frame.source_file} | Series {series.series_number if series.series_number is not None else '-'}"
                ),
                acquisition_datetime=parse_iso_datetime(frame.acquisition_datetime_iso),
                acquisition_display=build_zeiss_acquisition_display(frame),
                elapsed_seconds=elapsed_lookup.get(elapsed_key, frame.relative_time_seconds),
                width=width,
                height=height,
            )
        )

    first_series = series_list[0]
    laterality_summary = summarize_values(
        [
            laterality_display_text(clean_text(series.laterality).upper() or "UNKNOWN")
            for series in series_list
        ],
        fallback="Unknown",
    )
    study_code = summarize_values(
        [
            clean_text(series.series_description),
            clean_text(series.protocol_name),
            clean_text(series.study_description),
        ],
        fallback="Zeiss FA",
    )
    exam_date = "-"
    datetimes = [frame.acquisition_datetime for frame in unified_frames if frame.acquisition_datetime is not None]
    if datetimes:
        exam_date = min(datetimes).strftime("%Y-%m-%d")
    elif first_series.study_datetime_iso:
        exam_date = first_series.study_datetime_iso.split("T")[0]
    elif first_series.series_datetime_iso:
        exam_date = first_series.series_datetime_iso.split("T")[0]

    rebase_elapsed_seconds(unified_frames)

    normalized_study = UnifiedFAStudyInfo(
        vendor=VENDOR_ZEISS,
        vendor_label="Zeiss",
        patient_name=first_series.patient_name,
        patient_id=first_series.patient_id,
        sex=first_series.patient_sex,
        birth_date=first_series.patient_birth_date_iso or first_series.patient_birth_date_raw,
        exam_date=exam_date,
        laterality=laterality_summary,
        study_code=study_code,
        device_model=device_label,
    )

    return UnifiedFADataset(
        vendor=VENDOR_ZEISS,
        vendor_label="Zeiss",
        input_path=input_path.resolve(),
        study_info=normalized_study,
        frames=unified_frames,
    )


def load_cfp_dataset(input_path: Path) -> UnifiedFADataset:
    search_root = input_path if input_path.is_dir() else input_path.parent
    images = _find_cfp_images(input_path if input_path.is_file() else search_root)
    if not images:
        raise RuntimeError(f"No CFP image files found in {input_path}")

    frame_entries: list[tuple[tuple[int, str], tuple[int, int], datetime | None, Path, str, str, str, str]] = []
    study_codes: list[str] = []

    for image_path in images:
        stem = image_path.stem
        laterality_key = _extract_cfp_laterality(stem, image_path.parent.name, search_root.name)
        group_label = laterality_display_text(laterality_key)
        label, frame_rank = _parse_cfp_frame_label(stem)
        acquisition_datetime = _extract_cfp_acquisition_datetime(image_path)
        source_detail = image_path.parent.name or str(image_path.parent)
        study_code = _extract_cfp_study_code(stem)
        if study_code:
            study_codes.append(study_code)

        sort_laterality = {"OD": 0, "OS": 1, "UNKNOWN": 2}.get(laterality_key, 99)
        frame_entries.append(
            (
                (sort_laterality, laterality_key),
                frame_rank,
                acquisition_datetime,
                image_path.resolve(),
                laterality_key,
                group_label,
                label,
                source_detail,
            )
        )

    frame_entries.sort(
        key=lambda item: (
            item[0][0],
            item[1][0],
            item[1][1],
            item[2] or datetime.min,
            item[3].name.lower(),
        )
    )

    first_by_eye: dict[str, datetime] = {}
    for _, _, acquisition_datetime, _, laterality_key, _, _, _ in frame_entries:
        if acquisition_datetime is None:
            continue
        previous = first_by_eye.get(laterality_key)
        if previous is None or acquisition_datetime < previous:
            first_by_eye[laterality_key] = acquisition_datetime

    unified_frames: list[UnifiedFAFrame] = []
    for order_index, (
        _laterality_sort,
        _frame_rank,
        acquisition_datetime,
        image_path,
        laterality_key,
        group_label,
        label,
        source_detail,
    ) in enumerate(frame_entries):
        width, height = _read_image_size(image_path)
        first_eye_time = first_by_eye.get(laterality_key)
        elapsed_seconds: float | None = None
        if acquisition_datetime is not None and first_eye_time is not None:
            elapsed_seconds = max(0.0, (acquisition_datetime - first_eye_time).total_seconds())
        acquisition_display = acquisition_datetime.strftime("%Y-%m-%d %H:%M:%S") if acquisition_datetime else "-"

        unified_frames.append(
            UnifiedFAFrame(
                order_index=order_index,
                vendor=VENDOR_CFP,
                filename=image_path.name,
                source_path=image_path,
                image_array=None,
                group_key=laterality_key,
                group_label=group_label,
                laterality_key=laterality_key,
                laterality_label=group_label,
                label=label,
                source_detail=source_detail,
                acquisition_datetime=acquisition_datetime,
                acquisition_display=acquisition_display,
                elapsed_seconds=elapsed_seconds,
                width=width,
                height=height,
                is_proofsheet=False,
            )
        )

    exam_date = "-"
    datetimes = [frame.acquisition_datetime for frame in unified_frames if frame.acquisition_datetime is not None]
    if datetimes:
        exam_date = min(datetimes).strftime("%Y-%m-%d")

    study_code = summarize_values(study_codes, fallback=(search_root.name if search_root else "CFP"))
    laterality_summary = summarize_values(
        [frame.laterality_label for frame in unified_frames],
        fallback="Unknown",
    )

    rebase_elapsed_seconds(unified_frames)

    normalized_study = UnifiedFAStudyInfo(
        vendor=VENDOR_CFP,
        vendor_label="CFP",
        patient_name=study_code,
        patient_id=study_code,
        sex="-",
        birth_date="-",
        exam_date=exam_date,
        laterality=laterality_summary,
        study_code=study_code,
        device_model="CFP Image Set",
    )

    return UnifiedFADataset(
        vendor=VENDOR_CFP,
        vendor_label="CFP",
        input_path=(search_root if search_root else input_path).resolve(),
        study_info=normalized_study,
        frames=unified_frames,
    )


def load_unified_dataset(
    input_path: str | Path,
    *,
    vendor: str = VENDOR_AUTO,
    hdb_relative_time_parser: str = "default",
) -> UnifiedFADataset:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if vendor == VENDOR_TOPCON:
        return load_topcon_dataset(path)
    if vendor == VENDOR_ZEISS:
        return load_zeiss_dataset(path)
    if vendor == VENDOR_HDB:
        return load_hdb_dataset(path, relative_time_mode=hdb_relative_time_parser)
    if vendor == VENDOR_CFP:
        return load_cfp_dataset(path)

    detected_vendor = detect_vendor(path)
    candidate_vendors = unique_preserve_order(
        [detected_vendor or "", VENDOR_TOPCON, VENDOR_ZEISS, VENDOR_HDB, VENDOR_CFP]
    )
    errors: list[str] = []

    for candidate_vendor in candidate_vendors:
        try:
            if candidate_vendor == VENDOR_TOPCON:
                return load_topcon_dataset(path)
            if candidate_vendor == VENDOR_ZEISS:
                return load_zeiss_dataset(path)
            if candidate_vendor == VENDOR_HDB:
                return load_hdb_dataset(path, relative_time_mode=hdb_relative_time_parser)
            if candidate_vendor == VENDOR_CFP:
                return load_cfp_dataset(path)
        except Exception as exc:
            errors.append(f"{candidate_vendor}: {exc}")

    details = "\n".join(errors) if errors else "No parser succeeded."
    raise RuntimeError(f"Unable to parse FA dataset from {path}\n{details}")


def frame_to_qpixmap(frame: UnifiedFAFrame) -> QPixmap:
    if frame.source_path is not None and frame.image_array is None:
        pixmap = QPixmap(str(frame.source_path))
        if not pixmap.isNull():
            return pixmap

    if frame.image_array is None:
        return QPixmap()

    array = normalize_to_uint8(np.asarray(frame.image_array))
    if array.ndim == 2:
        contiguous = np.ascontiguousarray(array)
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
        return QPixmap.fromImage(image)

    if array.ndim == 3:
        if array.shape[2] == 1:
            contiguous = np.ascontiguousarray(array[:, :, 0])
            image = QImage(
                contiguous.data,
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(image)

        rgb = np.ascontiguousarray(array[:, :, :3])
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)

    return QPixmap()


def build_summary_text(dataset: UnifiedFADataset, frames: list[UnifiedFAFrame]) -> str:
    lines = [
        f"Vendor: {dataset.vendor_label}",
        f"Frames: {len(frames)}",
        f"Groups: {summarize_groups(frames)}",
        f"Time range: {format_time_range(frames)}",
    ]
    if dataset.study_info.study_code:
        lines.append(f"Study: {dataset.study_info.study_code}")
    return "\n".join(lines)


def build_frame_metadata_text(
    dataset: UnifiedFADataset,
    frame: UnifiedFAFrame,
    *,
    visible_index: int,
    visible_total: int,
    total_frames: int,
) -> str:
    absolute_time = frame.acquisition_datetime.strftime("%Y-%m-%d %H:%M:%S") if frame.acquisition_datetime else "-"
    return "\n".join(
        [
            f"Vendor: {dataset.vendor_label}",
            f"Track: {frame.group_label or '-'}",
            f"Eye: {frame.laterality_label or '-'}",
            f"Frame: {visible_index + 1}/{visible_total}",
            f"Original index: {frame.order_index + 1}/{total_frames}",
            f"File: {frame.filename}",
            f"Label: {frame.label or '-'}",
            f"Acquisition: {frame.acquisition_display or '-'}",
            f"Absolute time: {absolute_time}",
            f"Elapsed: {frame.elapsed_display}",
            f"Size: {frame.size_display}",
            f"Source: {frame.source_detail or '-'}",
            f"Path: {frame.source_path or '-'}",
        ]
    )


class ScaledImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("请选择 Topcon / Zeiss / HDB FA 数据")
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setWordWrap(True)
        self.setStyleSheet(
            """
            QLabel {
                background: #0B1220;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #CBD5E1;
            }
            """
        )

    def set_pixmap_content(self, pixmap: QPixmap, fallback_text: str) -> None:
        self._pixmap = pixmap
        self.setText("" if not pixmap.isNull() else fallback_text)
        self._refresh_pixmap()

    def clear_image(self, text: str) -> None:
        self._pixmap = QPixmap()
        super().setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap.isNull():
            super().setPixmap(QPixmap())
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        super().setPixmap(scaled)


class UnifiedFAViewerWindow(QMainWindow):
    def __init__(self, input_path: str | None, *, vendor: str) -> None:
        super().__init__()
        self.settings = QSettings("OpenAI", "UnifiedFAQtViewer")
        self.dataset: UnifiedFADataset | None = None
        self.frames: list[UnifiedFAFrame] = []
        self.visible_frames: list[UnifiedFAFrame] = []
        self.current_visible_index = -1
        self.playing = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)

        self.setWindowTitle("Unified FA Viewer")
        self.resize(1500, 920)
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self._apply_styles()
        self.setStatusBar(QStatusBar(self))

        self.image_label = ScaledImageLabel()
        self._build_controls(vendor)
        self._build_toolbar()
        self._build_layout()
        self._install_shortcuts()

        remembered_path = self.settings.value("last_path", "", type=str)
        startup_path = input_path or remembered_path
        if startup_path:
            self.load_path(startup_path)
        else:
            self._set_empty_state("No dataset loaded")

    def _apply_styles(self) -> None:
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
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
                background: #0B1220;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px 8px;
                color: #E5E7EB;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QSlider::groove:horizontal {
                border-radius: 3px;
                height: 6px;
                background: #334155;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #60A5FA;
            }
            QTableWidget {
                background: #0B1220;
                alternate-background-color: #101A2E;
                border: 1px solid #334155;
                border-radius: 8px;
                gridline-color: #243041;
            }
            QHeaderView::section {
                background: #1E293B;
                color: #E5E7EB;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #334155;
            }
            QLabel#SummaryLabel, QLabel#PathLabel, QLabel#FrameInfo, QLabel#ViewCaption {
                color: #CBD5E1;
            }
            """
        )

    def _build_controls(self, vendor: str) -> None:
        self.open_dir_button = QPushButton("Open Dir")
        self.open_dir_button.clicked.connect(self.choose_directory)

        self.open_file_button = QPushButton("Open File")
        self.open_file_button.clicked.connect(self.choose_file)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_path)

        self.path_edit = QLineEdit()
        self.path_edit.returnPressed.connect(self.reload_path)

        self.vendor_combo = QComboBox()
        self.vendor_combo.addItem("自动", VENDOR_AUTO)
        self.vendor_combo.addItem("Topcon", VENDOR_TOPCON)
        self.vendor_combo.addItem("Zeiss", VENDOR_ZEISS)
        self.vendor_combo.addItem("HDB", VENDOR_HDB)
        self.vendor_combo.addItem("CFP", VENDOR_CFP)
        initial_index = self.vendor_combo.findData(vendor)
        if initial_index >= 0:
            self.vendor_combo.setCurrentIndex(initial_index)

        self.group_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self.refresh_visible_frames)

        self.eye_combo = QComboBox()
        self.eye_combo.currentIndexChanged.connect(self.refresh_visible_frames)

        self.hide_proofsheet_checkbox = QCheckBox("隐藏 Proofsheet")
        self.hide_proofsheet_checkbox.setChecked(True)
        self.hide_proofsheet_checkbox.toggled.connect(self.refresh_visible_frames)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self.retreat_frame)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.advance_frame)

        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 30)
        self.fps_spinbox.setValue(3)
        self.fps_spinbox.setSuffix(" fps")
        self.fps_spinbox.valueChanged.connect(self._update_play_interval)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self.set_current_visible_index)

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setValue(1)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "文件", "分组", "标签", "相对时间", "采集时间"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._sync_selection_from_table)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(1, self.table.horizontalHeader().Stretch)

        self.dataset_path_label = QLabel("-")
        self.dataset_vendor_label = QLabel("-")
        self.dataset_frames_label = QLabel("-")
        self.dataset_groups_label = QLabel("-")
        self.dataset_time_range_label = QLabel("-")

        self.path_label = QLabel("No dataset loaded")
        self.path_label.setObjectName("PathLabel")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.summary_label = QLabel("Ready to load Topcon / Zeiss / HDB / CFP FA data")
        self.summary_label.setObjectName("SummaryLabel")
        self.summary_label.setWordWrap(True)

        self.frame_info_label = QLabel("Frame: -")
        self.frame_info_label.setObjectName("FrameInfo")

        self.view_caption_label = QLabel("FA frame")
        self.view_caption_label.setObjectName("ViewCaption")
        self.view_caption_label.setAlignment(Qt.AlignCenter)
        self.view_caption_label.setWordWrap(True)

        self.metadata_edit = QPlainTextEdit()
        self.metadata_edit.setReadOnly(True)
        self.metadata_edit.setFont(QFont("Consolas", 10))

        self.patient_name_label = QLabel("-")
        self.patient_id_label = QLabel("-")
        self.patient_sex_label = QLabel("-")
        self.patient_birth_label = QLabel("-")
        self.patient_exam_label = QLabel("-")
        self.patient_eye_label = QLabel("-")
        self.patient_study_code_label = QLabel("-")
        self.patient_device_label = QLabel("-")

        self.frame_position_label = QLabel("-")
        self.frame_file_label = QLabel("-")
        self.frame_group_label = QLabel("-")
        self.frame_label_label = QLabel("-")
        self.frame_elapsed_label = QLabel("-")
        self.frame_time_label = QLabel("-")
        self.frame_size_label = QLabel("-")
        self.frame_source_label = QLabel("-")

        stable_labels = [
            self.frame_info_label,
            self.patient_name_label,
            self.patient_id_label,
            self.patient_sex_label,
            self.patient_birth_label,
            self.patient_exam_label,
            self.patient_eye_label,
            self.patient_device_label,
            self.frame_group_label,
            self.frame_label_label,
            self.frame_elapsed_label,
            self.frame_time_label,
            self.frame_size_label,
            self.frame_source_label,
        ]
        for label in stable_labels:
            label.setWordWrap(False)
            label.setMinimumHeight(24)

        self._update_play_interval()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Actions", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_dir_action = QAction("Open Dir", self)
        open_dir_action.triggered.connect(self.choose_directory)
        toolbar.addAction(open_dir_action)

        open_file_action = QAction("Open File", self)
        open_file_action.triggered.connect(self.choose_file)
        toolbar.addAction(open_file_action)

        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self.reload_path)
        toolbar.addAction(reload_action)

        play_action = QAction("Play/Pause", self)
        play_action.triggered.connect(self.toggle_playback)
        toolbar.addAction(play_action)

    def _build_layout(self) -> None:
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(self.open_dir_button)
        path_row.addWidget(self.open_file_button)
        path_row.addWidget(self.reload_button)
        input_layout.addLayout(path_row)
        input_form = QFormLayout()
        input_form.addRow("Vendor", self.vendor_combo)
        input_form.addRow("Path", self.path_label)
        input_form.addRow("", self.hide_proofsheet_checkbox)
        input_layout.addLayout(input_form)

        navigation_group = QGroupBox("Navigation")
        navigation_layout = QGridLayout(navigation_group)
        navigation_layout.addWidget(QLabel("Track"), 0, 0)
        navigation_layout.addWidget(self.group_combo, 0, 1, 1, 3)
        navigation_layout.addWidget(QLabel("Eye"), 1, 0)
        navigation_layout.addWidget(self.eye_combo, 1, 1, 1, 3)
        navigation_layout.addWidget(QLabel("Frame"), 2, 0)
        navigation_layout.addWidget(self.timeline_slider, 2, 1, 1, 3)
        navigation_layout.addWidget(self.prev_button, 3, 0)
        navigation_layout.addWidget(self.frame_spin, 3, 1)
        navigation_layout.addWidget(self.next_button, 3, 2)
        navigation_layout.addWidget(self.play_button, 3, 3)
        navigation_layout.addWidget(QLabel("FPS"), 4, 0)
        navigation_layout.addWidget(self.fps_spinbox, 4, 1)
        navigation_layout.addWidget(self.frame_info_label, 4, 2, 1, 2)

        display_group = QGroupBox("Display")
        display_layout = QFormLayout(display_group)
        display_layout.addRow("Track", self.frame_group_label)
        display_layout.addRow("Eye", self.patient_eye_label)
        display_layout.addRow("Label", self.frame_label_label)
        display_layout.addRow("Time", self.frame_time_label)
        display_layout.addRow("Elapsed", self.frame_elapsed_label)
        display_layout.addRow("Size", self.frame_size_label)
        display_layout.addRow("Source", self.frame_source_label)

        patient_group = QGroupBox("Patient")
        patient_layout = QFormLayout(patient_group)
        patient_layout.addRow("Name", self.patient_name_label)
        patient_layout.addRow("ID", self.patient_id_label)
        patient_layout.addRow("Sex", self.patient_sex_label)
        patient_layout.addRow("Birth date", self.patient_birth_label)
        patient_layout.addRow("Exam date", self.patient_exam_label)
        patient_layout.addRow("Device", self.patient_device_label)

        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.summary_label)

        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        metadata_layout.addWidget(self.metadata_edit)

        control_widget = QWidget()
        control_widget.setMinimumWidth(420)
        control_widget.setMaximumWidth(420)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(input_group)
        control_layout.addWidget(navigation_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(patient_group)
        control_layout.addWidget(summary_group)
        control_layout.addWidget(metadata_group, stretch=1)

        view_widget = QWidget()
        view_layout = QVBoxLayout(view_widget)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.image_label, stretch=1)
        view_layout.addWidget(self.view_caption_label)
        view_layout.addWidget(self.table, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(control_widget)
        splitter.addWidget(view_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.advance_frame)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.retreat_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_playback)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_directory)

    def _selected_vendor(self) -> str:
        return self.vendor_combo.currentData() or VENDOR_AUTO

    def _update_play_interval(self) -> None:
        fps = max(1, self.fps_spinbox.value())
        interval_ms = int(1000 / fps)
        if self.playing:
            self.play_timer.start(interval_ms)
        else:
            self.play_timer.setInterval(interval_ms)

    def _on_frame_spin_changed(self, value: int) -> None:
        self.set_current_visible_index(max(0, value - 1))

    def choose_directory(self) -> None:
        default_path = default_input_path()
        start_dir = self.path_edit.text().strip()
        if not start_dir:
            start_dir = str(default_path.parent if default_path else Path.home())
        selected = QFileDialog.getExistingDirectory(self, "选择 FA 数据目录", start_dir)
        if selected:
            self.path_edit.setText(selected)
            self.load_path(selected)

    def choose_file(self) -> None:
        default_path = default_input_path()
        start_dir = self.path_edit.text().strip()
        if not start_dir:
            start_dir = str(default_path.parent if default_path else Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Zeiss DICOM / HDB E2E 文件",
            start_dir,
            "FA files (*.dcm *.dicom *.e2e *.E2E);;DICOM (*.dcm *.dicom);;E2E (*.e2e *.E2E);;All files (*)",
        )
        if selected:
            self.path_edit.setText(selected)
            self.load_path(selected)

    def reload_path(self) -> None:
        self.load_path(self.path_edit.text().strip())

    def load_path(self, path_text: str) -> None:
        if not path_text:
            QMessageBox.warning(self, "缺少路径", "请先选择或输入一个 Topcon/Zeiss FA 路径。")
            return

        try:
            dataset = load_unified_dataset(path_text, vendor=self._selected_vendor())
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.dataset = dataset
        self.frames = dataset.frames
        self.path_edit.setText(str(dataset.input_path))
        self.path_label.setText(str(dataset.input_path))
        self.settings.setValue("last_path", str(dataset.input_path))
        self.settings.setValue("vendor_mode", self._selected_vendor())
        self._populate_group_combo()
        self._populate_eye_combo()
        self._update_dataset_panel()
        self._update_patient_panel()
        self._update_vendor_specific_controls()
        self.summary_label.setText(build_summary_text(dataset, dataset.frames))
        self.refresh_visible_frames()
        self.statusBar().showMessage(
            f"已加载 {dataset.vendor_label} 数据：{len(dataset.frames)} 帧 | {dataset.input_path}",
            5000,
        )

    def _populate_group_combo(self) -> None:
        current = self.group_combo.currentText()
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("All")
        for group in unique_preserve_order([frame.group_label for frame in self.frames]):
            self.group_combo.addItem(group)
        if current:
            index = self.group_combo.findText(current)
            if index >= 0:
                self.group_combo.setCurrentIndex(index)
        self.group_combo.blockSignals(False)

    def _populate_eye_combo(self) -> None:
        current = self.eye_combo.currentText()
        self.eye_combo.blockSignals(True)
        self.eye_combo.clear()
        self.eye_combo.addItem("All")
        for eye in unique_preserve_order([frame.laterality_label for frame in self.frames]):
            self.eye_combo.addItem(eye)
        if current:
            index = self.eye_combo.findText(current)
            if index >= 0:
                self.eye_combo.setCurrentIndex(index)
        self.eye_combo.blockSignals(False)

    def _update_vendor_specific_controls(self) -> None:
        is_topcon = self.dataset is not None and self.dataset.vendor == VENDOR_TOPCON
        has_proofsheet = any(frame.is_proofsheet for frame in self.frames)
        self.hide_proofsheet_checkbox.blockSignals(True)
        self.hide_proofsheet_checkbox.setEnabled(is_topcon and has_proofsheet)
        self.hide_proofsheet_checkbox.setChecked(is_topcon and has_proofsheet)
        self.hide_proofsheet_checkbox.blockSignals(False)

    def _update_dataset_panel(self) -> None:
        if self.dataset is None:
            return
        self.dataset_path_label.setText(str(self.dataset.input_path))
        self.dataset_vendor_label.setText(self.dataset.vendor_label)
        self.dataset_frames_label.setText(str(len(self.frames)))
        self.dataset_groups_label.setText(summarize_groups(self.frames))
        self.dataset_time_range_label.setText(format_time_range(self.frames))
        self.summary_label.setText(build_summary_text(self.dataset, self.frames))

    def _update_patient_panel(self) -> None:
        if self.dataset is None:
            return
        study_info = self.dataset.study_info
        self.patient_name_label.setText(study_info.patient_name or "-")
        self.patient_id_label.setText(study_info.patient_id or "-")
        self.patient_sex_label.setText(study_info.sex or "-")
        self.patient_birth_label.setText(study_info.birth_date or "-")
        self.patient_exam_label.setText(study_info.exam_date or "-")
        self.patient_eye_label.setText(study_info.laterality or "-")
        self.patient_study_code_label.setText(study_info.study_code or "-")
        self.patient_device_label.setText(study_info.device_model or "-")

    def refresh_visible_frames(self) -> None:
        if not self.frames:
            self._set_empty_state("当前没有可显示的帧。")
            return

        current_file = None
        if 0 <= self.current_visible_index < len(self.visible_frames):
            current_file = self.visible_frames[self.current_visible_index].filename

        selected_group = self.group_combo.currentText()
        selected_eye = self.eye_combo.currentText()
        hide_proofsheets = self.hide_proofsheet_checkbox.isChecked()

        self.visible_frames = []
        for frame in self.frames:
            if selected_group != "All" and frame.group_label != selected_group:
                continue
            if selected_eye != "All" and frame.laterality_label != selected_eye:
                continue
            if hide_proofsheets and frame.is_proofsheet:
                continue
            self.visible_frames.append(frame)

        self._populate_table()

        if not self.visible_frames:
            self._set_frame_detail_state("当前筛选条件下没有帧。")
            return

        target_index = 0
        if current_file:
            for index, frame in enumerate(self.visible_frames):
                if frame.filename == current_file:
                    target_index = index
                    break

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(len(self.visible_frames) - 1)
        self.timeline_slider.setValue(target_index)
        self.timeline_slider.blockSignals(False)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(max(len(self.visible_frames), 1))
        self.frame_spin.setValue(target_index + 1)
        self.frame_spin.setSuffix(f" / {len(self.visible_frames)}")
        self.frame_spin.blockSignals(False)

        self.set_current_visible_index(target_index)

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.visible_frames))
        for row, frame in enumerate(self.visible_frames):
            values = [
                str(frame.order_index + 1),
                frame.filename,
                frame.group_label,
                frame.label or "-",
                frame.elapsed_display,
                frame.acquisition_display,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)

    def _sync_selection_from_table(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row != self.current_visible_index:
            self.timeline_slider.setValue(row)

    def set_current_visible_index(self, index: int) -> None:
        if not self.visible_frames:
            self.current_visible_index = -1
            return
        if index < 0 or index >= len(self.visible_frames):
            return

        self.current_visible_index = index
        frame = self.visible_frames[index]
        pixmap = frame_to_qpixmap(frame)
        fallback_text = f"无法加载图像\n{frame.source_path or frame.filename}"
        self.image_label.set_pixmap_content(pixmap, fallback_text)

        self.frame_position_label.setText(
            f"筛选后 {index + 1}/{len(self.visible_frames)} · 原始 {frame.order_index + 1}/{len(self.frames)}"
        )
        self.frame_file_label.setText(str(frame.source_path) if frame.source_path else frame.filename)
        self.frame_group_label.setText(frame.group_label or "-")
        self.frame_label_label.setText(frame.label or "-")
        self.frame_elapsed_label.setText(frame.elapsed_display)
        self.frame_time_label.setText(frame.acquisition_display or "-")
        self.frame_size_label.setText(frame.size_display)
        self.frame_source_label.setText(frame.source_detail or "-")
        self.patient_eye_label.setText(frame.laterality_label or (self.dataset.study_info.laterality if self.dataset else "-"))
        self.frame_info_label.setText(f"Frame: {index + 1}/{len(self.visible_frames)}")
        self.view_caption_label.setText(
            f"{frame.group_label or '-'} | Frame {index + 1}/{len(self.visible_frames)} | {frame.acquisition_display or '-'}"
        )

        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(index + 1)
        self.frame_spin.blockSignals(False)

        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.scrollToItem(self.table.item(index, 0))
        self.table.blockSignals(False)

        if self.dataset is not None:
            self.metadata_edit.setPlainText(
                build_frame_metadata_text(
                    self.dataset,
                    frame,
                    visible_index=index,
                    visible_total=len(self.visible_frames),
                    total_frames=len(self.frames),
                )
            )

        self.statusBar().showMessage(
            f"{frame.filename} | {frame.group_label} | {frame.label or '-'} | {frame.acquisition_display}"
        )

    def advance_frame(self) -> None:
        if not self.visible_frames:
            return
        next_index = (self.current_visible_index + 1) % len(self.visible_frames)
        self.timeline_slider.setValue(next_index)

    def retreat_frame(self) -> None:
        if not self.visible_frames:
            return
        previous_index = (self.current_visible_index - 1) % len(self.visible_frames)
        self.timeline_slider.setValue(previous_index)

    def toggle_playback(self) -> None:
        if not self.visible_frames:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText("Pause")
            self._update_play_interval()
        else:
            self.play_timer.stop()
            self.play_button.setText("Play")

    def _set_frame_detail_state(self, message: str) -> None:
        self.playing = False
        self.play_timer.stop()
        self.play_button.setText("Play")
        self.visible_frames = []
        self.current_visible_index = -1
        self.table.setRowCount(0)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setValue(1)
        self.frame_spin.setSuffix(" / 0")
        self.frame_spin.blockSignals(False)
        self.image_label.clear_image(message)
        self.view_caption_label.setText("FA frame")
        self.frame_info_label.setText("Frame: -")
        self.metadata_edit.setPlainText("")
        self.frame_position_label.setText("-")
        self.frame_file_label.setText("-")
        self.frame_group_label.setText("-")
        self.frame_label_label.setText("-")
        self.frame_elapsed_label.setText("-")
        self.frame_time_label.setText("-")
        self.frame_size_label.setText("-")
        self.frame_source_label.setText("-")

    def _set_empty_state(self, message: str) -> None:
        self.dataset = None
        self.frames = []
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("All")
        self.group_combo.blockSignals(False)
        self.eye_combo.blockSignals(True)
        self.eye_combo.clear()
        self.eye_combo.addItem("All")
        self.eye_combo.blockSignals(False)
        self._set_frame_detail_state(message)
        self.path_label.setText(message)
        self.summary_label.setText("Ready to load Topcon / Zeiss / HDB / CFP FA data")
        self.dataset_path_label.setText("-")
        self.dataset_vendor_label.setText("-")
        self.dataset_frames_label.setText("-")
        self.dataset_groups_label.setText("-")
        self.dataset_time_range_label.setText("-")
        self.patient_name_label.setText("-")
        self.patient_id_label.setText("-")
        self.patient_sex_label.setText("-")
        self.patient_birth_label.setText("-")
        self.patient_exam_label.setText("-")
        self.patient_eye_label.setText("-")
        self.patient_study_code_label.setText("-")
        self.patient_device_label.setText("-")


def dump_dataset(dataset: UnifiedFADataset) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    info = dataset.study_info
    print("vendor\t", info.vendor_label)
    print("path\t", dataset.input_path)
    print("patient_name\t", info.patient_name or "-")
    print("patient_id\t", info.patient_id or "-")
    print("sex\t", info.sex or "-")
    print("birth_date\t", info.birth_date or "-")
    print("exam_date\t", info.exam_date or "-")
    print("laterality\t", info.laterality or "-")
    print("study_code\t", info.study_code or "-")
    print("device_model\t", info.device_model or "-")
    print()
    print("index\tfilename\tgroup\tlabel\telapsed_s\tacquisition")
    for frame in dataset.frames:
        elapsed = "" if frame.elapsed_seconds is None else f"{frame.elapsed_seconds:.3f}"
        print(
            f"{frame.order_index + 1}\t{frame.filename}\t{frame.group_label}\t{frame.label}\t"
            f"{elapsed}\t{frame.acquisition_display}"
        )


def main() -> int:
    args = parse_args()
    if args.dump:
        if not args.input_path:
            print("No input path provided.")
            return 1
        dataset = load_unified_dataset(args.input_path, vendor=args.vendor)
        dump_dataset(dataset)
        return 0

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    application = QApplication(sys.argv)
    application.setApplicationName("UnifiedFAQtViewer")
    application.setStyle("Fusion")

    remembered_vendor = QSettings("OpenAI", "UnifiedFAQtViewer").value("vendor_mode", VENDOR_AUTO, type=str)
    startup_vendor = args.vendor if args.vendor != VENDOR_AUTO else remembered_vendor
    window = UnifiedFAViewerWindow(args.input_path, vendor=startup_vendor)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())
