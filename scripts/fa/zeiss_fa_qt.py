from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import convert_color_space


DICOM_SUFFIXES = {".dcm", ".dicom"}
OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"
FA_KEYWORDS = {
    "fa",
    "ffa",
    "fluorescein",
    "angiography",
}
DEFAULT_INPUT_PATH = Path(r"E:\Data\OCT\蔡司FA\2")


@dataclass
class FrameRecord:
    source_file: str
    frame_index: int
    instance_number: int | None
    acquisition_date: str
    acquisition_time: str
    acquisition_source: str
    acquisition_datetime_raw: str
    acquisition_datetime_iso: str | None
    acquisition_date_iso: str | None
    acquisition_time_iso: str | None
    relative_time_seconds: float | None
    image: np.ndarray


@dataclass
class SeriesRecord:
    patient_name: str
    patient_id: str
    patient_sex: str
    patient_birth_date_raw: str
    patient_birth_date_iso: str | None
    series_uid: str
    series_number: int | None
    series_description: str
    study_description: str
    protocol_name: str
    modality: str
    laterality: str
    sop_class_uid: str
    photometric_interpretation: str
    rows: int | None
    columns: int | None
    fa_score: int
    study_datetime_raw: str
    study_datetime_iso: str | None
    series_datetime_raw: str
    series_datetime_iso: str | None
    files: list[Path]
    frames: list[FrameRecord]


@dataclass
class ParsedTimeInfo:
    source: str
    raw_datetime: str
    raw_date: str
    raw_time: str
    datetime_value: datetime | None
    date_value: date | None
    time_value: time | None
    iso_datetime: str | None
    iso_date: str | None
    iso_time: str | None


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value).split("\x00", 1)[0]
    text = "".join(character for character in text if character.isprintable())
    return text.strip()


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def safe_dcmread(path: Path, *, stop_before_pixels: bool = False) -> pydicom.dataset.FileDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydicom.dcmread(path, force=True, stop_before_pixels=stop_before_pixels)


def slugify(text: str) -> str:
    normalized = clean_text(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def format_person_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = text.replace("^", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_patient_sex(value: Any) -> str:
    text = clean_text(value).upper()
    mapping = {
        "M": "Male",
        "F": "Female",
        "O": "Other",
    }
    return mapping.get(text, text)


def datetime_to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def date_to_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def time_to_iso(value: time | None) -> str | None:
    return value.isoformat() if value is not None else None


def parse_dicom_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = clean_text(value)
    if not text:
        return None

    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) < 8:
        return None

    try:
        return date(
            int(digits[0:4]),
            int(digits[4:6]),
            int(digits[6:8]),
        )
    except ValueError:
        return None


def parse_dicom_time(value: Any) -> time | None:
    if isinstance(value, datetime):
        return value.timetz().replace(tzinfo=None)
    if isinstance(value, time):
        return value

    text = clean_text(value)
    if not text:
        return None

    text = text.replace(":", "")
    match = re.fullmatch(r"(\d{2})(\d{2})?(\d{2})?(?:\.(\d+))?", text)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    fraction = match.group(4) or ""
    microseconds = int((fraction + "000000")[:6]) if fraction else 0

    try:
        return time(hours, minutes, seconds, microseconds)
    except ValueError:
        return None


def parse_dicom_timezone(offset_text: str | None) -> timezone | None:
    if not offset_text:
        return None

    match = re.fullmatch(r"([+-])(\d{2})(\d{2})", offset_text)
    if not match:
        return None

    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3))
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(sign * delta)


def parse_dicom_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value

    text = clean_text(value)
    if not text:
        return None

    timezone_match = re.search(r"([+-]\d{4})$", text)
    tzinfo = parse_dicom_timezone(timezone_match.group(1) if timezone_match else None)
    if timezone_match:
        text = text[: timezone_match.start()]

    if "." in text:
        main_part, fraction_part = text.split(".", 1)
    else:
        main_part, fraction_part = text, ""

    digits = re.sub(r"[^0-9]", "", main_part)
    if len(digits) < 4:
        return None

    try:
        year = int(digits[0:4])
        month = int(digits[4:6]) if len(digits) >= 6 else 1
        day = int(digits[6:8]) if len(digits) >= 8 else 1
        hour = int(digits[8:10]) if len(digits) >= 10 else 0
        minute = int(digits[10:12]) if len(digits) >= 12 else 0
        second = int(digits[12:14]) if len(digits) >= 14 else 0
        microseconds = int((fraction_part + "000000")[:6]) if fraction_part else 0
        return datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microseconds,
            tzinfo=tzinfo,
        )
    except ValueError:
        return None


def combine_dicom_date_time(date_value: Any, time_value: Any) -> datetime | None:
    parsed_date = parse_dicom_date(date_value)
    parsed_time = parse_dicom_time(time_value)
    if parsed_date is None or parsed_time is None:
        return None

    return datetime.combine(parsed_date, parsed_time)


def iter_nested_datasets(
    dataset: pydicom.dataset.Dataset,
    *,
    max_depth: int,
    prefix: str = "",
):
    yield prefix, dataset
    if max_depth <= 0:
        return

    for element in dataset:
        if element.VR != "SQ":
            continue

        sequence_keyword = element.keyword or str(element.tag)
        for index, item in enumerate(element.value):
            if not isinstance(item, pydicom.dataset.Dataset):
                continue
            nested_prefix = f"{prefix}{sequence_keyword}[{index}]."
            yield from iter_nested_datasets(
                item,
                max_depth=max_depth - 1,
                prefix=nested_prefix,
            )


def find_first_value(
    dataset: pydicom.dataset.Dataset,
    keywords: list[str] | tuple[str, ...],
    *,
    recursive: bool = False,
    max_depth: int = 3,
) -> tuple[Any, str]:
    depth = max_depth if recursive else 0
    for prefix, nested_dataset in iter_nested_datasets(dataset, max_depth=depth):
        for keyword in keywords:
            value = getattr(nested_dataset, keyword, None)
            if value is None:
                continue
            if clean_text(value):
                return value, f"{prefix}{keyword}"
    return None, ""


def empty_time_info() -> ParsedTimeInfo:
    return ParsedTimeInfo(
        source="",
        raw_datetime="",
        raw_date="",
        raw_time="",
        datetime_value=None,
        date_value=None,
        time_value=None,
        iso_datetime=None,
        iso_date=None,
        iso_time=None,
    )


def build_time_info_from_value(
    *,
    source: str,
    raw_datetime: str = "",
    raw_date: str = "",
    raw_time: str = "",
    datetime_value: datetime | None = None,
    date_value: date | None = None,
    time_value: time | None = None,
) -> ParsedTimeInfo:
    return ParsedTimeInfo(
        source=source,
        raw_datetime=raw_datetime,
        raw_date=raw_date,
        raw_time=raw_time,
        datetime_value=datetime_value,
        date_value=date_value,
        time_value=time_value,
        iso_datetime=datetime_to_iso(datetime_value),
        iso_date=date_to_iso(date_value if date_value is not None else (datetime_value.date() if datetime_value else None)),
        iso_time=time_to_iso(time_value if time_value is not None else (datetime_value.timetz().replace(tzinfo=None) if datetime_value else None)),
    )


def extract_time_info(
    dataset: pydicom.dataset.Dataset,
    *,
    recursive: bool = False,
    datetime_keywords: list[str] | tuple[str, ...] = (
        "AcquisitionDateTime",
        "FrameAcquisitionDateTime",
        "FrameReferenceDateTime",
        "ContentDateTime",
    ),
    date_time_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (
        ("AcquisitionDate", "AcquisitionTime"),
        ("ContentDate", "ContentTime"),
        ("SeriesDate", "SeriesTime"),
        ("StudyDate", "StudyTime"),
    ),
) -> ParsedTimeInfo:
    raw_datetime_value, raw_datetime_source = find_first_value(
        dataset,
        datetime_keywords,
        recursive=recursive,
    )
    parsed_datetime = parse_dicom_datetime(raw_datetime_value)
    if parsed_datetime is not None:
        return build_time_info_from_value(
            source=raw_datetime_source,
            raw_datetime=clean_text(raw_datetime_value),
            datetime_value=parsed_datetime,
        )

    best_partial: ParsedTimeInfo | None = None
    best_partial_score = -1

    for date_keyword, time_keyword in date_time_pairs:
        raw_date_value, raw_date_source = find_first_value(
            dataset,
            [date_keyword],
            recursive=recursive,
        )
        raw_time_value, raw_time_source = find_first_value(
            dataset,
            [time_keyword],
            recursive=recursive,
        )
        parsed_date = parse_dicom_date(raw_date_value)
        parsed_time = parse_dicom_time(raw_time_value)
        parsed_datetime = combine_dicom_date_time(raw_date_value, raw_time_value)

        if parsed_datetime is None and parsed_date is None and parsed_time is None:
            continue

        source = raw_date_source or raw_time_source or f"{date_keyword}+{time_keyword}"
        candidate = build_time_info_from_value(
            source=source,
            raw_date=clean_text(raw_date_value),
            raw_time=clean_text(raw_time_value),
            datetime_value=parsed_datetime,
            date_value=parsed_date,
            time_value=parsed_time,
        )

        if parsed_datetime is not None:
            return candidate

        candidate_score = 1 if (parsed_date is not None or parsed_time is not None) else 0
        if parsed_date is not None and parsed_time is not None:
            candidate_score = 2
        if candidate_score > best_partial_score:
            best_partial = candidate
            best_partial_score = candidate_score

    return best_partial if best_partial is not None else empty_time_info()


def parse_float_sequence(value: Any) -> list[float]:
    if value is None:
        return []

    if isinstance(value, (str, bytes)):
        parsed_value = parse_float(value)
        return [] if parsed_value is None else [parsed_value]

    if isinstance(value, (list, tuple)):
        iterable = value
    else:
        try:
            iterable = list(value)
        except TypeError:
            parsed_value = parse_float(value)
            return [] if parsed_value is None else [parsed_value]

    values: list[float] = []
    for item in iterable:
        parsed_value = parse_float(item)
        if parsed_value is not None:
            values.append(parsed_value)
    return values


def extract_relative_frame_times(dataset: pydicom.dataset.FileDataset, frame_count: int) -> list[float | None]:
    if frame_count <= 0:
        return []

    frame_delay_ms = parse_float(getattr(dataset, "FrameDelay", None)) or 0.0
    frame_time_vector = parse_float_sequence(getattr(dataset, "FrameTimeVector", None))

    if frame_time_vector:
        if len(frame_time_vector) == frame_count and all(
            frame_time_vector[index] <= frame_time_vector[index + 1]
            for index in range(len(frame_time_vector) - 1)
        ):
            return [(frame_delay_ms + value) / 1000.0 for value in frame_time_vector]

        offsets_seconds = [frame_delay_ms / 1000.0]
        cumulative_ms = frame_delay_ms
        if len(frame_time_vector) == frame_count:
            delta_values = frame_time_vector[1:]
        else:
            delta_values = frame_time_vector

        for delta_ms in delta_values[: max(frame_count - 1, 0)]:
            cumulative_ms += delta_ms
            offsets_seconds.append(cumulative_ms / 1000.0)

        while len(offsets_seconds) < frame_count:
            offsets_seconds.append(None)
        return offsets_seconds

    frame_time_ms = parse_float(getattr(dataset, "FrameTime", None))
    if frame_time_ms is not None:
        return [
            (frame_delay_ms + frame_index * frame_time_ms) / 1000.0
            for frame_index in range(frame_count)
        ]

    return [None] * frame_count


def extract_per_frame_time_infos(
    dataset: pydicom.dataset.FileDataset,
    frame_count: int,
) -> list[ParsedTimeInfo]:
    sequence = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if sequence is None:
        return [empty_time_info() for _ in range(frame_count)]

    time_infos: list[ParsedTimeInfo] = []
    for frame_index in range(frame_count):
        if frame_index >= len(sequence):
            time_infos.append(empty_time_info())
            continue

        frame_dataset = sequence[frame_index]
        time_infos.append(
            extract_time_info(
                frame_dataset,
                recursive=True,
                datetime_keywords=(
                    "FrameAcquisitionDateTime",
                    "FrameReferenceDateTime",
                    "AcquisitionDateTime",
                    "ContentDateTime",
                ),
                date_time_pairs=(
                    ("AcquisitionDate", "AcquisitionTime"),
                    ("ContentDate", "ContentTime"),
                    ("SeriesDate", "SeriesTime"),
                ),
            )
        )
    return time_infos


def apply_time_offset(time_info: ParsedTimeInfo, offset_seconds: float | None, source_suffix: str) -> ParsedTimeInfo:
    if offset_seconds is None:
        return time_info
    if time_info.datetime_value is None:
        return time_info

    shifted_datetime = time_info.datetime_value + timedelta(seconds=offset_seconds)
    return build_time_info_from_value(
        source=f"{time_info.source}{source_suffix}",
        raw_datetime=time_info.raw_datetime,
        raw_date=time_info.raw_date,
        raw_time=time_info.raw_time,
        datetime_value=shifted_datetime,
        date_value=shifted_datetime.date(),
        time_value=shifted_datetime.timetz().replace(tzinfo=None),
    )


def resolve_relative_acquisition_to_study_time(
    *,
    base_time_info: ParsedTimeInfo,
    study_time_info: ParsedTimeInfo,
) -> ParsedTimeInfo:
    if base_time_info.datetime_value is None or study_time_info.datetime_value is None:
        return base_time_info

    source_text = clean_text(base_time_info.source).lower()
    if "acquisition" not in source_text:
        return base_time_info

    candidate = base_time_info.datetime_value
    if candidate.hour != 0:
        return base_time_info

    reference = study_time_info.datetime_value
    if candidate.date() != reference.date():
        return base_time_info

    offset_seconds = (
        candidate.hour * 3600
        + candidate.minute * 60
        + candidate.second
        + candidate.microsecond / 1_000_000.0
    )
    shifted_datetime = reference + timedelta(seconds=offset_seconds)

    return build_time_info_from_value(
        source=f"{base_time_info.source}+study-offset",
        raw_datetime=base_time_info.raw_datetime,
        raw_date=base_time_info.raw_date,
        raw_time=base_time_info.raw_time,
        datetime_value=shifted_datetime,
        date_value=shifted_datetime.date(),
        time_value=shifted_datetime.timetz().replace(tzinfo=None),
    )


def choose_frame_time_info(
    *,
    base_time_info: ParsedTimeInfo,
    per_frame_time_info: ParsedTimeInfo,
    relative_time_seconds: float | None,
) -> ParsedTimeInfo:
    if per_frame_time_info.datetime_value is not None or per_frame_time_info.time_value is not None:
        return per_frame_time_info

    if relative_time_seconds is not None and base_time_info.datetime_value is not None:
        return apply_time_offset(base_time_info, relative_time_seconds, "+relative-offset")

    return base_time_info


def summarize_series_timing(series: SeriesRecord) -> dict[str, Any]:
    frame_datetimes = [
        frame.acquisition_datetime_iso
        for frame in series.frames
        if frame.acquisition_datetime_iso
    ]
    relative_seconds = [
        frame.relative_time_seconds
        for frame in series.frames
        if frame.relative_time_seconds is not None
    ]
    frame_sources = sorted(
        {
            frame.acquisition_source
            for frame in series.frames
            if frame.acquisition_source
        }
    )

    time_span_seconds = None
    if len(frame_datetimes) >= 2:
        first_datetime = datetime.fromisoformat(frame_datetimes[0])
        last_datetime = datetime.fromisoformat(frame_datetimes[-1])
        time_span_seconds = (last_datetime - first_datetime).total_seconds()
    elif relative_seconds:
        time_span_seconds = max(relative_seconds) - min(relative_seconds)

    return {
        "study_datetime_raw": series.study_datetime_raw,
        "study_datetime_iso": series.study_datetime_iso,
        "series_datetime_raw": series.series_datetime_raw,
        "series_datetime_iso": series.series_datetime_iso,
        "first_frame_datetime_iso": frame_datetimes[0] if frame_datetimes else None,
        "last_frame_datetime_iso": frame_datetimes[-1] if frame_datetimes else None,
        "frame_time_span_seconds": time_span_seconds,
        "frame_count_with_datetime": len(frame_datetimes),
        "frame_count_with_relative_time": len(relative_seconds),
        "time_sources": frame_sources,
    }


def default_output_dir(input_path: Path) -> Path:
    stem = input_path.stem if input_path.is_file() else input_path.name
    return input_path.parent / f"{stem}_fa_exports"


def discover_candidate_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file():
        return [input_path]

    search_root = input_path / "DataFiles" if (input_path / "DataFiles").is_dir() else input_path

    dicom_like_files = sorted(
        candidate
        for candidate in search_root.rglob("*")
        if candidate.is_file()
        and (candidate.suffix.lower() in DICOM_SUFFIXES or candidate.name.upper() == "DICOMDIR")
    )
    if dicom_like_files:
        return dicom_like_files

    return sorted(candidate for candidate in search_root.rglob("*") if candidate.is_file())


def classify_dicom(ds: pydicom.dataset.FileDataset) -> str:
    number_of_frames = parse_int(getattr(ds, "NumberOfFrames", None))
    rows = parse_int(getattr(ds, "Rows", None))
    columns = parse_int(getattr(ds, "Columns", None))

    if number_of_frames is not None and number_of_frames > 1:
        return "multi_frame_image"

    if rows is not None and columns is not None:
        return "single_frame_image"

    if "PixelData" not in ds:
        return "non_image_dicom"

    return "unknown_image"


def tokenize_text(*values: Any) -> set[str]:
    merged = " ".join(clean_text(value).lower() for value in values if clean_text(value))
    return set(re.findall(r"[a-z0-9]+", merged))


def score_fa_series(ds: pydicom.dataset.FileDataset) -> int:
    tokens = tokenize_text(
        getattr(ds, "StudyDescription", ""),
        getattr(ds, "SeriesDescription", ""),
        getattr(ds, "ProtocolName", ""),
        getattr(ds, "ImageComments", ""),
    )

    score = 0
    if "fluorescein" in tokens:
        score += 5
    if "angiography" in tokens:
        score += 4
    if "ffa" in tokens:
        score += 4
    if "fa" in tokens:
        score += 3
    if any(keyword in tokens for keyword in FA_KEYWORDS):
        score += 1
    if clean_text(getattr(ds, "SOPClassUID", "")) == OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID:
        score += 2
    if (parse_int(getattr(ds, "NumberOfFrames", None)) or 1) > 1:
        score += 1
    return score


def build_series_key(ds: pydicom.dataset.FileDataset, path: Path) -> str:
    series_uid = clean_text(getattr(ds, "SeriesInstanceUID", ""))
    if series_uid:
        return series_uid

    fallback_parts = [
        clean_text(getattr(ds, "StudyInstanceUID", "")),
        clean_text(getattr(ds, "SeriesNumber", "")),
        clean_text(getattr(ds, "SeriesDescription", "")),
        path.parent.as_posix(),
    ]
    return "::".join(part for part in fallback_parts if part) or path.parent.as_posix()


def build_file_sort_key(ds: pydicom.dataset.FileDataset, path: Path) -> tuple[Any, ...]:
    acquisition_time_info = extract_time_info(
        ds,
        recursive=False,
        datetime_keywords=("AcquisitionDateTime",),
        date_time_pairs=(
            ("AcquisitionDate", "AcquisitionTime"),
            ("ContentDate", "ContentTime"),
            ("SeriesDate", "SeriesTime"),
            ("StudyDate", "StudyTime"),
        ),
    )
    return (
        acquisition_time_info.iso_datetime or "",
        acquisition_time_info.iso_date or "",
        acquisition_time_info.iso_time or "",
        acquisition_time_info.raw_datetime,
        acquisition_time_info.raw_date,
        acquisition_time_info.raw_time,
        parse_int(getattr(ds, "InstanceNumber", None)) or 0,
        path.name.lower(),
    )


def decode_pixel_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    image = np.asarray(ds.pixel_array)
    photometric = clean_text(getattr(ds, "PhotometricInterpretation", ""))

    if photometric.startswith("YBR"):
        source_space = "YBR_FULL_422" if photometric == "YBR_FULL_422" else "YBR_FULL"
        image = convert_color_space(image, source_space, "RGB")

    if photometric == "MONOCHROME1":
        image = np.max(image) - image

    return np.asarray(image)


def split_frames(image: np.ndarray, ds: pydicom.dataset.FileDataset) -> list[np.ndarray]:
    number_of_frames = parse_int(getattr(ds, "NumberOfFrames", None)) or 1
    samples_per_pixel = parse_int(getattr(ds, "SamplesPerPixel", None)) or 1

    if number_of_frames <= 1:
        return [image]

    if samples_per_pixel > 1 and image.ndim == 4:
        return [image[index] for index in range(image.shape[0])]

    if image.ndim >= 3:
        return [image[index] for index in range(image.shape[0])]

    return [image]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    array = (array - min_value) * (255.0 / (max_value - min_value))
    return np.clip(array, 0, 255).astype(np.uint8)


def format_frame_for_export(image: np.ndarray, *, keep_16bit: bool) -> np.ndarray:
    array = np.asarray(image)

    if array.ndim == 3:
        return normalize_to_uint8(array)

    if keep_16bit:
        if array.dtype == np.uint16:
            return array
        if array.dtype == np.uint8:
            return (array.astype(np.uint16) * 257).astype(np.uint16)

        normalized = cv2.normalize(
            array.astype(np.float32),
            None,
            0,
            65535,
            cv2.NORM_MINMAX,
        )
        return normalized.astype(np.uint16)

    return normalize_to_uint8(array)


def frame_to_bgr(image: np.ndarray) -> np.ndarray:
    display_image = normalize_to_uint8(image)
    if display_image.ndim == 2:
        return cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(display_image[:, :, :3], cv2.COLOR_RGB2BGR)


def load_zeiss_fa_series(input_path: str | Path, *, prefer_fa_only: bool = True) -> list[SeriesRecord]:
    input_path = Path(input_path).expanduser().resolve()
    candidate_files = discover_candidate_files(input_path)

    grouped_items: dict[str, dict[str, Any]] = {}
    image_file_count = 0

    for file_path in candidate_files:
        try:
            header_ds = safe_dcmread(file_path, stop_before_pixels=True)
        except Exception:
            continue

        classification = classify_dicom(header_ds)
        if classification == "non_image_dicom":
            continue

        image_file_count += 1
        series_key = build_series_key(header_ds, file_path)
        study_time_info = extract_time_info(
            header_ds,
            recursive=False,
            datetime_keywords=(),
            date_time_pairs=(("StudyDate", "StudyTime"),),
        )
        series_time_info = extract_time_info(
            header_ds,
            recursive=False,
            datetime_keywords=(),
            date_time_pairs=(
                ("SeriesDate", "SeriesTime"),
                ("AcquisitionDate", "AcquisitionTime"),
                ("ContentDate", "ContentTime"),
            ),
        )
        entry = grouped_items.setdefault(
            series_key,
            {
                "patient_name": format_person_name(getattr(header_ds, "PatientName", "")),
                "patient_id": clean_text(getattr(header_ds, "PatientID", "")),
                "patient_sex": format_patient_sex(getattr(header_ds, "PatientSex", "")),
                "patient_birth_date_raw": clean_text(getattr(header_ds, "PatientBirthDate", "")),
                "patient_birth_date_iso": date_to_iso(parse_dicom_date(getattr(header_ds, "PatientBirthDate", ""))),
                "series_uid": clean_text(getattr(header_ds, "SeriesInstanceUID", "")) or series_key,
                "series_number": parse_int(getattr(header_ds, "SeriesNumber", None)),
                "series_description": clean_text(getattr(header_ds, "SeriesDescription", "")),
                "study_description": clean_text(getattr(header_ds, "StudyDescription", "")),
                "protocol_name": clean_text(getattr(header_ds, "ProtocolName", "")),
                "modality": clean_text(getattr(header_ds, "Modality", "")),
                "laterality": clean_text(getattr(header_ds, "Laterality", "")) or clean_text(getattr(header_ds, "ImageLaterality", "")),
                "sop_class_uid": clean_text(getattr(header_ds, "SOPClassUID", "")),
                "photometric_interpretation": clean_text(getattr(header_ds, "PhotometricInterpretation", "")),
                "rows": parse_int(getattr(header_ds, "Rows", None)),
                "columns": parse_int(getattr(header_ds, "Columns", None)),
                "fa_score": 0,
                "study_datetime_raw": study_time_info.raw_datetime or (
                    f"{study_time_info.raw_date} {study_time_info.raw_time}".strip()
                ),
                "study_datetime_iso": study_time_info.iso_datetime
                or (
                    f"{study_time_info.iso_date}T{study_time_info.iso_time}"
                    if study_time_info.iso_date and study_time_info.iso_time
                    else study_time_info.iso_date
                ),
                "series_datetime_raw": series_time_info.raw_datetime or (
                    f"{series_time_info.raw_date} {series_time_info.raw_time}".strip()
                ),
                "series_datetime_iso": series_time_info.iso_datetime
                or (
                    f"{series_time_info.iso_date}T{series_time_info.iso_time}"
                    if series_time_info.iso_date and series_time_info.iso_time
                    else series_time_info.iso_date
                ),
                "files_with_headers": [],
            },
        )

        entry["fa_score"] = max(entry["fa_score"], score_fa_series(header_ds))
        entry["files_with_headers"].append((file_path, header_ds))

    if image_file_count == 0:
        raise RuntimeError(f"未在 {input_path} 中找到可解码的图像 DICOM。")

    series_records: list[SeriesRecord] = []

    for grouped in grouped_items.values():
        sorted_items = sorted(
            grouped["files_with_headers"],
            key=lambda item: build_file_sort_key(item[1], item[0]),
        )

        frames: list[FrameRecord] = []
        files: list[Path] = []

        for file_path, header_ds in sorted_items:
            ds = safe_dcmread(file_path, stop_before_pixels=False)
            pixel_array = decode_pixel_array(ds)
            split_images = split_frames(pixel_array, ds)
            study_time_info = extract_time_info(
                ds,
                recursive=False,
                datetime_keywords=(),
                date_time_pairs=(("StudyDate", "StudyTime"),),
            )
            base_time_info = extract_time_info(
                ds,
                recursive=False,
                datetime_keywords=("AcquisitionDateTime",),
                date_time_pairs=(
                    ("AcquisitionDate", "AcquisitionTime"),
                    ("ContentDate", "ContentTime"),
                    ("SeriesDate", "SeriesTime"),
                    ("StudyDate", "StudyTime"),
                ),
            )
            base_time_info = resolve_relative_acquisition_to_study_time(
                base_time_info=base_time_info,
                study_time_info=study_time_info,
            )
            per_frame_time_infos = extract_per_frame_time_infos(ds, len(split_images))
            relative_time_seconds_list = extract_relative_frame_times(ds, len(split_images))

            files.append(file_path)
            instance_number = parse_int(getattr(ds, "InstanceNumber", None))

            for frame_index, image in enumerate(split_images):
                relative_time_seconds = (
                    relative_time_seconds_list[frame_index]
                    if frame_index < len(relative_time_seconds_list)
                    else None
                )
                resolved_time_info = choose_frame_time_info(
                    base_time_info=base_time_info,
                    per_frame_time_info=per_frame_time_infos[frame_index]
                    if frame_index < len(per_frame_time_infos)
                    else empty_time_info(),
                    relative_time_seconds=relative_time_seconds,
                )
                frames.append(
                    FrameRecord(
                        source_file=file_path.name,
                        frame_index=frame_index,
                        instance_number=instance_number,
                        acquisition_date=resolved_time_info.raw_date,
                        acquisition_time=resolved_time_info.raw_time,
                        acquisition_source=resolved_time_info.source,
                        acquisition_datetime_raw=resolved_time_info.raw_datetime,
                        acquisition_datetime_iso=resolved_time_info.iso_datetime,
                        acquisition_date_iso=resolved_time_info.iso_date,
                        acquisition_time_iso=resolved_time_info.iso_time,
                        relative_time_seconds=relative_time_seconds,
                        image=np.asarray(image),
                    )
                )

        if not frames:
            continue

        series_records.append(
            SeriesRecord(
                patient_name=grouped["patient_name"],
                patient_id=grouped["patient_id"],
                patient_sex=grouped["patient_sex"],
                patient_birth_date_raw=grouped["patient_birth_date_raw"],
                patient_birth_date_iso=grouped["patient_birth_date_iso"],
                series_uid=grouped["series_uid"],
                series_number=grouped["series_number"],
                series_description=grouped["series_description"],
                study_description=grouped["study_description"],
                protocol_name=grouped["protocol_name"],
                modality=grouped["modality"],
                laterality=grouped["laterality"],
                sop_class_uid=grouped["sop_class_uid"],
                photometric_interpretation=grouped["photometric_interpretation"],
                rows=grouped["rows"],
                columns=grouped["columns"],
                fa_score=grouped["fa_score"],
                study_datetime_raw=grouped["study_datetime_raw"],
                study_datetime_iso=grouped["study_datetime_iso"],
                series_datetime_raw=grouped["series_datetime_raw"],
                series_datetime_iso=grouped["series_datetime_iso"],
                files=files,
                frames=frames,
            )
        )

    if not series_records:
        raise RuntimeError(f"未在 {input_path} 中找到可导出的图像序列。")

    series_records.sort(
        key=lambda series: (
            -series.fa_score,
            series.series_number if series.series_number is not None else 10**9,
            series.series_description.lower(),
            series.series_uid,
        )
    )

    if prefer_fa_only:
        fa_series = [series for series in series_records if series.fa_score > 0]
        if fa_series:
            return fa_series

    return series_records


try:
    from .zeiss_fa_parser import (
        DEFAULT_INPUT_PATH,
        FrameRecord,
        SeriesRecord,
        clean_text,
        load_zeiss_fa_series,
        normalize_to_uint8,
        summarize_series_timing,
    )
except ImportError:
    from zeiss_fa_parser import (
        DEFAULT_INPUT_PATH,
        FrameRecord,
        SeriesRecord,
        clean_text,
        load_zeiss_fa_series,
        normalize_to_uint8,
        summarize_series_timing,
    )


@dataclass
class FAViewerFrame:
    image: np.ndarray
    laterality: str
    source_file: str
    series_uid: str
    series_number: int | None
    series_description: str
    protocol_name: str
    frame_index: int
    acquisition_source: str
    acquisition_datetime_raw: str
    acquisition_datetime_iso: str | None
    acquisition_date_iso: str | None
    acquisition_time_iso: str | None
    relative_time_seconds: float | None
    elapsed_seconds: float | None


@dataclass
class FAViewerTrack:
    key: str
    label: str
    laterality: str
    patient_name: str
    patient_id: str
    patient_sex: str
    patient_birth_date_iso: str | None
    frame_count: int
    series_count: int
    first_datetime_iso: str | None
    last_datetime_iso: str | None
    frames: list[FAViewerFrame]


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def laterality_display_text(laterality: str) -> str:
    normalized = clean_text(laterality).upper()
    if normalized == "L":
        return "OS / Left"
    if normalized == "R":
        return "OD / Right"
    if normalized == "ALL":
        return "All eyes"
    return normalized or "Unknown"


def viewer_frame_sort_key(series: SeriesRecord, frame: FrameRecord) -> tuple[Any, ...]:
    parsed_datetime = parse_iso_datetime(frame.acquisition_datetime_iso)
    return (
        0 if parsed_datetime is not None else 1,
        parsed_datetime or datetime.max,
        0 if frame.relative_time_seconds is not None else 1,
        frame.relative_time_seconds if frame.relative_time_seconds is not None else float("inf"),
        series.series_number if series.series_number is not None else 10**9,
        frame.instance_number if frame.instance_number is not None else 10**9,
        frame.source_file.lower(),
        frame.frame_index,
    )


def build_viewer_track(
    key: str,
    laterality: str,
    grouped_items: list[tuple[SeriesRecord, FrameRecord]],
) -> FAViewerTrack:
    sorted_items = sorted(
        grouped_items,
        key=lambda item: viewer_frame_sort_key(item[0], item[1]),
    )

    first_datetime: datetime | None = None
    first_relative_seconds: float | None = None
    frames: list[FAViewerFrame] = []
    first_series = sorted_items[0][0]

    for series, frame in sorted_items:
        parsed_datetime = parse_iso_datetime(frame.acquisition_datetime_iso)
        if parsed_datetime is not None and first_datetime is None:
            first_datetime = parsed_datetime
        if parsed_datetime is None and frame.relative_time_seconds is not None and first_relative_seconds is None:
            first_relative_seconds = frame.relative_time_seconds

        elapsed_seconds: float | None = None
        if parsed_datetime is not None and first_datetime is not None:
            elapsed_seconds = (parsed_datetime - first_datetime).total_seconds()
        elif frame.relative_time_seconds is not None and first_relative_seconds is not None:
            elapsed_seconds = frame.relative_time_seconds - first_relative_seconds

        frames.append(
            FAViewerFrame(
                image=np.asarray(frame.image),
                laterality=series.laterality,
                source_file=frame.source_file,
                series_uid=series.series_uid,
                series_number=series.series_number,
                series_description=series.series_description,
                protocol_name=series.protocol_name,
                frame_index=frame.frame_index,
                acquisition_source=frame.acquisition_source,
                acquisition_datetime_raw=frame.acquisition_datetime_raw,
                acquisition_datetime_iso=frame.acquisition_datetime_iso,
                acquisition_date_iso=frame.acquisition_date_iso,
                acquisition_time_iso=frame.acquisition_time_iso,
                relative_time_seconds=frame.relative_time_seconds,
                elapsed_seconds=elapsed_seconds,
            )
        )

    first_datetime_iso = next(
        (frame.acquisition_datetime_iso for frame in frames if frame.acquisition_datetime_iso),
        None,
    )
    last_datetime_iso = next(
        (frame.acquisition_datetime_iso for frame in reversed(frames) if frame.acquisition_datetime_iso),
        None,
    )
    frame_count = len(frames)
    series_count = len({series.series_uid for series, _ in sorted_items})

    time_suffix = ""
    if first_datetime_iso and last_datetime_iso and first_datetime_iso != last_datetime_iso:
        time_suffix = f" | {first_datetime_iso.split('T')[-1]} → {last_datetime_iso.split('T')[-1]}"
    elif first_datetime_iso:
        time_suffix = f" | {first_datetime_iso.split('T')[-1]}"

    label = f"{laterality_display_text(laterality)} | {frame_count} frames | {series_count} series{time_suffix}"
    return FAViewerTrack(
        key=key,
        label=label,
        laterality=laterality,
        patient_name=first_series.patient_name,
        patient_id=first_series.patient_id,
        patient_sex=first_series.patient_sex,
        patient_birth_date_iso=first_series.patient_birth_date_iso,
        frame_count=frame_count,
        series_count=series_count,
        first_datetime_iso=first_datetime_iso,
        last_datetime_iso=last_datetime_iso,
        frames=frames,
    )


def build_fa_viewer_tracks(series_list: list[SeriesRecord]) -> list[FAViewerTrack]:
    grouped: dict[str, list[tuple[SeriesRecord, FrameRecord]]] = {}

    for series in series_list:
        laterality = clean_text(series.laterality).upper() or "UNKNOWN"
        grouped.setdefault(laterality, [])
        for frame in series.frames:
            grouped[laterality].append((series, frame))

    tracks: list[FAViewerTrack] = []
    if len(grouped) > 1:
        all_items = [item for items in grouped.values() for item in items]
        tracks.append(build_viewer_track("ALL", "ALL", all_items))

    for laterality in sorted(grouped.keys()):
        tracks.append(build_viewer_track(laterality, laterality, grouped[laterality]))

    return [track for track in tracks if track.frames]


def apply_image_window(image: np.ndarray, contrast_percent: int, brightness_offset: int) -> np.ndarray:
    display_image = normalize_to_uint8(image)
    adjusted = display_image.astype(np.float32) * (contrast_percent / 100.0) + brightness_offset
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def frame_time_text(frame: FAViewerFrame) -> str:
    if frame.acquisition_datetime_iso:
        return frame.acquisition_datetime_iso
    if frame.acquisition_time_iso:
        return frame.acquisition_time_iso
    if frame.acquisition_datetime_raw:
        return frame.acquisition_datetime_raw
    if frame.relative_time_seconds is not None:
        return f"+{frame.relative_time_seconds:.3f}s"
    return "-"


def build_frame_metadata_text(track: FAViewerTrack, frame: FAViewerFrame, frame_position: int) -> str:
    payload = {
        "track": {
            "key": track.key,
            "label": track.label,
            "laterality": track.laterality,
            "laterality_label": laterality_display_text(track.laterality),
            "patient_name": track.patient_name,
            "patient_id": track.patient_id,
            "patient_sex": track.patient_sex,
            "patient_birth_date_iso": track.patient_birth_date_iso,
            "frame_count": track.frame_count,
            "series_count": track.series_count,
            "first_datetime_iso": track.first_datetime_iso,
            "last_datetime_iso": track.last_datetime_iso,
        },
        "frame": {
            "index": frame_position + 1,
            "source_file": frame.source_file,
            "series_uid": frame.series_uid,
            "series_number": frame.series_number,
            "series_description": frame.series_description,
            "protocol_name": frame.protocol_name,
            "frame_index_in_file": frame.frame_index,
            "laterality": frame.laterality,
            "time_text": frame_time_text(frame),
            "acquisition_source": frame.acquisition_source,
            "acquisition_datetime_raw": frame.acquisition_datetime_raw,
            "acquisition_datetime_iso": frame.acquisition_datetime_iso,
            "acquisition_date_iso": frame.acquisition_date_iso,
            "acquisition_time_iso": frame.acquisition_time_iso,
            "relative_time_seconds": frame.relative_time_seconds,
            "elapsed_seconds": frame.elapsed_seconds,
            "shape": list(frame.image.shape),
            "dtype": str(frame.image.dtype),
            "min_value": float(np.min(frame.image)),
            "max_value": float(np.max(frame.image)),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def summarize_tracks_for_console(tracks: list[FAViewerTrack]) -> list[str]:
    lines: list[str] = []
    for index, track in enumerate(tracks, start=1):
        lines.append(
            f"[{index}] {track.label} | first={track.first_datetime_iso or '-'} | last={track.last_datetime_iso or '-'}"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zeiss FA Qt viewer.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Zeiss FA directory or DICOM file. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--all-image-series",
        action="store_true",
        help="Disable FA keyword filtering and load all image series.",
    )
    return parser.parse_args()


def run_qt_viewer(input_path: Path, *, prefer_fa_only: bool) -> None:
    import matplotlib

    matplotlib.use("Qt5Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt5.QtCore import QSettings, Qt, QTimer
    from PyQt5.QtGui import QFont, QKeySequence
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
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
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    class ViewerCanvas(FigureCanvas):
        def __init__(self):
            figure = Figure(figsize=(12, 8), facecolor="#111827")
            self.figure = figure
            self.ax_image = figure.add_subplot(2, 1, 1)
            self.ax_timeline = figure.add_subplot(2, 1, 2)
            figure.subplots_adjust(left=0.04, right=0.985, top=0.95, bottom=0.07, hspace=0.24)
            super().__init__(figure)
            self._style_axes()

        def _style_axes(self):
            for axis in (self.ax_image, self.ax_timeline):
                axis.set_facecolor("#0B1220")
                axis.tick_params(colors="#CBD5E1")
                for spine in axis.spines.values():
                    spine.set_color("#334155")

        def clear_views(self):
            self.ax_image.clear()
            self.ax_timeline.clear()
            self._style_axes()
            self.ax_image.set_title("FA frame", color="#E5E7EB")
            self.ax_timeline.set_title("Timeline", color="#E5E7EB")
            self.draw_idle()

    class FAViewerWindow(QMainWindow):
        def __init__(self, startup_path: Path, *, prefer_fa_only: bool):
            super().__init__()
            self.settings = QSettings("OpenAI", "ZeissFAViewer")
            self.prefer_fa_only = prefer_fa_only
            self.series_list: list[SeriesRecord] = []
            self.tracks: list[FAViewerTrack] = []
            self.current_track_index = 0
            self.current_frame_index = 0
            self.playing = False
            self.canvas = ViewerCanvas()
            self.play_timer = QTimer(self)
            self.play_timer.timeout.connect(self.advance_frame)

            self.setWindowTitle("Zeiss FA Viewer")
            self.resize(1500, 920)
            self.setFont(QFont("Microsoft YaHei UI", 10))
            self._apply_styles()
            self.setStatusBar(QStatusBar(self))

            self._build_controls()
            self._build_toolbar()
            self._build_layout()
            self._install_shortcuts()
            self._set_empty_state()

            self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)

            if startup_path:
                self.path_edit.setText(str(startup_path))
            remembered_dir = self.settings.value("last_dir", "", type=str)
            if not self.path_edit.text().strip() and remembered_dir:
                self.path_edit.setText(remembered_dir)

            initial_path = self.path_edit.text().strip()
            if initial_path and Path(initial_path).exists():
                self.load_path(initial_path)

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
                QLabel#SummaryLabel, QLabel#PathLabel, QLabel#FrameInfo {
                    color: #CBD5E1;
                }
                """
            )

        def _build_controls(self):
            self.path_edit = QLineEdit()
            self.path_edit.returnPressed.connect(self.reload_from_path)

            self.open_button = QPushButton("Open")
            self.open_button.clicked.connect(self.open_path_dialog)

            self.reload_button = QPushButton("Reload")
            self.reload_button.clicked.connect(self.reload_from_path)

            self.sequence_combo = QComboBox()
            self.sequence_combo.currentIndexChanged.connect(self.on_track_changed)

            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setSingleStep(1)
            self.frame_slider.valueChanged.connect(self.on_frame_changed)

            self.frame_spin = QSpinBox()
            self.frame_spin.setMinimum(0)
            self.frame_spin.valueChanged.connect(self.on_frame_changed)

            self.prev_button = QPushButton("Prev")
            self.prev_button.clicked.connect(self.previous_frame)

            self.next_button = QPushButton("Next")
            self.next_button.clicked.connect(self.next_frame)

            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self.toggle_playback)

            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 30)
            self.fps_spin.setValue(4)
            self.fps_spin.valueChanged.connect(self._refresh_timer)

            self.contrast_slider = QSlider(Qt.Horizontal)
            self.contrast_slider.setRange(50, 200)
            self.contrast_slider.setValue(100)
            self.contrast_slider.valueChanged.connect(self.redraw_views)

            self.brightness_slider = QSlider(Qt.Horizontal)
            self.brightness_slider.setRange(-80, 80)
            self.brightness_slider.setValue(0)
            self.brightness_slider.valueChanged.connect(self.redraw_views)

            self.path_label = QLabel("No dataset loaded")
            self.path_label.setWordWrap(True)
            self.path_label.setObjectName("PathLabel")
            self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

            self.summary_label = QLabel("Ready to load Zeiss FA data")
            self.summary_label.setWordWrap(True)
            self.summary_label.setObjectName("SummaryLabel")

            self.frame_info_label = QLabel("Frame: -")
            self.frame_info_label.setObjectName("FrameInfo")

            self.track_info_label = QLabel("Track: -")
            self.time_info_label = QLabel("Time: -")
            self.absolute_time_label = QLabel("Absolute: -")
            self.elapsed_time_label = QLabel("Elapsed: -")
            self.source_info_label = QLabel("Source: -")
            self.eye_label = QLabel("Eye: -")
            self.patient_name_label = QLabel("Name: -")
            self.patient_id_label = QLabel("ID: -")
            self.patient_sex_label = QLabel("Sex: -")
            self.patient_birth_label = QLabel("DOB: -")
            self.absolute_time_label.setWordWrap(True)
            self.elapsed_time_label.setWordWrap(True)
            self.source_info_label.setWordWrap(True)
            self.eye_label.setWordWrap(True)
            self.patient_name_label.setWordWrap(True)
            self.patient_id_label.setWordWrap(True)
            self.patient_sex_label.setWordWrap(True)
            self.patient_birth_label.setWordWrap(True)
            stable_labels = [
                self.frame_info_label,
                self.track_info_label,
                self.time_info_label,
                self.absolute_time_label,
                self.elapsed_time_label,
                self.source_info_label,
                self.eye_label,
                self.patient_name_label,
                self.patient_id_label,
                self.patient_sex_label,
                self.patient_birth_label,
            ]
            for label in stable_labels:
                label.setWordWrap(False)
                label.setMinimumHeight(24)
                label.setMaximumHeight(24)

            self.metadata_edit = QPlainTextEdit()
            self.metadata_edit.setReadOnly(True)
            self.metadata_edit.setFont(QFont("Consolas", 10))

        def _build_toolbar(self):
            toolbar = QToolBar("Actions", self)
            toolbar.setMovable(False)
            self.addToolBar(toolbar)

            open_action = QAction("Open", self)
            open_action.triggered.connect(self.open_path_dialog)
            toolbar.addAction(open_action)

            reload_action = QAction("Reload", self)
            reload_action.triggered.connect(self.reload_from_path)
            toolbar.addAction(reload_action)

            play_action = QAction("Play/Pause", self)
            play_action.triggered.connect(self.toggle_playback)
            toolbar.addAction(play_action)

        def _build_layout(self):
            path_group = QGroupBox("Input")
            path_layout = QVBoxLayout(path_group)

            path_row = QHBoxLayout()
            path_row.addWidget(self.path_edit, stretch=1)
            path_row.addWidget(self.open_button)
            path_row.addWidget(self.reload_button)
            path_layout.addLayout(path_row)
            path_layout.addWidget(self.path_label)

            navigation_group = QGroupBox("Navigation")
            navigation_layout = QGridLayout(navigation_group)
            navigation_layout.addWidget(QLabel("Track"), 0, 0)
            navigation_layout.addWidget(self.sequence_combo, 0, 1, 1, 3)
            navigation_layout.addWidget(QLabel("Frame"), 1, 0)
            navigation_layout.addWidget(self.frame_slider, 1, 1, 1, 3)
            navigation_layout.addWidget(self.prev_button, 2, 0)
            navigation_layout.addWidget(self.frame_spin, 2, 1)
            navigation_layout.addWidget(self.next_button, 2, 2)
            navigation_layout.addWidget(self.play_button, 2, 3)
            navigation_layout.addWidget(QLabel("FPS"), 3, 0)
            navigation_layout.addWidget(self.fps_spin, 3, 1)
            navigation_layout.addWidget(self.frame_info_label, 3, 2, 1, 2)

            display_group = QGroupBox("Display")
            display_layout = QFormLayout(display_group)
            display_layout.addRow("Contrast", self.contrast_slider)
            display_layout.addRow("Brightness", self.brightness_slider)
            display_layout.addRow("Track info", self.track_info_label)
            display_layout.addRow("Eye", self.eye_label)
            display_layout.addRow("Time info", self.time_info_label)
            display_layout.addRow("Absolute time", self.absolute_time_label)
            display_layout.addRow("Elapsed", self.elapsed_time_label)
            display_layout.addRow("Source", self.source_info_label)

            patient_group = QGroupBox("Patient")
            patient_layout = QFormLayout(patient_group)
            patient_layout.addRow("Name", self.patient_name_label)
            patient_layout.addRow("ID", self.patient_id_label)
            patient_layout.addRow("Sex", self.patient_sex_label)
            patient_layout.addRow("Birth date", self.patient_birth_label)

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
            control_layout.addWidget(path_group)
            control_layout.addWidget(navigation_group)
            control_layout.addWidget(display_group)
            control_layout.addWidget(patient_group)
            control_layout.addWidget(summary_group)
            control_layout.addWidget(metadata_group, stretch=1)

            view_widget = QWidget()
            view_layout = QVBoxLayout(view_widget)
            view_layout.setContentsMargins(0, 0, 0, 0)
            view_layout.addWidget(self.canvas)

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

        def _install_shortcuts(self):
            QShortcut(QKeySequence(Qt.Key_Left), self, self.previous_frame)
            QShortcut(QKeySequence(Qt.Key_Right), self, self.next_frame)
            QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_playback)
            QShortcut(QKeySequence("Ctrl+O"), self, self.open_path_dialog)

        def _set_empty_state(self):
            self.sequence_combo.blockSignals(True)
            self.sequence_combo.clear()
            self.sequence_combo.blockSignals(False)

            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)

            self.frame_spin.blockSignals(True)
            self.frame_spin.setMaximum(0)
            self.frame_spin.setValue(0)
            self.frame_spin.blockSignals(False)

            self.frame_info_label.setText("Frame: -")
            self.track_info_label.setText("Track: -")
            self.eye_label.setText("Eye: -")
            self.time_info_label.setText("Time: -")
            self.absolute_time_label.setText("Absolute: -")
            self.elapsed_time_label.setText("Elapsed: -")
            self.source_info_label.setText("Source: -")
            self.patient_name_label.setText("Name: -")
            self.patient_id_label.setText("ID: -")
            self.patient_sex_label.setText("Sex: -")
            self.patient_birth_label.setText("DOB: -")
            self.metadata_edit.setPlainText("")
            self.canvas.clear_views()

        def current_track(self) -> FAViewerTrack | None:
            if not self.tracks:
                return None
            return self.tracks[self.current_track_index]

        def current_frame(self) -> FAViewerFrame | None:
            track = self.current_track()
            if track is None or not track.frames:
                return None
            return track.frames[self.current_frame_index]

        def _refresh_timer(self):
            if self.playing:
                interval_ms = max(1, int(round(1000 / max(self.fps_spin.value(), 1))))
                self.play_timer.start(interval_ms)

        def open_path_dialog(self):
            start_dir = (
                self.path_edit.text().strip()
                or self.settings.value("last_dir", "", type=str)
                or str(DEFAULT_INPUT_PATH)
            )
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Zeiss FA folder",
                start_dir,
            )
            if not directory:
                return
            self.path_edit.setText(directory)
            self.load_path(directory)

        def reload_from_path(self):
            self.load_path(self.path_edit.text().strip())

        def load_path(self, path_text: str):
            if not path_text:
                QMessageBox.warning(self, "Missing path", "Please choose a Zeiss FA folder or DICOM file.")
                return

            path = Path(path_text).expanduser().resolve()
            try:
                series_list = load_zeiss_fa_series(path, prefer_fa_only=self.prefer_fa_only)
                tracks = build_fa_viewer_tracks(series_list)
            except Exception as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
                return

            if not tracks:
                QMessageBox.warning(self, "No frames", "No FA frames were found.")
                return

            self.series_list = series_list
            self.tracks = tracks
            self.current_track_index = 0
            self.current_frame_index = 0
            self.path_label.setText(str(path))
            self.settings.setValue("last_dir", str(path if path.is_dir() else path.parent))

            self.sequence_combo.blockSignals(True)
            self.sequence_combo.clear()
            for track in self.tracks:
                self.sequence_combo.addItem(track.label)
            self.sequence_combo.blockSignals(False)
            self.sequence_combo.setCurrentIndex(0)

            summary_lines = summarize_tracks_for_console(self.tracks)
            summary_text = "Loaded tracks:\n" + "\n".join(summary_lines[:6])
            if len(summary_lines) > 6:
                summary_text += "\n..."
            patient_track = self.tracks[0]
            patient_text = (
                f"\n\nPatient: {patient_track.patient_name or '-'} | "
                f"ID: {patient_track.patient_id or '-'} | "
                f"Sex: {patient_track.patient_sex or '-'} | "
                f"DOB: {patient_track.patient_birth_date_iso or '-'}"
            )
            summary_text += patient_text
            self.summary_label.setText(summary_text)
            self.refresh_track()

        def refresh_track(self):
            track = self.current_track()
            if track is None:
                self._set_empty_state()
                return

            self.current_frame_index = min(self.current_frame_index, max(track.frame_count - 1, 0))

            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(max(track.frame_count - 1, 0))
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)

            self.frame_spin.blockSignals(True)
            self.frame_spin.setMaximum(max(track.frame_count - 1, 0))
            self.frame_spin.setValue(self.current_frame_index)
            self.frame_spin.setSuffix(f" / {track.frame_count}")
            self.frame_spin.blockSignals(False)

            self.track_info_label.setText(
                f"{track.label} | first={track.first_datetime_iso or '-'} | last={track.last_datetime_iso or '-'}"
            )
            self.eye_label.setText(laterality_display_text(track.laterality))
            self.patient_name_label.setText(track.patient_name or "-")
            self.patient_id_label.setText(track.patient_id or "-")
            self.patient_sex_label.setText(track.patient_sex or "-")
            self.patient_birth_label.setText(track.patient_birth_date_iso or "-")
            self.redraw_views()

        def redraw_views(self):
            track = self.current_track()
            frame = self.current_frame()
            if track is None or frame is None:
                self._set_empty_state()
                return

            display_image = apply_image_window(
                frame.image,
                contrast_percent=self.contrast_slider.value(),
                brightness_offset=self.brightness_slider.value(),
            )

            self.canvas.ax_image.clear()
            self.canvas.ax_timeline.clear()
            self.canvas._style_axes()

            time_text = frame_time_text(frame)
            elapsed_text = (
                f"{frame.elapsed_seconds:.3f}s"
                if frame.elapsed_seconds is not None
                else "-"
            )

            if display_image.ndim == 2:
                self.canvas.ax_image.imshow(display_image, cmap="gray", origin="upper")
            else:
                self.canvas.ax_image.imshow(display_image[:, :, :3], origin="upper")
            self.canvas.ax_image.axis("off")
            self.canvas.ax_image.set_title(
                f"Eye: {laterality_display_text(track.laterality)} | Frame {self.current_frame_index + 1}/{track.frame_count} | {time_text}",
                color="#E5E7EB",
            )

            has_elapsed_time = any(item.elapsed_seconds is not None for item in track.frames)
            x_values = [
                item.elapsed_seconds if item.elapsed_seconds is not None else float(index)
                for index, item in enumerate(track.frames)
            ]
            colors = ["#64748B"] * len(track.frames)
            sizes = [36] * len(track.frames)
            colors[self.current_frame_index] = "#F97316"
            sizes[self.current_frame_index] = 80

            self.canvas.ax_timeline.scatter(
                x_values,
                [1.0] * len(x_values),
                c=colors,
                s=sizes,
                alpha=0.95,
            )

            current_series = None
            for index, item in enumerate(track.frames):
                if item.source_file != current_series:
                    current_series = item.source_file
                    self.canvas.ax_timeline.axvline(
                        x_values[index],
                        color="#334155",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.7,
                    )

            self.canvas.ax_timeline.set_yticks([])
            self.canvas.ax_timeline.set_ylim(0.7, 1.3)
            self.canvas.ax_timeline.set_xlabel(
                "Elapsed seconds" if has_elapsed_time else "Frame order",
                color="#CBD5E1",
            )
            self.canvas.ax_timeline.set_title("Acquisition timeline", color="#E5E7EB")
            self.canvas.ax_timeline.grid(True, axis="x", color="#1E293B", linewidth=0.8, alpha=0.8)
            current_x = x_values[self.current_frame_index]
            self.canvas.ax_timeline.axvline(
                current_x,
                color="#F97316",
                linewidth=1.4,
                alpha=0.9,
            )

            self.frame_info_label.setText(f"Frame: {self.current_frame_index + 1}/{track.frame_count}")
            self.time_info_label.setText(
                f"{time_text} | elapsed={elapsed_text}"
            )
            self.eye_label.setText(laterality_display_text(track.laterality))
            self.absolute_time_label.setText(frame.acquisition_datetime_iso or frame.acquisition_time_iso or "-")
            self.elapsed_time_label.setText(elapsed_text)
            self.source_info_label.setText(
                f"{frame.source_file} | Series {frame.series_number if frame.series_number is not None else '-'}"
            )
            self.metadata_edit.setPlainText(
                build_frame_metadata_text(track, frame, self.current_frame_index)
            )
            self.statusBar().showMessage(
                f"{track.patient_name or '-'} | {frame.source_file} | Eye={laterality_display_text(track.laterality)} | frame {self.current_frame_index + 1}/{track.frame_count} | {time_text}"
            )
            self.canvas.draw_idle()

        def set_frame_index(self, index: int):
            track = self.current_track()
            if track is None or not track.frames:
                return

            self.current_frame_index = max(0, min(index, track.frame_count - 1))

            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)

            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(self.current_frame_index)
            self.frame_spin.blockSignals(False)

            self.redraw_views()

        def previous_frame(self):
            self.set_frame_index(self.current_frame_index - 1)

        def next_frame(self):
            self.set_frame_index(self.current_frame_index + 1)

        def advance_frame(self):
            track = self.current_track()
            if track is None or not track.frames:
                self.toggle_playback(force_stop=True)
                return
            next_index = (self.current_frame_index + 1) % track.frame_count
            self.set_frame_index(next_index)

        def toggle_playback(self, force_stop: bool = False):
            if force_stop:
                self.playing = False
            else:
                self.playing = not self.playing

            if self.playing:
                self.play_button.setText("Pause")
                self._refresh_timer()
            else:
                self.play_button.setText("Play")
                self.play_timer.stop()

        def on_track_changed(self, index: int):
            if index < 0 or index >= len(self.tracks):
                return
            self.current_track_index = index
            self.current_frame_index = 0
            self.refresh_track()

        def on_frame_changed(self, value: int):
            self.set_frame_index(int(value))

        def on_canvas_scroll(self, event):
            if event.button == "up":
                self.next_frame()
            elif event.button == "down":
                self.previous_frame()

    application = QApplication(sys.argv)
    window = FAViewerWindow(input_path, prefer_fa_only=prefer_fa_only)
    window.show()
    sys.exit(application.exec_())


def main() -> None:
    args = parse_args()
    run_qt_viewer(
        Path(args.input_path).expanduser().resolve(),
        prefer_fa_only=not args.all_image_series,
    )


if __name__ == "__main__":
    main()
