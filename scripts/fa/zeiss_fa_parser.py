from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
import unicodedata
from pydicom.pixel_data_handlers import convert_color_space

DICOM_SUFFIXES = {".dcm", ".dicom"}
VL_PHOTOGRAPHIC_IMAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.4"
OPHTHALMIC_PHOTOGRAPHY_8BIT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"
FA_IMAGE_SOP_CLASS_UIDS = {
    VL_PHOTOGRAPHIC_IMAGE_SOP_CLASS_UID,
    OPHTHALMIC_PHOTOGRAPHY_8BIT_SOP_CLASS_UID,
}
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


def extract_patient_sex(dataset: pydicom.dataset.Dataset) -> str:
    direct_value = format_patient_sex(getattr(dataset, "PatientSex", ""))
    if direct_value:
        return direct_value

    fallback_value, _ = find_first_value(
        dataset,
        ["PatientSex", "PatientGender", "Sex"],
        recursive=True,
    )
    return format_patient_sex(fallback_value)


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
        iso_date=date_to_iso(
            date_value if date_value is not None else (datetime_value.date() if datetime_value else None)),
        iso_time=time_to_iso(time_value if time_value is not None else (
            datetime_value.timetz().replace(tzinfo=None) if datetime_value else None)),
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


def classify_dicom(
    ds: pydicom.dataset.FileDataset,
    *,
    header_only: bool = False,
) -> str:
    number_of_frames = parse_int(getattr(ds, "NumberOfFrames", None))
    rows = parse_int(getattr(ds, "Rows", None))
    columns = parse_int(getattr(ds, "Columns", None))

    if not header_only and "PixelData" not in ds:
        return "non_image_dicom"

    if number_of_frames is not None and number_of_frames > 1:
        return "multi_frame_image"

    if rows is not None and columns is not None:
        return "single_frame_image"

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
    if clean_text(getattr(ds, "SOPClassUID", "")) in FA_IMAGE_SOP_CLASS_UIDS:
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

        classification = classify_dicom(header_ds, header_only=True)
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
                "patient_sex": extract_patient_sex(header_ds),
                "patient_birth_date_raw": clean_text(getattr(header_ds, "PatientBirthDate", "")),
                "patient_birth_date_iso": date_to_iso(parse_dicom_date(getattr(header_ds, "PatientBirthDate", ""))),
                "series_uid": clean_text(getattr(header_ds, "SeriesInstanceUID", "")) or series_key,
                "series_number": parse_int(getattr(header_ds, "SeriesNumber", None)),
                "series_description": clean_text(getattr(header_ds, "SeriesDescription", "")),
                "study_description": clean_text(getattr(header_ds, "StudyDescription", "")),
                "protocol_name": clean_text(getattr(header_ds, "ProtocolName", "")),
                "modality": clean_text(getattr(header_ds, "Modality", "")),
                "laterality": clean_text(getattr(header_ds, "Laterality", "")) or clean_text(
                    getattr(header_ds, "ImageLaterality", "")),
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

        if not entry["patient_sex"]:
            entry["patient_sex"] = extract_patient_sex(header_ds)
        entry["fa_score"] = max(entry["fa_score"], score_fa_series(header_ds))
        entry["files_with_headers"].append((file_path, header_ds))

    if image_file_count == 0:
        raise RuntimeError(f"No decodable image DICOM files found in {input_path}")

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
            if classify_dicom(ds, header_only=False) == "non_image_dicom":
                continue
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
        raise RuntimeError(f"No exportable image series found in {input_path}")

    series_records.sort(
        key=lambda series: (
            -series.fa_score,
            series.series_number if series.series_number is not None else 10 ** 9,
            series.series_description.lower(),
            series.series_uid,
        )
    )

    if prefer_fa_only:
        fa_series = [series for series in series_records if series.fa_score > 0]
        if fa_series:
            return fa_series

    return series_records


def _display_width(text: object) -> int:
    width = 0
    for character in str(text):
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def _pad_display_text(text: object, width: int) -> str:
    rendered = str(text)
    padding = max(0, width - _display_width(rendered))
    return rendered + (" " * padding)


def _display_datetime_text(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("T", " ", 1)


def dump_series_summary(series_list: list[SeriesRecord]) -> None:
    headers = [
        "index",
        "series_no",
        "laterality",
        "fa_score",
        "frames",
        "study_time",
        "series_time",
        "description",
    ]
    rows = []
    for index, series in enumerate(series_list, start=1):
        rows.append(
            [
                str(index),
                "-" if series.series_number is None else str(series.series_number),
                series.laterality or "-",
                str(series.fa_score),
                str(len(series.frames)),
                _display_datetime_text(series.study_datetime_iso or series.study_datetime_raw) or "-",
                _display_datetime_text(series.series_datetime_iso or series.series_datetime_raw) or "-",
                series.series_description or series.protocol_name or series.study_description or "-",
            ]
        )

    widths = [
        max(_display_width(header), *(_display_width(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ] if rows else [_display_width(header) for header in headers]

    print("  ".join(_pad_display_text(header, widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(_pad_display_text(value, widths[i]) for i, value in enumerate(row)))
    print()


def dump_series_detail(series: SeriesRecord) -> None:
    rows = [
        ("patient_name", series.patient_name or "-"),
        ("patient_id", series.patient_id or "-"),
        ("patient_sex", series.patient_sex or "-"),
        ("birth_date", series.patient_birth_date_iso or series.patient_birth_date_raw or "-"),
        ("series_uid", series.series_uid or "-"),
        ("series_number", "-" if series.series_number is None else str(series.series_number)),
        ("series_description", series.series_description or "-"),
        ("study_description", series.study_description or "-"),
        ("protocol_name", series.protocol_name or "-"),
        ("modality", series.modality or "-"),
        ("laterality", series.laterality or "-"),
        ("sop_class_uid", series.sop_class_uid or "-"),
        ("photometric", series.photometric_interpretation or "-"),
        ("size", f"{series.rows or '-'} x {series.columns or '-'}"),
        ("fa_score", str(series.fa_score)),
        ("study_datetime", _display_datetime_text(series.study_datetime_iso or series.study_datetime_raw) or "-"),
        ("series_datetime", _display_datetime_text(series.series_datetime_iso or series.series_datetime_raw) or "-"),
        ("file_count", str(len(series.files))),
        ("frame_count", str(len(series.frames))),
    ]
    key_width = max(_display_width(key) for key, _ in rows)
    for key, value in rows:
        print(f"{_pad_display_text(key, key_width)} : {value}")
    print()


def dump_frames(series: SeriesRecord) -> None:
    headers = [
        "index",
        "source_file",
        "frame_idx",
        "instance",
        "acquisition",
        "relative_s",
        "source",
    ]
    rows = []
    for index, frame in enumerate(series.frames, start=1):
        acquisition = (
            _display_datetime_text(frame.acquisition_datetime_iso)
            or (
                _display_datetime_text(f"{frame.acquisition_date_iso}T{frame.acquisition_time_iso}")
                if frame.acquisition_date_iso and frame.acquisition_time_iso
                else None
            )
            or frame.acquisition_datetime_raw
            or (f"{frame.acquisition_date} {frame.acquisition_time}".strip())
            or "-"
        )
        rows.append(
            [
                str(index),
                frame.source_file,
                str(frame.frame_index),
                "-" if frame.instance_number is None else str(frame.instance_number),
                acquisition,
                "" if frame.relative_time_seconds is None else f"{frame.relative_time_seconds:.3f}",
                frame.acquisition_source or "-",
            ]
        )

    widths = [
        max(_display_width(header), *(_display_width(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ] if rows else [_display_width(header) for header in headers]

    print("  ".join(_pad_display_text(header, widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(_pad_display_text(value, widths[i]) for i, value in enumerate(row)))
    print()


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DICOM_SUFFIXES",
    "FA_KEYWORDS",
    "FrameRecord",
    "ParsedTimeInfo",
    "SeriesRecord",
    "build_file_sort_key",
    "build_series_key",
    "classify_dicom",
    "clean_text",
    "decode_pixel_array",
    "discover_candidate_files",
    "dump_frames",
    "dump_series_detail",
    "dump_series_summary",
    "extract_relative_frame_times",
    "extract_time_info",
    "load_zeiss_fa_series",
    "normalize_to_uint8",
    "safe_dcmread",
    "score_fa_series",
    "split_frames",
    "summarize_series_timing",
]

if __name__ == "__main__":
    example_path = Path(r"E:\Data\OCT\蔡司FA\2")
    print(f"Testing Zeiss FA parser with: {example_path}")
    if not example_path.exists():
        print("Example path not found.")
    else:
        try:
            series_list = load_zeiss_fa_series(example_path, prefer_fa_only=True)
        except Exception as exc:
            print(f"Failed to parse example path: {exc}")
        else:
            print(f"Series count: {len(series_list)}")
            print()
            dump_series_summary(series_list)
            for index, series in enumerate(series_list, start=1):
                print(f"[Series {index}]")
                dump_series_detail(series)
                dump_frames(series)
