from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import unicodedata
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oct_converter.readers import E2E
from oct_converter.readers.binary_structs import e2e_binary

DEFAULT_INPUT_PATH = Path(r"E:\Data\OCT\海德堡\海德堡FA.E2E")


@dataclass
class HeidelbergFAStudyInfo:
    source_file: Path | None = None
    patient_first_name: str = ""
    patient_surname: str = ""
    patient_id: str = ""
    sex: str = ""
    birth_date: str = ""
    device_model: str = ""
    device_short_name: str = ""
    timezone1: str = ""
    timezone2: str = ""
    timezone_offset_hours: float | None = None
    study_datetime_local: datetime | None = None
    frame_count: int = 0
    series_count: int = 0
    modality_summary: str = ""
    laterality_summary: str = ""

    @property
    def patient_name(self) -> str:
        combined = " ".join(part for part in [self.patient_surname, self.patient_first_name] if part)
        return combined.strip()

    @property
    def sex_display(self) -> str:
        return {
            "M": "男",
            "F": "女",
            "O": "其他",
        }.get(self.sex.upper(), self.sex or "-")

    @property
    def device_display(self) -> str:
        if self.device_model and self.device_short_name:
            if self.device_short_name.lower() in self.device_model.lower():
                return self.device_model
            return f"{self.device_model} ({self.device_short_name})"
        return self.device_model or self.device_short_name or "-"

    @property
    def timezone_display(self) -> str:
        parts = [part for part in [self.timezone1, self.timezone2] if part]
        return " / ".join(parts) if parts else "-"

    @property
    def study_datetime_iso(self) -> str:
        return self.study_datetime_local.isoformat(sep=" ") if self.study_datetime_local else "-"


@dataclass
class HeidelbergFAFrame:
    order_index: int
    image_id: str
    series_key: str
    patient_db_id: int
    study_id: int
    series_id: int
    slice_id: int
    series_frame_index: int
    modality: str
    modality_name: str
    scan_pattern: str
    examined_structure: str
    laterality: str
    timezone1: str
    timezone2: str
    time_of_day_ms: int | None
    raw_elapsed_ms: int | None
    raw_elapsed_ms_alt: int | None
    acquisition_datetime_local: datetime | None
    acquisition_source: str
    width: int
    height: int
    image: np.ndarray
    metadata_text: str = ""

    @property
    def size_display(self) -> str:
        return f"{self.width} × {self.height}"

    @property
    def laterality_display(self) -> str:
        return laterality_to_chinese(self.laterality)

    @property
    def modality_display(self) -> str:
        if self.modality and self.modality_name:
            return f"{self.modality} ({self.modality_name})"
        return self.modality or self.modality_name or "-"

    @property
    def series_display(self) -> str:
        return str(self.series_id)

    @property
    def sequence_display(self) -> str:
        slice_text = "-" if self.slice_id < 0 else str(self.slice_id)
        return f"{self.series_frame_index + 1} / slice {slice_text}"

    @property
    def structure_display(self) -> str:
        parts = [part for part in [self.examined_structure, self.scan_pattern] if part]
        return " / ".join(parts) if parts else "-"

    @property
    def timezone_display(self) -> str:
        parts = [part for part in [self.timezone1, self.timezone2] if part]
        return " / ".join(parts) if parts else "-"

    @property
    def acquisition_datetime_iso(self) -> str | None:
        if self.acquisition_datetime_local is None:
            return None
        return self.acquisition_datetime_local.isoformat(timespec="milliseconds")

    @property
    def acquisition_date_iso(self) -> str | None:
        if self.acquisition_datetime_local is None:
            return None
        return self.acquisition_datetime_local.date().isoformat()

    @property
    def acquisition_time_iso(self) -> str | None:
        if self.acquisition_datetime_local is None:
            return None
        return self.acquisition_datetime_local.time().isoformat(timespec="milliseconds")

    @property
    def time_display(self) -> str:
        if self.acquisition_datetime_local is not None:
            return self.acquisition_datetime_local.isoformat(sep=" ", timespec="milliseconds")
        if self.time_of_day_ms is not None:
            milliseconds = self.time_of_day_ms % 1000
            total_seconds = self.time_of_day_ms // 1000
            seconds = total_seconds % 60
            minutes = (total_seconds // 60) % 60
            hours = (total_seconds // 3600) % 24
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        return "-"


@dataclass
class Chunk39Timing:
    time_of_day_ms: int | None
    raw_elapsed_ms: int | None
    raw_elapsed_ms_alt: int | None
    timezone1: str
    timezone2: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heidelberg FA E2E Qt 查看器")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH) if DEFAULT_INPUT_PATH.exists() else None,
        help="海德堡 .E2E 文件路径，或包含 .E2E 的目录。",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="仅解析并打印结果，不启动 Qt。",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    text = str(value).replace("\x00", " ").strip()
    text = "".join(character for character in text if character.isprintable() or character.isspace())
    return " ".join(text.split())


def normalize_sex(value: str) -> str:
    return clean_text(value).upper()


def laterality_to_chinese(value: str) -> str:
    normalized = clean_text(value).upper()
    return {
        "R": "右眼",
        "L": "左眼",
        "OD": "右眼",
        "OS": "左眼",
        "ALL": "双眼",
        "UNKNOWN": "未标注",
    }.get(normalized, normalized or "-")


def modality_sort_key(value: str) -> tuple[int, str]:
    normalized = clean_text(value).upper()
    priority = {
        "FA": 0,
        "ICGA": 1,
        "BR": 2,
        "IR": 3,
    }.get(normalized, 99)
    return priority, normalized


def safe_slug(text: str) -> str:
    characters = []
    for character in clean_text(text):
        if character.isalnum() or character in {"-", "_"}:
            characters.append(character)
        else:
            characters.append("_")
    slug = "".join(characters).strip("_")
    return slug or "frame"


def parse_birth_date(reader: E2E, raw_value: Any) -> str:
    try:
        integer_value = int(raw_value)
    except (TypeError, ValueError):
        return clean_text(raw_value)

    if len(str(integer_value)) == 8:
        text = str(integer_value)
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"

    try:
        parsed = reader.julian_to_ymd((integer_value / 64) - 14558805)
    except Exception:
        return clean_text(raw_value)

    if isinstance(parsed, date):
        return parsed.isoformat()
    return clean_text(parsed)


def parse_ole_datetime(raw_value: float | None) -> datetime | None:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not (-100000 <= value <= 1000000):
        return None
    try:
        return datetime(1899, 12, 30) + timedelta(days=value)
    except OverflowError:
        return None


def normalize_timezone_name(value: str) -> str:
    return clean_text(value).casefold()


def parse_timezone_offset_hours(*values: str) -> float | None:
    candidates = [normalize_timezone_name(value) for value in values if clean_text(value)]
    if not candidates:
        return None

    mapping = {
        "china standard time": 8.0,
        "中国标准时间": 8.0,
        "beijing standard time": 8.0,
        "utc+08:00": 8.0,
        "gmt+08:00": 8.0,
        "china daylight time": 9.0,
        "中国夏令时": 9.0,
    }
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
        if "china" in candidate and "standard" in candidate:
            return 8.0
        if "china" in candidate and "daylight" in candidate:
            return 9.0
    return None


def parse_study_datetime_from_chunk58(payload: bytes) -> datetime | None:
    if len(payload) < 14:
        return None
    try:
        raw_value = struct.unpack("<d", payload[6:14])[0]
    except struct.error:
        return None
    return parse_ole_datetime(raw_value)


def parse_chunk39_timing(payload: bytes) -> Chunk39Timing:
    timezone1 = ""
    timezone2 = ""
    try:
        parsed = e2e_binary.time_data.parse(payload)
        timezone1 = clean_text(getattr(parsed, "timezone1", ""))
        timezone2 = clean_text(getattr(parsed, "timezone2", ""))
    except Exception:
        pass

    time_of_day_ms = None
    raw_elapsed_ms = None
    raw_elapsed_ms_alt = None
    if len(payload) >= 100:
        candidate = int.from_bytes(payload[96:100], byteorder="little", signed=False)
        if 0 <= candidate < 86_400_000:
            time_of_day_ms = candidate
    if len(payload) >= 364:
        raw_elapsed_ms = int.from_bytes(payload[360:364], byteorder="little", signed=False)
    if len(payload) >= 376:
        raw_elapsed_ms_alt = int.from_bytes(payload[372:376], byteorder="little", signed=False)

    return Chunk39Timing(
        time_of_day_ms=time_of_day_ms,
        raw_elapsed_ms=raw_elapsed_ms,
        raw_elapsed_ms_alt=raw_elapsed_ms_alt,
        timezone1=timezone1,
        timezone2=timezone2,
        source="chunk39[96:100]+timezone",
    )


def choose_frame_datetime(
    study_datetime_local: datetime | None,
    study_date_value: date | None,
    time_of_day_ms: int | None,
    timezone_offset_hours: float | None,
) -> tuple[datetime | None, str]:
    if study_date_value is None or time_of_day_ms is None:
        return None, ""

    midnight = datetime.combine(study_date_value, datetime.min.time())
    candidate_local = midnight + timedelta(milliseconds=time_of_day_ms)
    candidates = [(candidate_local, "study_date + chunk39_ms")]

    if timezone_offset_hours is not None:
        shifted = candidate_local + timedelta(hours=timezone_offset_hours)
        candidates.append((shifted, "study_date + chunk39_ms + timezone_offset"))

    if study_datetime_local is not None:
        best_datetime, best_source = min(
            candidates,
            key=lambda item: abs((item[0] - study_datetime_local).total_seconds()),
        )
        return best_datetime, best_source

    if len(candidates) > 1:
        return candidates[-1]
    return candidates[0]


def resolve_input_file(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    if path.is_file():
        return path

    if path.is_dir():
        candidates = sorted(
            list(path.glob("*.E2E")) + list(path.glob("*.e2e")),
            key=lambda item: ("fa" not in item.stem.lower(), item.name.lower()),
        )
        return candidates[0] if candidates else path

    return path


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - min_value) * (255.0 / (max_value - min_value))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_image_window(image: np.ndarray, contrast_percent: int, brightness_offset: int) -> np.ndarray:
    display_image = normalize_to_uint8(image)
    adjusted = display_image.astype(np.float32) * (contrast_percent / 100.0) + brightness_offset
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def save_image_array(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = normalize_to_uint8(image)
    Image.fromarray(array).save(output_path)


def modality_summary(frames: list[HeidelbergFAFrame]) -> str:
    counts = Counter(frame.modality or "Unknown" for frame in frames)
    ordered = sorted(counts.items(), key=lambda item: modality_sort_key(item[0]))
    return ", ".join(f"{key} {value}" for key, value in ordered) or "-"


def laterality_summary(frames: list[HeidelbergFAFrame]) -> str:
    counts = Counter(frame.laterality_display for frame in frames)
    ordered = sorted(counts.items(), key=lambda item: item[0])
    return ", ".join(f"{key} {value}" for key, value in ordered) or "-"


def build_frame_metadata_text(
    input_file: Path,
    study_info: HeidelbergFAStudyInfo,
    frame: HeidelbergFAFrame,
) -> str:
    payload = {
        "source_file": str(input_file),
        "patient": {
            "name": study_info.patient_name or "",
            "patient_id": study_info.patient_id,
            "sex": study_info.sex,
            "birth_date": study_info.birth_date,
        },
        "frame": {
            "order_index": frame.order_index + 1,
            "image_id": frame.image_id,
            "series_key": frame.series_key,
            "patient_db_id": frame.patient_db_id,
            "study_id": frame.study_id,
            "series_id": frame.series_id,
            "slice_id": frame.slice_id,
            "series_frame_index": frame.series_frame_index + 1,
            "modality": frame.modality,
            "modality_name": frame.modality_name,
            "laterality": frame.laterality,
            "scan_pattern": frame.scan_pattern,
            "examined_structure": frame.examined_structure,
            "width": frame.width,
            "height": frame.height,
            "time_of_day_ms": frame.time_of_day_ms,
            "raw_elapsed_ms": frame.raw_elapsed_ms,
            "raw_elapsed_ms_alt": frame.raw_elapsed_ms_alt,
            "acquisition_datetime_iso": frame.acquisition_datetime_iso,
            "acquisition_date_iso": frame.acquisition_date_iso,
            "acquisition_time_iso": frame.acquisition_time_iso,
            "acquisition_source": frame.acquisition_source,
            "timezones": [value for value in [frame.timezone1, frame.timezone2] if value],
        },
        "study": {
            "study_datetime_local": study_info.study_datetime_iso,
            "timezone_offset_hours": study_info.timezone_offset_hours,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def frame_sort_key(frame: HeidelbergFAFrame) -> tuple[int, Any, float, int, int]:
    slice_sort = frame.slice_id if frame.slice_id >= 0 else 10 ** 9 + frame.series_frame_index
    time_marker = (
        frame.acquisition_datetime_local
        if frame.acquisition_datetime_local is not None
        else datetime.max
    )
    raw_marker = (
        float(frame.time_of_day_ms)
        if frame.time_of_day_ms is not None
        else float("inf")
    )
    return (
        0 if frame.acquisition_datetime_local is not None else 1,
        time_marker,
        raw_marker,
        frame.series_id,
        slice_sort,
    )


def apply_chunk39_timing_to_frame(
    frame: HeidelbergFAFrame,
    timing: Chunk39Timing,
    *,
    fallback_timezone1: str = "",
    fallback_timezone2: str = "",
) -> None:
    frame.timezone1 = timing.timezone1 or frame.timezone1 or fallback_timezone1
    frame.timezone2 = timing.timezone2 or frame.timezone2 or fallback_timezone2
    frame.time_of_day_ms = timing.time_of_day_ms
    frame.raw_elapsed_ms = timing.raw_elapsed_ms
    frame.raw_elapsed_ms_alt = timing.raw_elapsed_ms_alt
    frame.acquisition_source = timing.source


def load_heidelberg_fa_dataset(
    input_path: Optional[str],
) -> tuple[Optional[Path], HeidelbergFAStudyInfo, list[HeidelbergFAFrame]]:
    input_file = resolve_input_file(input_path)
    if input_file is None:
        return None, HeidelbergFAStudyInfo(), []
    if not input_file.exists() or not input_file.is_file():
        return input_file, HeidelbergFAStudyInfo(source_file=input_file), []

    reader = E2E(input_file)
    study_info = HeidelbergFAStudyInfo(source_file=input_file)
    series_metadata: dict[str, dict[str, Any]] = {}
    frames: list[HeidelbergFAFrame] = []
    series_frame_counts: Counter[str] = Counter()
    pending_timing_by_series_and_slice: defaultdict[str, defaultdict[int, deque[Chunk39Timing]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    unmatched_frame_indices_by_series_and_slice: defaultdict[str, defaultdict[int, deque[int]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    study_date_value: date | None = None

    with input_file.open("rb") as handle:
        chunk_stack: list[tuple[int, int]] = []
        for position in reader.directory_stack:
            handle.seek(position + reader.byte_skip)
            directory_chunk = e2e_binary.main_directory_structure.parse(handle.read(52))
            for _ in range(directory_chunk.num_entries):
                chunk = e2e_binary.sub_directory_structure.parse(handle.read(44))
                if chunk.start > chunk.pos:
                    chunk_stack.append((chunk.start, chunk.size))

        for start, _ in chunk_stack:
            handle.seek(start + reader.byte_skip)
            raw_header = handle.read(60)
            try:
                chunk = e2e_binary.chunk_structure.parse(raw_header)
                # print(chunk)
            except Exception:
                continue
            series_key = f"{chunk.patient_db_id}_{chunk.study_id}_{chunk.series_id}"
            series_entry = series_metadata.setdefault(
                series_key,
                {
                    "patient_db_id": chunk.patient_db_id,
                    "study_id": chunk.study_id,
                    "series_id": chunk.series_id,
                    "modality": "",
                    "modality_name": "",
                    "scan_pattern": "",
                    "examined_structure": "",
                    "laterality": "",
                    "timezone1": "",
                    "timezone2": "",
                },
            )

            if chunk.type == 58:
                payload = handle.read(chunk.size)
                parsed_study_datetime = parse_study_datetime_from_chunk58(payload)
                if parsed_study_datetime is not None:
                    study_info.study_datetime_local = parsed_study_datetime
                    study_date_value = parsed_study_datetime.date()
                continue

            if chunk.type == 9:
                try:
                    patient_data = e2e_binary.patient_id_structure.parse(handle.read(127))
                except Exception:
                    continue
                study_info.patient_first_name = clean_text(patient_data.first_name) or study_info.patient_first_name
                study_info.patient_surname = clean_text(patient_data.surname) or study_info.patient_surname
                study_info.patient_id = clean_text(patient_data.patient_id) or study_info.patient_id
                study_info.sex = normalize_sex(patient_data.sex) or study_info.sex
                parsed_birth_date = parse_birth_date(reader, patient_data.birthdate)
                if parsed_birth_date:
                    study_info.birth_date = parsed_birth_date
                continue

            if chunk.type == 9001:
                try:
                    device_data = e2e_binary.device_name.parse(handle.read(chunk.size))
                except Exception:
                    continue
                texts = [clean_text(text) for text in getattr(device_data, "text", []) if clean_text(text)]
                if texts:
                    study_info.device_model = texts[0] or study_info.device_model
                if len(texts) > 1:
                    study_info.device_short_name = texts[1] or study_info.device_short_name
                continue

            if chunk.type == 9005:
                try:
                    structure_data = e2e_binary.examined_structure.parse(handle.read(chunk.size))
                except Exception:
                    continue
                texts = [clean_text(text) for text in getattr(structure_data, "text", []) if clean_text(text)]
                if texts and not series_entry["examined_structure"]:
                    series_entry["examined_structure"] = texts[0]
                continue

            if chunk.type == 9006:
                try:
                    scan_pattern = e2e_binary.scan_pattern.parse(handle.read(chunk.size))
                except Exception:
                    continue
                texts = [clean_text(text) for text in getattr(scan_pattern, "text", []) if clean_text(text)]
                if texts and not series_entry["scan_pattern"]:
                    series_entry["scan_pattern"] = texts[0]
                continue

            if chunk.type == 9007:
                try:
                    modality_data = e2e_binary.enface_modality.parse(handle.read(chunk.size))
                except Exception:
                    continue
                texts = [clean_text(text) for text in getattr(modality_data, "text", []) if clean_text(text)]
                if texts:
                    if len(texts) > 1:
                        series_entry["modality_name"] = series_entry["modality_name"] or texts[0]
                        series_entry["modality"] = series_entry["modality"] or texts[1]
                    else:
                        series_entry["modality"] = series_entry["modality"] or texts[0]
                continue

            if chunk.type == 11:
                try:
                    laterality_data = e2e_binary.lat_structure.parse(handle.read(20))
                except Exception:
                    continue
                laterality_value = clean_text(laterality_data.laterality).upper()
                if laterality_value and not series_entry["laterality"]:
                    series_entry["laterality"] = laterality_value
                continue

            if chunk.type == 3:
                try:
                    pre_data = e2e_binary.pre_data.parse(handle.read(chunk.size))
                except Exception:
                    continue
                laterality_value = clean_text(pre_data.laterality).upper()
                if laterality_value in {"R", "L"} and not series_entry["laterality"]:
                    series_entry["laterality"] = laterality_value
                continue

            if chunk.type == 39:
                payload = handle.read(chunk.size)
                timing = parse_chunk39_timing(payload)
                timezone1 = timing.timezone1
                timezone2 = timing.timezone2
                if timezone1 and not study_info.timezone1:
                    study_info.timezone1 = timezone1
                if timezone2 and not study_info.timezone2:
                    study_info.timezone2 = timezone2
                if timezone1 and not series_entry["timezone1"]:
                    series_entry["timezone1"] = timezone1
                if timezone2 and not series_entry["timezone2"]:
                    series_entry["timezone2"] = timezone2
                pending_frames = unmatched_frame_indices_by_series_and_slice[series_key][chunk.slice_id]
                if pending_frames:
                    frame_index = pending_frames.popleft()
                    apply_chunk39_timing_to_frame(
                        frames[frame_index],
                        timing,
                        fallback_timezone1=series_entry["timezone1"] or study_info.timezone1,
                        fallback_timezone2=series_entry["timezone2"] or study_info.timezone2,
                    )
                else:
                    pending_timing_by_series_and_slice[series_key][chunk.slice_id].append(timing)
                continue

            if chunk.type == 1073741824 and chunk.ind == 0:
                try:
                    image_data = e2e_binary.image_structure.parse(handle.read(20))
                    # print("image_data: ", image_data)
                    # print("image_data.height: ", image_data.height)
                except Exception:
                    continue

                pixel_count = image_data.height * image_data.width
                if pixel_count <= 0:
                    continue

                raw_image = np.frombuffer(handle.read(pixel_count), dtype=np.uint8)
                if raw_image.size != pixel_count:
                    continue

                image = raw_image.reshape(image_data.height, image_data.width)
                series_frame_index = series_frame_counts[series_key]
                series_frame_counts[series_key] += 1
                timing_queue = pending_timing_by_series_and_slice[series_key][chunk.slice_id]
                timing = timing_queue.popleft() if timing_queue else None

                image_id = f"{series_key}:f{series_frame_index + 1:03d}:s{chunk.slice_id}"
                frame = HeidelbergFAFrame(
                    order_index=len(frames),
                    image_id=image_id,
                    series_key=series_key,
                    patient_db_id=chunk.patient_db_id,
                    study_id=chunk.study_id,
                    series_id=chunk.series_id,
                    slice_id=chunk.slice_id,
                    series_frame_index=series_frame_index,
                    modality="",
                    modality_name="",
                    scan_pattern="",
                    examined_structure="",
                    laterality="",
                    timezone1=series_entry["timezone1"] or study_info.timezone1,
                    timezone2=series_entry["timezone2"] or study_info.timezone2,
                    time_of_day_ms=None,
                    raw_elapsed_ms=None,
                    raw_elapsed_ms_alt=None,
                    acquisition_datetime_local=None,
                    acquisition_source="",
                    width=int(image_data.width),
                    height=int(image_data.height),
                    image=image,
                )
                if timing is not None:
                    apply_chunk39_timing_to_frame(
                        frame,
                        timing,
                        fallback_timezone1=series_entry["timezone1"] or study_info.timezone1,
                        fallback_timezone2=series_entry["timezone2"] or study_info.timezone2,
                    )
                frames.append(frame)
                if timing is None:
                    unmatched_frame_indices_by_series_and_slice[series_key][chunk.slice_id].append(len(frames) - 1)

    if study_date_value is None and study_info.study_datetime_local is not None:
        study_date_value = study_info.study_datetime_local.date()
    study_info.timezone_offset_hours = parse_timezone_offset_hours(study_info.timezone1, study_info.timezone2)

    for frame in frames:
        meta = series_metadata.get(frame.series_key, {})
        frame.modality = clean_text(meta.get("modality", "")) or "Unknown"
        frame.modality_name = clean_text(meta.get("modality_name", ""))
        frame.scan_pattern = clean_text(meta.get("scan_pattern", ""))
        frame.examined_structure = clean_text(meta.get("examined_structure", ""))
        frame.laterality = clean_text(meta.get("laterality", "")).upper()
        frame.timezone1 = frame.timezone1 or clean_text(meta.get("timezone1", "")) or study_info.timezone1
        frame.timezone2 = frame.timezone2 or clean_text(meta.get("timezone2", "")) or study_info.timezone2
        frame.acquisition_datetime_local, inferred_source = choose_frame_datetime(
            study_datetime_local=study_info.study_datetime_local,
            study_date_value=study_date_value,
            time_of_day_ms=frame.time_of_day_ms,
            timezone_offset_hours=study_info.timezone_offset_hours,
        )
        if inferred_source:
            frame.acquisition_source = inferred_source

    frames.sort(key=frame_sort_key)
    series_display_counts: Counter[str] = Counter()
    for index, frame in enumerate(frames):
        frame.order_index = index
        frame.series_frame_index = series_display_counts[frame.series_key]
        series_display_counts[frame.series_key] += 1
        frame.image_id = f"{frame.series_key}:f{frame.series_frame_index + 1:03d}:s{frame.slice_id}"
        frame.metadata_text = build_frame_metadata_text(input_file, study_info, frame)

    study_info.frame_count = len(frames)
    study_info.series_count = len({frame.series_key for frame in frames})
    study_info.modality_summary = modality_summary(frames)
    study_info.laterality_summary = laterality_summary(frames)

    return input_file, study_info, frames


def dump_study_info(study_info: HeidelbergFAStudyInfo) -> None:
    print("source_file\t", str(study_info.source_file) if study_info.source_file else "-")
    print("patient_name\t", study_info.patient_name or "-")
    print("patient_id\t", study_info.patient_id or "-")
    print("sex\t", study_info.sex_display)
    print("birth_date\t", study_info.birth_date or "-")
    print("device\t", study_info.device_display)
    print("timezone\t", study_info.timezone_display)
    print("study_datetime\t", study_info.study_datetime_iso)
    print("series_count\t", study_info.series_count)
    print("frame_count\t", study_info.frame_count)
    print("modalities\t", study_info.modality_summary or "-")
    print("laterality\t", study_info.laterality_summary or "-")
    print()


def _display_width(text: object) -> int:
    width = 0
    for character in str(text):
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def _pad_display_text(text: object, width: int) -> str:
    rendered = str(text)
    padding = max(0, width - _display_width(rendered))
    return rendered + (" " * padding)


def dump_frames(frames: list[HeidelbergFAFrame]) -> None:
    headers = ["index", "series", "modality", "eye", "series_index", "slice", "size", "acquisition"]
    rows = [
        [
            str(frame.order_index + 1),
            str(frame.series_id),
            frame.modality or "-",
            frame.laterality_display,
            str(frame.series_frame_index + 1),
            str(frame.slice_id),
            frame.size_display,
            frame.time_display,
        ]
        for frame in frames
    ]

    widths = [
        max(_display_width(header), *(_display_width(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ] if rows else [_display_width(header) for header in headers]

    print("  ".join(_pad_display_text(header, widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        print("  ".join(_pad_display_text(value, widths[index]) for index, value in enumerate(row)))


@dataclass
class HeidelbergViewerTrack:
    key: str
    label: str
    modality: str
    laterality: str
    display_laterality: str
    frame_count: int
    series_count: int
    first_datetime_iso: str | None
    last_datetime_iso: str | None
    frames: list[HeidelbergFAFrame]


def viewer_laterality_label(value: str) -> str:
    normalized = clean_text(value).upper()
    if normalized == "ALL":
        return "双眼"
    if not normalized:
        return "-"
    return laterality_to_chinese(normalized)


def viewer_modality_label(value: str) -> str:
    normalized = clean_text(value).upper()
    return normalized or "全部模式"


def viewer_laterality_short_label(value: str) -> str:
    normalized = clean_text(value).upper()
    return {
        "R": "OD",
        "OD": "OD",
        "L": "OS",
        "OS": "OS",
        "ALL": "OU",
        "UNKNOWN": "UNK",
    }.get(normalized, normalized or "-")


def infer_group_laterality(
    frames: list[HeidelbergFAFrame],
    fallback: str = "ALL",
) -> str:
    normalized = {
        clean_text(frame.laterality).upper() or "UNKNOWN"
        for frame in frames
    }
    known = {value for value in normalized if value in {"R", "L", "OD", "OS"}}
    if len(known) == 1:
        return next(iter(known))
    if len(known) > 1:
        return "ALL"
    if normalized == {"UNKNOWN"}:
        return "UNKNOWN"
    return fallback


def build_track_label(
    modality: str,
    laterality: str,
    frames: list[HeidelbergFAFrame],
) -> str:
    modality_text = "全部模式" if modality == "ALL" else viewer_modality_label(modality)
    display_laterality = infer_group_laterality(frames, laterality)
    laterality_text = viewer_laterality_label(display_laterality)
    first_iso = next((frame.acquisition_datetime_iso for frame in frames if frame.acquisition_datetime_iso), None)
    last_iso = next((frame.acquisition_datetime_iso for frame in reversed(frames) if frame.acquisition_datetime_iso),
                    None)
    time_suffix = ""
    if first_iso and last_iso and first_iso != last_iso:
        time_suffix = f" | {first_iso.split('T')[-1]} → {last_iso.split('T')[-1]}"
    elif first_iso:
        time_suffix = f" | {first_iso.split('T')[-1]}"
    return (
        f"{modality_text} | {laterality_text} | "
        f"{len(frames)} 帧 | {len({frame.series_key for frame in frames})} 序列{time_suffix}"
    )


def build_heidelberg_viewer_tracks(
    frames: list[HeidelbergFAFrame],
) -> list[HeidelbergViewerTrack]:
    grouped: dict[tuple[str, str], list[HeidelbergFAFrame]] = {}

    if frames:
        grouped[("ALL", "ALL")] = list(frames)

    modalities = sorted({frame.modality for frame in frames}, key=modality_sort_key)
    lateralities = sorted({frame.laterality or "UNKNOWN" for frame in frames})

    for modality in modalities:
        modality_frames = [frame for frame in frames if frame.modality == modality]
        if modality_frames:
            grouped[(modality, "ALL")] = modality_frames
        for laterality in lateralities:
            filtered = [
                frame
                for frame in modality_frames
                if (frame.laterality or "UNKNOWN") == laterality
            ]
            if filtered:
                grouped[(modality, laterality)] = filtered

    def sort_key(item: tuple[tuple[str, str], list[HeidelbergFAFrame]]) -> tuple[int, int, str, str]:
        (modality, laterality), _frames = item
        if modality == "ALL" and laterality == "ALL":
            return (0, 0, "", "")
        if laterality == "ALL":
            return (1, modality_sort_key(modality)[0], modality, laterality)
        return (2, modality_sort_key(modality)[0], modality, laterality)

    tracks: list[HeidelbergViewerTrack] = []
    for (modality, laterality), grouped_frames in sorted(grouped.items(), key=sort_key):
        display_laterality = infer_group_laterality(grouped_frames, laterality)
        label = build_track_label(modality, laterality, grouped_frames)
        tracks.append(
            HeidelbergViewerTrack(
                key=f"{modality}:{laterality}",
                label=label,
                modality=modality,
                laterality=laterality,
                display_laterality=display_laterality,
                frame_count=len(grouped_frames),
                series_count=len({frame.series_key for frame in grouped_frames}),
                first_datetime_iso=next(
                    (frame.acquisition_datetime_iso for frame in grouped_frames if frame.acquisition_datetime_iso),
                    None,
                ),
                last_datetime_iso=next(
                    (
                        frame.acquisition_datetime_iso
                        for frame in reversed(grouped_frames)
                        if frame.acquisition_datetime_iso
                    ),
                    None,
                ),
                frames=grouped_frames,
            )
        )
    return tracks


def summarize_tracks_for_console(tracks: list[HeidelbergViewerTrack]) -> list[str]:
    lines: list[str] = []
    for track in tracks:
        lines.append(track.label)
    return lines


def frame_time_text(frame: HeidelbergFAFrame) -> str:
    return frame.time_display


if __name__ == "__main__":
    filepath = r"E:\Data\OCT2\海德堡\KH902-R10-007-007003DME-V1-FFA\KH902-R10-007-007003DME-V1-FFA-OD.E2E"

    input_file, study_info, frames = load_heidelberg_fa_dataset(filepath)
    # dump_study_info(study_info)
    # dump_frames(frames)
