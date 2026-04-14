from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import unicodedata

from PIL import Image

DEFAULT_INPUT_DIR = Path(r"E:\Data\OCT\topcon-FA")
IMAGE_NAME_PATTERN = re.compile(rb"IM\d{6}\.JPG", re.IGNORECASE)
ELAPSED_LABEL_PATTERN = re.compile(r"^(?P<minutes>\d+):(?P<seconds>\d{1,2}(?:\.\d+)?)$")
LATERALITY_PATTERN = re.compile(r"(?:^|[-_/ ])(?P<eye>OD|OS|R|L)(?=\d|\b)", re.IGNORECASE)


@dataclass
class TopconFAFrame:
    order_index: int
    image_path: Path
    filename: str
    modality: str
    label: str
    device: str
    acquisition_date: str
    acquisition_time: str
    acquisition_datetime: datetime | None
    elapsed_seconds: float | None
    width: int | None
    height: int | None

    @property
    def is_proofsheet(self) -> bool:
        return self.label.lower().startswith("proofsheet")

    @property
    def acquisition_display(self) -> str:
        if self.acquisition_date and self.acquisition_time:
            return f"{self.acquisition_date} {self.acquisition_time}"
        if self.acquisition_date:
            return self.acquisition_date
        if self.acquisition_time:
            return self.acquisition_time
        return "-"

    @property
    def elapsed_display(self) -> str:
        if self.elapsed_seconds is None:
            return "-"
        return f"{self.elapsed_seconds:.1f} s"

    @property
    def size_display(self) -> str:
        if self.width and self.height:
            return f"{self.width} 脳 {self.height}"
        return "-"


@dataclass
class TopconFAStudyInfo:
    patient_name: str = ""
    patient_id: str = ""
    sex: str = ""
    birth_date: str = ""
    exam_date: str = ""
    register_date: str = ""
    laterality: str = ""
    laterality_note: str = ""
    study_code: str = ""
    device_model: str = ""
    extra_value: str = ""

    @property
    def laterality_display(self) -> str:
        if self.laterality and self.laterality_note:
            return f"{self.laterality}"
        return self.laterality or "-"

    @property
    def sex_display(self) -> str:
        return self.sex or "-"

    @property
    def exam_date_display(self) -> str:
        if self.exam_date and self.register_date and self.register_date != self.exam_date:
            return f"{self.exam_date} / {self.register_date}"
        return self.exam_date or self.register_date or "-"


def decode_text(raw: bytes) -> str:
    chunk = raw.split(b"\x00", 1)[0].strip()
    if not chunk:
        return ""
    for encoding in ("utf-8", "gb18030", "latin1"):
        try:
            return chunk.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return chunk.decode("latin1", errors="replace").strip()


def parse_elapsed_seconds(label: str) -> float | None:
    match = ELAPSED_LABEL_PATTERN.match(label.strip())
    if not match:
        return None
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    return minutes * 60.0 + seconds


def parse_datetime(date_text: str, time_text: str) -> datetime | None:
    if not date_text or not time_text:
        return None
    try:
        return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def resolve_input_dir(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        return path.parent
    return path


def load_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def normalize_sex(value: str) -> str:
    value = value.strip().upper()
    return {
        "M": "鐢?",
        "F": "濂?",
        "O": "鍏朵粬",
    }.get(value, value)


def infer_laterality(*candidates: str) -> tuple[str, str]:
    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        match = LATERALITY_PATTERN.search(text)
        if not match:
            continue
        eye = match.group("eye").upper()
        if eye in {"OD", "R"}:
            return "鍙崇溂", f"鐢?{text} 鎺ㄦ柇"
        if eye in {"OS", "L"}:
            return "宸︾溂", f"鐢?{text} 鎺ㄦ柇"
    return "", ""


def parse_study_info(input_dir: Path) -> TopconFAStudyInfo:
    datafile = input_dir / "DATAFILE"
    if not datafile.exists():
        return TopconFAStudyInfo()

    raw = datafile.read_bytes()
    header = raw[:176].ljust(176, b"\x00")
    study_code = decode_text(header[24:48])
    device_model = decode_text(header[4:24])
    patient_name = decode_text(header[96:132])
    patient_id = decode_text(header[144:160])
    sex = normalize_sex(decode_text(header[76:80]))
    birth_date = decode_text(header[80:96])
    exam_date = decode_text(header[132:144])
    register_date = decode_text(header[160:176])
    extra_value = decode_text(header[48:76])
    laterality, laterality_note = infer_laterality(study_code, patient_name, input_dir.name)

    return TopconFAStudyInfo(
        patient_name=patient_name,
        patient_id=patient_id,
        sex=sex,
        birth_date=birth_date,
        exam_date=exam_date,
        register_date=register_date,
        laterality=laterality,
        laterality_note=laterality_note,
        study_code=study_code,
        device_model=device_model,
        extra_value=extra_value,
    )


def parse_datafile_records(input_dir: Path) -> list[TopconFAFrame]:
    datafile = input_dir / "DATAFILE"
    if not datafile.exists():
        return []

    image_paths = {
        image.name.upper(): image
        for image in sorted(input_dir.glob("IM*.JPG"))
        if image.is_file()
    }
    raw = datafile.read_bytes()
    frames: list[TopconFAFrame] = []

    for order_index, match in enumerate(IMAGE_NAME_PATTERN.finditer(raw)):
        record_start = match.start() - 96
        if record_start < 0:
            continue
        record = raw[record_start : record_start + 200]
        if len(record) < 200:
            record = record.ljust(200, b"\x00")

        filename = decode_text(record[96:112]).upper()
        image_path = image_paths.get(filename)
        if image_path is None:
            continue

        modality = decode_text(record[4:24]) or "Unknown"
        label = decode_text(record[24:96])
        device = decode_text(record[160:180])
        acquisition_date = decode_text(record[180:192])
        acquisition_time = decode_text(record[192:200])
        acquisition_datetime = parse_datetime(acquisition_date, acquisition_time)
        elapsed_seconds = parse_elapsed_seconds(label)
        width = int.from_bytes(record[124:126], byteorder="little", signed=False)
        height = int.from_bytes(record[126:128], byteorder="little", signed=False)
        if width <= 0 or height <= 0:
            width = None
            height = None

        frames.append(
            TopconFAFrame(
                order_index=order_index,
                image_path=image_path,
                filename=image_path.name,
                modality=modality,
                label=label,
                device=device,
                acquisition_date=acquisition_date,
                acquisition_time=acquisition_time,
                acquisition_datetime=acquisition_datetime,
                elapsed_seconds=elapsed_seconds,
                width=width,
                height=height,
            )
        )

    previous_by_modality: dict[str, TopconFAFrame] = {}
    for frame in frames:
        previous = previous_by_modality.get(frame.modality)
        if frame.acquisition_datetime is None and previous is not None:
            frame.acquisition_date = previous.acquisition_date
            frame.acquisition_time = previous.acquisition_time
            frame.acquisition_datetime = previous.acquisition_datetime
        if (not frame.device or frame.device.startswith("*IM")) and previous is not None:
            frame.device = previous.device
        if frame.width is None or frame.height is None:
            frame.width, frame.height = load_image_size(frame.image_path)
        previous_by_modality[frame.modality] = frame

    return frames


def fallback_records_from_images(input_dir: Path) -> list[TopconFAFrame]:
    image_paths = sorted(input_dir.glob("IM*.JPG"))
    frames: list[TopconFAFrame] = []
    for order_index, image_path in enumerate(image_paths):
        width, height = load_image_size(image_path)
        modified = datetime.fromtimestamp(image_path.stat().st_mtime)
        frames.append(
            TopconFAFrame(
                order_index=order_index,
                image_path=image_path,
                filename=image_path.name,
                modality="Unknown",
                label="",
                device="",
                acquisition_date=modified.strftime("%Y-%m-%d"),
                acquisition_time=modified.strftime("%H:%M:%S"),
                acquisition_datetime=modified,
                elapsed_seconds=None,
                width=width,
                height=height,
            )
        )
    return frames


def load_topcon_fa_dataset(
    input_path: str | None,
) -> tuple[Path | None, TopconFAStudyInfo, list[TopconFAFrame]]:
    input_dir = resolve_input_dir(input_path)
    if input_dir is None or not input_dir.exists() or not input_dir.is_dir():
        return input_dir, TopconFAStudyInfo(), []

    study_info = parse_study_info(input_dir)
    frames = parse_datafile_records(input_dir)
    if not frames:
        frames = fallback_records_from_images(input_dir)
    return input_dir, study_info, frames


def load_topcon_fa_frames(input_path: str | None) -> tuple[Path | None, list[TopconFAFrame]]:
    input_dir, _, frames = load_topcon_fa_dataset(input_path)
    return input_dir, frames


def modality_summary(frames: list[TopconFAFrame]) -> str:
    counts = Counter(frame.modality for frame in frames)
    return ", ".join(f"{modality} {count}" for modality, count in counts.items()) or "-"


def _display_width(text: object) -> int:
    width = 0
    for character in str(text):
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def _pad_display_text(text: object, width: int) -> str:
    rendered = str(text)
    padding = max(0, width - _display_width(rendered))
    return rendered + (" " * padding)


def dump_study_info(study_info: TopconFAStudyInfo) -> None:
    rows = [
        ("patient_name", study_info.patient_name or "-"),
        ("patient_id", study_info.patient_id or "-"),
        ("sex", study_info.sex_display),
        ("birth_date", study_info.birth_date or "-"),
        ("exam_date", study_info.exam_date_display),
        ("laterality", study_info.laterality_display),
        ("study_code", study_info.study_code or "-"),
        ("device_model", study_info.device_model or "-"),
        ("extra_value", study_info.extra_value or "-"),
    ]
    key_width = max(_display_width(key) for key, _ in rows)
    for key, value in rows:
        print(f"{_pad_display_text(key, key_width)} : {value}")
    print()


def dump_frames(frames: list[TopconFAFrame]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    headers = ["index", "filename", "modality", "label", "elapsed_s", "acquisition", "size", "device"]
    rows = []
    for frame in frames:
        rows.append(
            [
                str(frame.order_index + 1),
                frame.filename,
                frame.modality,
                frame.label or "-",
                "" if frame.elapsed_seconds is None else f"{frame.elapsed_seconds:.1f}",
                frame.acquisition_display,
                frame.size_display,
                frame.device or "-",
            ]
        )

    widths = [
        max(_display_width(header), *(_display_width(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ] if rows else [_display_width(header) for header in headers]

    print("  ".join(_pad_display_text(header, widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        print("  ".join(_pad_display_text(value, widths[index]) for index, value in enumerate(row)))


__all__ = [
    "DEFAULT_INPUT_DIR",
    "TopconFAFrame",
    "TopconFAStudyInfo",
    "decode_text",
    "dump_frames",
    "dump_study_info",
    "fallback_records_from_images",
    "infer_laterality",
    "load_topcon_fa_dataset",
    "load_topcon_fa_frames",
    "modality_summary",
    "normalize_sex",
    "parse_datafile_records",
    "parse_datetime",
    "parse_elapsed_seconds",
    "parse_study_info",
    "resolve_input_dir",
]


if __name__ == "__main__":
    example_path = DEFAULT_INPUT_DIR
    print(f"Testing Topcon FA parser with: {example_path}")
    input_dir, study_info, frames = load_topcon_fa_dataset(str(example_path))

    if input_dir is None or not input_dir.exists():
        print("Example path not found.")
    elif not frames:
        print(f"No Topcon FA data parsed from: {input_dir}")
    else:
        print(f"Resolved input directory: {input_dir}")
        print(f"Frame count: {len(frames)}")
        print(f"Modalities: {modality_summary(frames)}")
        print()
        dump_study_info(study_info)
        dump_frames(frames)
