from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oct_converter.readers import E2E
from oct_converter.readers.binary_structs import e2e_binary

try:
    from PyQt5.QtCore import QSettings, Qt, QTimer
    from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QShortcut,
        QSlider,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


DEFAULT_INPUT_PATH = Path(r"E:\Data\OCT\海德堡\海德堡FA.E2E")
TIME_PARSE_NOTE = (
    "当前仓库能稳定解析海德堡 E2E 的图像、眼别、模式、系列和时区信息；"
    "chunk 39 中的绝对采集时间尚未被这个仓库解码，因此界面当前按 series_id + slice_id 排序显示帧。"
)


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
    frame_count: int = 0
    series_count: int = 0
    modality_summary: str = ""
    laterality_summary: str = ""
    timing_note: str = TIME_PARSE_NOTE

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


def image_to_qpixmap(image: np.ndarray) -> "QPixmap":
    display = np.ascontiguousarray(normalize_to_uint8(image))
    if display.ndim == 2:
        qimage = QImage(
            display.data,
            display.shape[1],
            display.shape[0],
            display.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
        return QPixmap.fromImage(qimage)

    if display.ndim == 3 and display.shape[2] >= 3:
        rgb = np.ascontiguousarray(display[:, :, :3])
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimage)

    raise ValueError(f"不支持的图像维度：{display.shape}")


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
            "timezones": [value for value in [frame.timezone1, frame.timezone2] if value],
        },
        "note": TIME_PARSE_NOTE,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def frame_sort_key(frame: HeidelbergFAFrame) -> tuple[int, int, int, int]:
    slice_sort = frame.slice_id if frame.slice_id >= 0 else 10**9 + frame.series_frame_index
    return frame.series_id, slice_sort, frame.order_index, frame.series_frame_index


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
                try:
                    time_data = e2e_binary.time_data.parse(handle.read(chunk.size))
                except Exception:
                    continue
                timezone1 = clean_text(getattr(time_data, "timezone1", ""))
                timezone2 = clean_text(getattr(time_data, "timezone2", ""))
                if timezone1 and not study_info.timezone1:
                    study_info.timezone1 = timezone1
                if timezone2 and not study_info.timezone2:
                    study_info.timezone2 = timezone2
                if timezone1 and not series_entry["timezone1"]:
                    series_entry["timezone1"] = timezone1
                if timezone2 and not series_entry["timezone2"]:
                    series_entry["timezone2"] = timezone2
                continue

            if chunk.type == 1073741824 and chunk.ind == 0:
                try:
                    image_data = e2e_binary.image_structure.parse(handle.read(20))
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

                image_id = f"{series_key}:f{series_frame_index + 1:03d}:s{chunk.slice_id}"
                frames.append(
                    HeidelbergFAFrame(
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
                        width=int(image_data.width),
                        height=int(image_data.height),
                        image=image,
                    )
                )

    for frame in frames:
        meta = series_metadata.get(frame.series_key, {})
        frame.modality = clean_text(meta.get("modality", "")) or "Unknown"
        frame.modality_name = clean_text(meta.get("modality_name", ""))
        frame.scan_pattern = clean_text(meta.get("scan_pattern", ""))
        frame.examined_structure = clean_text(meta.get("examined_structure", ""))
        frame.laterality = clean_text(meta.get("laterality", "")).upper()
        frame.timezone1 = frame.timezone1 or clean_text(meta.get("timezone1", "")) or study_info.timezone1
        frame.timezone2 = frame.timezone2 or clean_text(meta.get("timezone2", "")) or study_info.timezone2

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
    print("series_count\t", study_info.series_count)
    print("frame_count\t", study_info.frame_count)
    print("modalities\t", study_info.modality_summary or "-")
    print("laterality\t", study_info.laterality_summary or "-")
    print("timing_note\t", study_info.timing_note)
    print()


def dump_frames(frames: list[HeidelbergFAFrame]) -> None:
    print("index\tseries\tmodality\teye\tseries_index\tslice\tsize")
    for frame in frames:
        print(
            f"{frame.order_index + 1}\t{frame.series_id}\t{frame.modality}\t"
            f"{frame.laterality_display}\t{frame.series_frame_index + 1}\t"
            f"{frame.slice_id}\t{frame.size_display}"
        )


if QT_AVAILABLE:

    class ScaledPixmapLabel(QLabel):
        def __init__(self) -> None:
            super().__init__("请打开一个海德堡 E2E 文件")
            self._pixmap = QPixmap()
            self.setAlignment(Qt.AlignCenter)
            self.setMinimumSize(680, 520)
            self.setWordWrap(True)
            self.setStyleSheet(
                """
                QLabel {
                    background: #0B1220;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    color: #CBD5E1;
                    padding: 8px;
                }
                """
            )

        def set_pixmap(self, pixmap: QPixmap, text: str = "") -> None:
            self._pixmap = pixmap
            self.setText(text if pixmap.isNull() else "")
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


    class HeidelbergFAViewerWindow(QMainWindow):
        def __init__(self, input_path: Optional[str]) -> None:
            super().__init__()
            self.settings = QSettings("OpenAI", "HeidelbergFAQtViewer")
            self.input_file: Optional[Path] = None
            self.study_info = HeidelbergFAStudyInfo()
            self.frames: list[HeidelbergFAFrame] = []
            self.visible_frames: list[HeidelbergFAFrame] = []
            self.current_visible_index = -1
            self.play_timer = QTimer(self)
            self.play_timer.timeout.connect(self.advance_frame)

            self.setWindowTitle("Heidelberg FA E2E 查看器")
            self.resize(1680, 980)
            self.setFont(QFont("Microsoft YaHei UI", 10))
            self._apply_styles()

            self.image_label = ScaledPixmapLabel()
            self.setStatusBar(QStatusBar(self))
            self._build_controls()
            self._build_layout()
            self._install_shortcuts()

            if input_path:
                self.load_file(input_path)
            else:
                self._set_empty_state("未指定 E2E 文件，请点击“打开文件”。")

        def _apply_styles(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background: #111827; color: #E5E7EB; }
                QGroupBox {
                    border: 1px solid #334155;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 10px;
                    font-weight: 600;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
                QPushButton, QComboBox, QSpinBox, QLineEdit {
                    background: #0F172A;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    min-height: 30px;
                    padding: 4px 8px;
                }
                QPushButton:hover { border-color: #60A5FA; }
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
                QTableWidget, QPlainTextEdit {
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
                """
            )

        def _build_controls(self) -> None:
            self.open_button = QPushButton("打开文件")
            self.open_button.clicked.connect(self.choose_file)

            self.reload_button = QPushButton("重新载入")
            self.reload_button.clicked.connect(self.reload_current_file)

            self.save_current_button = QPushButton("保存当前帧")
            self.save_current_button.clicked.connect(self.save_current_frame)

            self.export_visible_button = QPushButton("导出筛选帧")
            self.export_visible_button.clicked.connect(self.export_visible_frames)

            self.path_edit = QLineEdit()
            self.path_edit.setReadOnly(True)

            self.modality_combo = QComboBox()
            self.modality_combo.currentIndexChanged.connect(self.refresh_visible_frames)

            self.eye_combo = QComboBox()
            self.eye_combo.currentIndexChanged.connect(self.refresh_visible_frames)

            self.series_combo = QComboBox()
            self.series_combo.currentIndexChanged.connect(self.refresh_visible_frames)

            self.fa_only_checkbox = QCheckBox("默认聚焦 FA")
            self.fa_only_checkbox.setChecked(True)
            self.fa_only_checkbox.toggled.connect(self._apply_default_modality_if_needed)

            self.play_button = QPushButton("播放")
            self.play_button.clicked.connect(self.toggle_playback)

            self.fps_spinbox = QSpinBox()
            self.fps_spinbox.setRange(1, 30)
            self.fps_spinbox.setValue(3)
            self.fps_spinbox.setSuffix(" fps")
            self.fps_spinbox.valueChanged.connect(self._update_play_interval)

            self.contrast_slider = QSlider(Qt.Horizontal)
            self.contrast_slider.setRange(25, 300)
            self.contrast_slider.setValue(100)
            self.contrast_slider.valueChanged.connect(self._redraw_current_frame)

            self.brightness_slider = QSlider(Qt.Horizontal)
            self.brightness_slider.setRange(-128, 128)
            self.brightness_slider.setValue(0)
            self.brightness_slider.valueChanged.connect(self._redraw_current_frame)

            self.timeline_slider = QSlider(Qt.Horizontal)
            self.timeline_slider.setMinimum(0)
            self.timeline_slider.setMaximum(0)
            self.timeline_slider.valueChanged.connect(self.set_current_visible_index)

            self.table = QTableWidget(0, 8)
            self.table.setHorizontalHeaderLabels(
                ["#", "系列", "模式", "眼别", "序列帧", "Slice", "尺寸", "结构"]
            )
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.setAlternatingRowColors(True)
            self.table.verticalHeader().setVisible(False)
            self.table.itemSelectionChanged.connect(self._sync_selection_from_table)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)

            self.dataset_file_label = QLabel("-")
            self.dataset_frames_label = QLabel("-")
            self.dataset_series_label = QLabel("-")
            self.dataset_modalities_label = QLabel("-")
            self.dataset_eyes_label = QLabel("-")
            self.dataset_timezone_label = QLabel("-")
            self.dataset_note_label = QLabel("-")
            self.dataset_note_label.setWordWrap(True)

            self.patient_name_label = QLabel("-")
            self.patient_id_label = QLabel("-")
            self.patient_sex_label = QLabel("-")
            self.patient_birth_label = QLabel("-")
            self.patient_device_label = QLabel("-")

            self.frame_position_label = QLabel("-")
            self.frame_image_id_label = QLabel("-")
            self.frame_series_label = QLabel("-")
            self.frame_modality_label = QLabel("-")
            self.frame_eye_label = QLabel("-")
            self.frame_sequence_label = QLabel("-")
            self.frame_size_label = QLabel("-")
            self.frame_structure_label = QLabel("-")
            self.frame_time_label = QLabel("-")

            self.metadata_edit = QPlainTextEdit()
            self.metadata_edit.setReadOnly(True)
            self.metadata_edit.setPlaceholderText("选中一帧后，这里会显示解析元数据。")

            self._update_play_interval()

        def _build_layout(self) -> None:
            top_bar = QWidget()
            top_layout = QHBoxLayout(top_bar)
            top_layout.setContentsMargins(0, 0, 0, 0)
            top_layout.addWidget(self.open_button)
            top_layout.addWidget(self.reload_button)
            top_layout.addWidget(self.save_current_button)
            top_layout.addWidget(self.export_visible_button)
            top_layout.addWidget(self.path_edit, 1)
            top_layout.addWidget(QLabel("模式"))
            top_layout.addWidget(self.modality_combo)
            top_layout.addWidget(QLabel("眼别"))
            top_layout.addWidget(self.eye_combo)
            top_layout.addWidget(QLabel("系列"))
            top_layout.addWidget(self.series_combo)
            top_layout.addWidget(self.fa_only_checkbox)
            top_layout.addWidget(self.play_button)
            top_layout.addWidget(self.fps_spinbox)

            dataset_box = QGroupBox("数据集")
            dataset_form = QFormLayout(dataset_box)
            dataset_form.addRow("文件", self.dataset_file_label)
            dataset_form.addRow("帧数", self.dataset_frames_label)
            dataset_form.addRow("系列数", self.dataset_series_label)
            dataset_form.addRow("模式", self.dataset_modalities_label)
            dataset_form.addRow("眼别", self.dataset_eyes_label)
            dataset_form.addRow("时区", self.dataset_timezone_label)
            dataset_form.addRow("说明", self.dataset_note_label)

            patient_box = QGroupBox("患者信息")
            patient_form = QFormLayout(patient_box)
            patient_form.addRow("姓名", self.patient_name_label)
            patient_form.addRow("患者 ID", self.patient_id_label)
            patient_form.addRow("性别", self.patient_sex_label)
            patient_form.addRow("出生日期", self.patient_birth_label)
            patient_form.addRow("设备", self.patient_device_label)

            frame_box = QGroupBox("当前帧")
            frame_form = QFormLayout(frame_box)
            frame_form.addRow("位置", self.frame_position_label)
            frame_form.addRow("图像 ID", self.frame_image_id_label)
            frame_form.addRow("系列", self.frame_series_label)
            frame_form.addRow("模式", self.frame_modality_label)
            frame_form.addRow("眼别", self.frame_eye_label)
            frame_form.addRow("序列位置", self.frame_sequence_label)
            frame_form.addRow("尺寸", self.frame_size_label)
            frame_form.addRow("结构", self.frame_structure_label)
            frame_form.addRow("时间", self.frame_time_label)

            display_box = QGroupBox("显示调节")
            display_form = QFormLayout(display_box)
            display_form.addRow("对比度", self.contrast_slider)
            display_form.addRow("亮度", self.brightness_slider)

            metadata_box = QGroupBox("元数据")
            metadata_layout = QVBoxLayout(metadata_box)
            metadata_layout.addWidget(self.metadata_edit)

            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.addWidget(self.image_label, 1)

            right_top = QWidget()
            right_top_layout = QVBoxLayout(right_top)
            right_top_layout.setContentsMargins(0, 0, 0, 0)
            right_top_layout.addWidget(dataset_box)
            right_top_layout.addWidget(patient_box)
            right_top_layout.addWidget(frame_box)
            right_top_layout.addWidget(display_box)

            right_bottom = QWidget()
            right_bottom_layout = QVBoxLayout(right_bottom)
            right_bottom_layout.setContentsMargins(0, 0, 0, 0)
            right_bottom_layout.addWidget(self.table, 2)
            right_bottom_layout.addWidget(metadata_box, 1)

            right_splitter = QSplitter(Qt.Vertical)
            right_splitter.addWidget(right_top)
            right_splitter.addWidget(right_bottom)
            right_splitter.setChildrenCollapsible(False)
            right_splitter.setStretchFactor(0, 0)
            right_splitter.setStretchFactor(1, 1)

            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(left_panel)
            splitter.addWidget(right_splitter)
            splitter.setChildrenCollapsible(False)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)

            central = QWidget()
            main_layout = QVBoxLayout(central)
            main_layout.addWidget(top_bar)
            main_layout.addWidget(splitter, 1)
            main_layout.addWidget(self.timeline_slider)
            self.setCentralWidget(central)

        def _install_shortcuts(self) -> None:
            QShortcut(QKeySequence("Right"), self, activated=self.advance_frame)
            QShortcut(QKeySequence("Left"), self, activated=self.retreat_frame)
            QShortcut(QKeySequence("Space"), self, activated=self.toggle_playback)
            QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_file)
            QShortcut(QKeySequence("Ctrl+R"), self, activated=self.reload_current_file)
            QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current_frame)

        def _update_play_interval(self) -> None:
            fps = max(1, self.fps_spinbox.value())
            self.play_timer.setInterval(int(1000 / fps))

        def choose_file(self) -> None:
            start_dir = (
                str(self.input_file.parent)
                if self.input_file
                else self.settings.value(
                    "last_dir",
                    str(DEFAULT_INPUT_PATH.parent if DEFAULT_INPUT_PATH.exists() else Path.home()),
                    type=str,
                )
            )
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "选择海德堡 E2E 文件",
                start_dir,
                "Heidelberg E2E (*.E2E *.e2e);;All Files (*)",
            )
            if selected:
                self.load_file(selected)

        def reload_current_file(self) -> None:
            if self.input_file is not None:
                self.load_file(str(self.input_file))

        def load_file(self, input_path: str) -> None:
            input_file, study_info, frames = load_heidelberg_fa_dataset(input_path)
            if input_file is None or not input_file.exists():
                QMessageBox.warning(self, "文件不存在", f"找不到文件：\n{input_path}")
                return
            if not input_file.is_file():
                QMessageBox.warning(self, "路径无效", f"不是文件：\n{input_file}")
                return
            if not frames:
                QMessageBox.warning(self, "没有图像", f"文件中未解析到 fundus 图像：\n{input_file}")
                self._set_empty_state("文件存在，但没有解析到可显示的 fundus 帧。")
                return

            self.input_file = input_file
            self.study_info = study_info
            self.frames = frames
            self.play_timer.stop()
            self.play_button.setText("播放")
            self.path_edit.setText(str(input_file))
            self.settings.setValue("last_dir", str(input_file.parent))

            self._populate_filters()
            self._update_dataset_panel()
            self._update_patient_panel()
            self.refresh_visible_frames()
            self.statusBar().showMessage(f"已载入 {len(frames)} 帧：{input_file}", 5000)

        def _populate_filters(self) -> None:
            current_modality = self.modality_combo.currentText()
            current_eye = self.eye_combo.currentData()
            current_series = self.series_combo.currentData()

            modalities = sorted({frame.modality for frame in self.frames}, key=modality_sort_key)
            series_items: list[tuple[str, str]] = []
            seen_series: set[str] = set()
            for frame in self.frames:
                if frame.series_key in seen_series:
                    continue
                seen_series.add(frame.series_key)
                count = sum(1 for item in self.frames if item.series_key == frame.series_key)
                series_items.append(
                    (
                        frame.series_key,
                        f"{frame.series_id} | {frame.modality or '-'} | {frame.laterality_display} | {count} 帧",
                    )
                )

            eye_items = sorted({frame.laterality or "UNKNOWN" for frame in self.frames})

            self.modality_combo.blockSignals(True)
            self.modality_combo.clear()
            self.modality_combo.addItem("全部", "ALL")
            for modality in modalities:
                self.modality_combo.addItem(modality, modality)
            self.modality_combo.blockSignals(False)

            self.eye_combo.blockSignals(True)
            self.eye_combo.clear()
            self.eye_combo.addItem("全部", "ALL")
            for eye in eye_items:
                self.eye_combo.addItem(laterality_to_chinese(eye), eye)
            self.eye_combo.blockSignals(False)

            self.series_combo.blockSignals(True)
            self.series_combo.clear()
            self.series_combo.addItem("全部", "ALL")
            for series_key, label in series_items:
                self.series_combo.addItem(label, series_key)
            self.series_combo.blockSignals(False)

            if current_eye is not None:
                index = self.eye_combo.findData(current_eye)
                if index >= 0:
                    self.eye_combo.setCurrentIndex(index)

            if current_series is not None:
                index = self.series_combo.findData(current_series)
                if index >= 0:
                    self.series_combo.setCurrentIndex(index)

            self._apply_default_modality_if_needed(preferred_text=current_modality)

        def _apply_default_modality_if_needed(self, *_args, preferred_text: str = "") -> None:
            target = preferred_text or self.modality_combo.currentText()
            if self.fa_only_checkbox.isChecked() and self.modality_combo.findText("FA") >= 0:
                target = "FA"
            index = self.modality_combo.findText(target)
            if index < 0:
                index = 0
            self.modality_combo.blockSignals(True)
            self.modality_combo.setCurrentIndex(index)
            self.modality_combo.blockSignals(False)
            if self.frames and _args:
                self.refresh_visible_frames()

        def _update_dataset_panel(self) -> None:
            self.dataset_file_label.setText(str(self.input_file) if self.input_file else "-")
            self.dataset_frames_label.setText(str(len(self.frames)))
            self.dataset_series_label.setText(str(self.study_info.series_count))
            self.dataset_modalities_label.setText(self.study_info.modality_summary or "-")
            self.dataset_eyes_label.setText(self.study_info.laterality_summary or "-")
            self.dataset_timezone_label.setText(self.study_info.timezone_display)
            self.dataset_note_label.setText(self.study_info.timing_note)

        def _update_patient_panel(self) -> None:
            self.patient_name_label.setText(self.study_info.patient_name or "-")
            self.patient_id_label.setText(self.study_info.patient_id or "-")
            self.patient_sex_label.setText(self.study_info.sex_display)
            self.patient_birth_label.setText(self.study_info.birth_date or "-")
            self.patient_device_label.setText(self.study_info.device_display)

        def refresh_visible_frames(self) -> None:
            if not self.frames:
                self._set_empty_state("没有可显示的帧。")
                return

            current_image_id = None
            if 0 <= self.current_visible_index < len(self.visible_frames):
                current_image_id = self.visible_frames[self.current_visible_index].image_id

            selected_modality = self.modality_combo.currentData() or "ALL"
            selected_eye = self.eye_combo.currentData() or "ALL"
            selected_series = self.series_combo.currentData() or "ALL"

            self.visible_frames = []
            for frame in self.frames:
                if selected_modality != "ALL" and frame.modality != selected_modality:
                    continue
                if selected_eye != "ALL" and (frame.laterality or "UNKNOWN") != selected_eye:
                    continue
                if selected_series != "ALL" and frame.series_key != selected_series:
                    continue
                self.visible_frames.append(frame)

            self._populate_table()

            if not self.visible_frames:
                self._set_empty_state("当前筛选条件下没有帧。")
                return

            target_index = 0
            if current_image_id:
                for index, frame in enumerate(self.visible_frames):
                    if frame.image_id == current_image_id:
                        target_index = index
                        break

            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setMaximum(len(self.visible_frames) - 1)
            self.timeline_slider.setValue(target_index)
            self.timeline_slider.blockSignals(False)
            self.set_current_visible_index(target_index)

        def _populate_table(self) -> None:
            self.table.blockSignals(True)
            self.table.setRowCount(len(self.visible_frames))
            for row, frame in enumerate(self.visible_frames):
                values = [
                    str(frame.order_index + 1),
                    str(frame.series_id),
                    frame.modality or "-",
                    frame.laterality_display,
                    str(frame.series_frame_index + 1),
                    str(frame.slice_id),
                    frame.size_display,
                    frame.structure_display,
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column in {0, 1, 4, 5, 6}:
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
            self._redraw_current_frame()

            self.table.blockSignals(True)
            self.table.selectRow(index)
            item = self.table.item(index, 0)
            if item is not None:
                self.table.scrollToItem(item)
            self.table.blockSignals(False)

        def _redraw_current_frame(self) -> None:
            if not self.visible_frames or self.current_visible_index < 0:
                return

            frame = self.visible_frames[self.current_visible_index]
            display_image = apply_image_window(
                frame.image,
                contrast_percent=self.contrast_slider.value(),
                brightness_offset=self.brightness_slider.value(),
            )
            pixmap = image_to_qpixmap(display_image)
            self.image_label.set_pixmap(pixmap, text="无法显示当前帧。")

            self.frame_position_label.setText(
                f"过滤后 {self.current_visible_index + 1}/{len(self.visible_frames)} · "
                f"原始 {frame.order_index + 1}/{len(self.frames)}"
            )
            self.frame_image_id_label.setText(frame.image_id)
            self.frame_series_label.setText(f"{frame.series_id} ({frame.series_key})")
            self.frame_modality_label.setText(frame.modality_display)
            self.frame_eye_label.setText(frame.laterality_display)
            self.frame_sequence_label.setText(frame.sequence_display)
            self.frame_size_label.setText(frame.size_display)
            self.frame_structure_label.setText(frame.structure_display)
            self.frame_time_label.setText(
                f"未解码绝对时间；时区 {frame.timezone_display}"
                if frame.timezone_display != "-"
                else "未解码绝对时间"
            )
            self.metadata_edit.setPlainText(frame.metadata_text)

            self.statusBar().showMessage(
                f"系列 {frame.series_id} | {frame.modality or '-'} | "
                f"{frame.laterality_display} | 帧 {self.current_visible_index + 1}/{len(self.visible_frames)}"
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
            if self.play_timer.isActive():
                self.play_timer.stop()
                self.play_button.setText("播放")
            else:
                self.play_timer.start()
                self.play_button.setText("暂停")

        def current_frame(self) -> Optional[HeidelbergFAFrame]:
            if not self.visible_frames or self.current_visible_index < 0:
                return None
            return self.visible_frames[self.current_visible_index]

        def save_current_frame(self) -> None:
            frame = self.current_frame()
            if frame is None or self.input_file is None:
                return

            default_name = (
                f"{safe_slug(self.input_file.stem)}_"
                f"{safe_slug(frame.modality)}_"
                f"{safe_slug(frame.laterality or 'eye')}_"
                f"s{frame.series_id}_"
                f"f{frame.series_frame_index + 1:03d}.png"
            )
            start_dir = self.settings.value("last_export_dir", str(self.input_file.parent), type=str)
            output_path_text, _ = QFileDialog.getSaveFileName(
                self,
                "保存当前帧",
                str(Path(start_dir) / default_name),
                "PNG (*.png);;BMP (*.bmp);;JPEG (*.jpg *.jpeg)",
            )
            if not output_path_text:
                return

            output_path = Path(output_path_text)
            try:
                save_image_array(frame.image, output_path)
            except Exception as exc:
                QMessageBox.critical(self, "保存失败", str(exc))
                return

            self.settings.setValue("last_export_dir", str(output_path.parent))
            self.statusBar().showMessage(f"已保存：{output_path}", 5000)

        def export_visible_frames(self) -> None:
            if not self.visible_frames or self.input_file is None:
                return

            start_dir = self.settings.value("last_export_dir", str(self.input_file.parent), type=str)
            selected_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", start_dir)
            if not selected_dir:
                return

            output_dir = Path(selected_dir)
            try:
                for index, frame in enumerate(self.visible_frames, start=1):
                    filename = (
                        f"{index:03d}_ord{frame.order_index + 1:03d}_"
                        f"s{frame.series_id}_f{frame.series_frame_index + 1:03d}_"
                        f"{safe_slug(frame.modality)}_{safe_slug(frame.laterality or 'eye')}.png"
                    )
                    save_image_array(frame.image, output_dir / filename)
            except Exception as exc:
                QMessageBox.critical(self, "导出失败", str(exc))
                return

            self.settings.setValue("last_export_dir", str(output_dir))
            self.statusBar().showMessage(f"已导出 {len(self.visible_frames)} 帧到：{output_dir}", 5000)

        def _set_empty_state(self, message: str) -> None:
            self.play_timer.stop()
            self.play_button.setText("播放")
            self.visible_frames = []
            self.current_visible_index = -1
            self.table.setRowCount(0)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setMaximum(0)
            self.timeline_slider.setValue(0)
            self.timeline_slider.blockSignals(False)
            self.image_label.clear_image(message)
            self.metadata_edit.setPlainText("")

            self.frame_position_label.setText("-")
            self.frame_image_id_label.setText("-")
            self.frame_series_label.setText("-")
            self.frame_modality_label.setText("-")
            self.frame_eye_label.setText("-")
            self.frame_sequence_label.setText("-")
            self.frame_size_label.setText("-")
            self.frame_structure_label.setText("-")
            self.frame_time_label.setText("-")


def main() -> int:
    args = parse_args()
    input_file, study_info, frames = load_heidelberg_fa_dataset(args.input_path)

    if args.dump:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if input_file is None or not frames:
            print("没有解析到海德堡 FA 数据。")
            return 1
        dump_study_info(study_info)
        dump_frames(frames)
        return 0

    if not QT_AVAILABLE:
        print("PyQt5 未安装，无法启动 Qt 查看器。可以先使用 --dump 验证解析结果。")
        return 1

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    application = QApplication(sys.argv)
    application.setApplicationName("HeidelbergFAQtViewer")
    application.setStyle("Fusion")

    window = HeidelbergFAViewerWindow(args.input_path)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())
