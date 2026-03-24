from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
    acquisition_datetime: Optional[datetime]
    elapsed_seconds: Optional[float]
    width: Optional[int]
    height: Optional[int]

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
            return f"{self.width} × {self.height}"
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
            return f"{self.laterality} ({self.laterality_note})"
        return self.laterality or "-"

    @property
    def sex_display(self) -> str:
        return self.sex or "-"

    @property
    def exam_date_display(self) -> str:
        if self.exam_date and self.register_date and self.register_date != self.exam_date:
            return f"{self.exam_date} / {self.register_date}"
        return self.exam_date or self.register_date or "-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topcon FA Qt 时序查看器")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR) if DEFAULT_INPUT_DIR.exists() else None,
        help="包含 DATAFILE 和 IM*.JPG 的目录，或目录内任意文件路径。",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="只解析并打印时序，不启动 Qt 窗口。",
    )
    return parser.parse_args()


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


def parse_elapsed_seconds(label: str) -> Optional[float]:
    match = ELAPSED_LABEL_PATTERN.match(label.strip())
    if not match:
        return None
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    return minutes * 60.0 + seconds


def parse_datetime(date_text: str, time_text: str) -> Optional[datetime]:
    if not date_text or not time_text:
        return None
    try:
        return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def resolve_input_dir(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        return path.parent
    return path


def load_image_size(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def normalize_sex(value: str) -> str:
    value = value.strip().upper()
    return {
        "M": "男",
        "F": "女",
        "O": "其他",
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
            return "右眼", f"由 {text} 推断"
        if eye in {"OS", "L"}:
            return "左眼", f"由 {text} 推断"
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
    input_path: Optional[str],
) -> tuple[Optional[Path], TopconFAStudyInfo, list[TopconFAFrame]]:
    input_dir = resolve_input_dir(input_path)
    if input_dir is None or not input_dir.exists() or not input_dir.is_dir():
        return input_dir, TopconFAStudyInfo(), []

    study_info = parse_study_info(input_dir)
    frames = parse_datafile_records(input_dir)
    if not frames:
        frames = fallback_records_from_images(input_dir)
    return input_dir, study_info, frames


def load_topcon_fa_frames(input_path: Optional[str]) -> tuple[Optional[Path], list[TopconFAFrame]]:
    input_dir, _, frames = load_topcon_fa_dataset(input_path)
    return input_dir, frames


def modality_summary(frames: list[TopconFAFrame]) -> str:
    counts = Counter(frame.modality for frame in frames)
    return ", ".join(f"{modality} {count}" for modality, count in counts.items()) or "-"


class ScaledImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("请打开一个 Topcon FA 目录")
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

    def set_image(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        self._pixmap = pixmap
        self.setText("" if not pixmap.isNull() else f"无法加载图像\n{image_path}")
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


class TopconFAViewerWindow(QMainWindow):
    def __init__(self, input_path: Optional[str]) -> None:
        super().__init__()
        self.settings = QSettings("OpenAI", "TopconFAQtViewer")
        self.input_dir: Optional[Path] = None
        self.study_info = TopconFAStudyInfo()
        self.frames: list[TopconFAFrame] = []
        self.visible_frames: list[TopconFAFrame] = []
        self.current_visible_index = -1
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)

        self.setWindowTitle("Topcon FA 时序查看器")
        self.resize(1560, 960)
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self._apply_styles()

        self.image_label = ScaledImageLabel()
        self.setStatusBar(QStatusBar(self))
        self._build_controls()
        self._build_layout()
        self._install_shortcuts()

        if input_path:
            self.load_directory(input_path)
        else:
            self._set_empty_state("未指定目录，请点击“打开目录”。")

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
            """
        )

    def _build_controls(self) -> None:
        self.open_button = QPushButton("打开目录")
        self.open_button.clicked.connect(self.choose_directory)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        self.modality_combo = QComboBox()
        self.modality_combo.currentIndexChanged.connect(self.refresh_visible_frames)

        self.hide_proofsheet_checkbox = QCheckBox("隐藏 Proofsheet")
        self.hide_proofsheet_checkbox.setChecked(True)
        self.hide_proofsheet_checkbox.toggled.connect(self.refresh_visible_frames)

        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)

        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 30)
        self.fps_spinbox.setValue(3)
        self.fps_spinbox.setSuffix(" fps")
        self.fps_spinbox.valueChanged.connect(self._update_play_interval)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self.set_current_visible_index)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "文件", "模式", "标签", "序列", "采集时间"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._sync_selection_from_table)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(1, self.table.horizontalHeader().Stretch)

        self.dataset_dir_label = QLabel("-")
        self.dataset_frames_label = QLabel("-")
        self.dataset_modalities_label = QLabel("-")
        self.dataset_time_range_label = QLabel("-")

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
        self.frame_modality_label = QLabel("-")
        self.frame_label_label = QLabel("-")
        self.frame_elapsed_label = QLabel("-")
        self.frame_time_label = QLabel("-")
        self.frame_size_label = QLabel("-")
        self.frame_device_label = QLabel("-")

        self._update_play_interval()

    def _build_layout(self) -> None:
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.path_edit, 1)
        top_layout.addWidget(QLabel("模式"))
        top_layout.addWidget(self.modality_combo)
        top_layout.addWidget(self.hide_proofsheet_checkbox)
        top_layout.addWidget(self.play_button)
        top_layout.addWidget(self.fps_spinbox)

        dataset_box = QGroupBox("数据集")
        dataset_form = QFormLayout(dataset_box)
        dataset_form.addRow("目录", self.dataset_dir_label)
        dataset_form.addRow("帧数", self.dataset_frames_label)
        dataset_form.addRow("模式", self.dataset_modalities_label)
        dataset_form.addRow("时间范围", self.dataset_time_range_label)

        patient_box = QGroupBox("患者信息")
        patient_form = QFormLayout(patient_box)
        patient_form.addRow("姓名", self.patient_name_label)
        patient_form.addRow("患者ID", self.patient_id_label)
        patient_form.addRow("性别", self.patient_sex_label)
        patient_form.addRow("出生日期", self.patient_birth_label)
        patient_form.addRow("检查日期", self.patient_exam_label)
        patient_form.addRow("眼别", self.patient_eye_label)
        patient_form.addRow("检查标识", self.patient_study_code_label)
        patient_form.addRow("设备", self.patient_device_label)

        frame_box = QGroupBox("当前帧")
        frame_form = QFormLayout(frame_box)
        frame_form.addRow("位置", self.frame_position_label)
        frame_form.addRow("文件", self.frame_file_label)
        frame_form.addRow("模式", self.frame_modality_label)
        frame_form.addRow("标签", self.frame_label_label)
        frame_form.addRow("序列", self.frame_elapsed_label)
        frame_form.addRow("采集时间", self.frame_time_label)
        frame_form.addRow("尺寸", self.frame_size_label)
        frame_form.addRow("设备", self.frame_device_label)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.image_label, 1)
        left_layout.addWidget(dataset_box)
        left_layout.addWidget(patient_box)
        left_layout.addWidget(frame_box)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
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
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_directory)

    def _update_play_interval(self) -> None:
        fps = max(1, self.fps_spinbox.value())
        self.play_timer.setInterval(int(1000 / fps))

    def choose_directory(self) -> None:
        start_dir = str(self.input_dir) if self.input_dir else str(DEFAULT_INPUT_DIR.parent if DEFAULT_INPUT_DIR.exists() else Path.home())
        selected = QFileDialog.getExistingDirectory(self, "选择 Topcon FA 目录", start_dir)
        if selected:
            self.load_directory(selected)

    def load_directory(self, input_path: str) -> None:
        input_dir, study_info, frames = load_topcon_fa_dataset(input_path)
        if input_dir is None or not input_dir.exists():
            QMessageBox.warning(self, "目录不存在", f"找不到目录：\n{input_path}")
            return
        if not frames:
            QMessageBox.warning(self, "没有图像", f"目录中未找到可解析的 Topcon FA 数据：\n{input_dir}")
            self._set_empty_state("目录存在，但没有解析到 IM*.JPG。")
            return

        self.input_dir = input_dir
        self.study_info = study_info
        self.frames = frames
        self.path_edit.setText(str(input_dir))
        self._populate_modality_combo()
        self._update_dataset_panel()
        self._update_patient_panel()
        self.refresh_visible_frames()
        self.statusBar().showMessage(f"已载入 {len(frames)} 帧：{input_dir}", 5000)

    def _populate_modality_combo(self) -> None:
        current = self.modality_combo.currentText()
        self.modality_combo.blockSignals(True)
        self.modality_combo.clear()
        self.modality_combo.addItem("全部")
        for modality in sorted({frame.modality for frame in self.frames}):
            self.modality_combo.addItem(modality)
        preferred = "Fluorescein" if "Fluorescein" in {frame.modality for frame in self.frames} else current
        if preferred:
            index = self.modality_combo.findText(preferred)
            if index >= 0:
                self.modality_combo.setCurrentIndex(index)
        self.modality_combo.blockSignals(False)

    def _update_dataset_panel(self) -> None:
        self.dataset_dir_label.setText(str(self.input_dir) if self.input_dir else "-")
        self.dataset_frames_label.setText(str(len(self.frames)))
        self.dataset_modalities_label.setText(modality_summary(self.frames))
        datetimes = [frame.acquisition_datetime for frame in self.frames if frame.acquisition_datetime is not None]
        if datetimes:
            earliest = min(datetimes)
            latest = max(datetimes)
            self.dataset_time_range_label.setText(
                f"{earliest:%Y-%m-%d %H:%M:%S} ~ {latest:%Y-%m-%d %H:%M:%S}"
            )
        else:
            self.dataset_time_range_label.setText("-")

    def _update_patient_panel(self) -> None:
        exam_date_display = self.study_info.exam_date_display
        if exam_date_display == "-":
            datetimes = [frame.acquisition_datetime for frame in self.frames if frame.acquisition_datetime is not None]
            if datetimes:
                exam_date_display = min(datetimes).strftime("%Y-%m-%d")
        self.patient_name_label.setText(self.study_info.patient_name or "-")
        self.patient_id_label.setText(self.study_info.patient_id or "-")
        self.patient_sex_label.setText(self.study_info.sex_display)
        self.patient_birth_label.setText(self.study_info.birth_date or "-")
        self.patient_exam_label.setText(exam_date_display)
        self.patient_eye_label.setText(self.study_info.laterality_display)
        self.patient_study_code_label.setText(self.study_info.study_code or "-")
        self.patient_device_label.setText(self.study_info.device_model or "-")

    def refresh_visible_frames(self) -> None:
        if not self.frames:
            self._set_empty_state("没有可显示的帧。")
            return

        current_file = None
        if 0 <= self.current_visible_index < len(self.visible_frames):
            current_file = self.visible_frames[self.current_visible_index].filename

        modality = self.modality_combo.currentText()
        hide_proofsheets = self.hide_proofsheet_checkbox.isChecked()

        self.visible_frames = []
        for frame in self.frames:
            if modality != "全部" and frame.modality != modality:
                continue
            if hide_proofsheets and frame.is_proofsheet:
                continue
            self.visible_frames.append(frame)

        self._populate_table()

        if not self.visible_frames:
            self._set_empty_state("当前筛选条件下没有帧。")
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
        self.set_current_visible_index(target_index)

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.visible_frames))
        for row, frame in enumerate(self.visible_frames):
            values = [
                str(frame.order_index + 1),
                frame.filename,
                frame.modality,
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
        self.image_label.set_image(frame.image_path)

        self.frame_position_label.setText(
            f"过滤后 {index + 1}/{len(self.visible_frames)} · 原始 {frame.order_index + 1}/{len(self.frames)}"
        )
        self.frame_file_label.setText(str(frame.image_path))
        self.frame_modality_label.setText(frame.modality or "-")
        self.frame_label_label.setText(frame.label or "-")
        self.frame_elapsed_label.setText(frame.elapsed_display)
        self.frame_time_label.setText(frame.acquisition_display)
        self.frame_size_label.setText(frame.size_display)
        self.frame_device_label.setText(frame.device or "-")

        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.scrollToItem(self.table.item(index, 0))
        self.table.blockSignals(False)

        self.statusBar().showMessage(
            f"{frame.filename} | {frame.modality} | {frame.label or '-'} | {frame.acquisition_display}"
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

    def _set_empty_state(self, message: str) -> None:
        self.play_timer.stop()
        self.play_button.setText("播放")
        self.study_info = TopconFAStudyInfo()
        self.visible_frames = []
        self.current_visible_index = -1
        self.table.setRowCount(0)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.image_label.clear_image(message)

        self.frame_position_label.setText("-")
        self.frame_file_label.setText("-")
        self.frame_modality_label.setText("-")
        self.frame_label_label.setText("-")
        self.frame_elapsed_label.setText("-")
        self.frame_time_label.setText("-")
        self.frame_size_label.setText("-")
        self.frame_device_label.setText("-")
        self.patient_name_label.setText("-")
        self.patient_id_label.setText("-")
        self.patient_sex_label.setText("-")
        self.patient_birth_label.setText("-")
        self.patient_exam_label.setText("-")
        self.patient_eye_label.setText("-")
        self.patient_study_code_label.setText("-")
        self.patient_device_label.setText("-")


def dump_study_info(study_info: TopconFAStudyInfo) -> None:
    print("patient_name\t", study_info.patient_name or "-")
    print("patient_id\t", study_info.patient_id or "-")
    print("sex\t", study_info.sex_display)
    print("birth_date\t", study_info.birth_date or "-")
    print("exam_date\t", study_info.exam_date_display)
    print("laterality\t", study_info.laterality_display)
    print("study_code\t", study_info.study_code or "-")
    print("device_model\t", study_info.device_model or "-")
    print()


def dump_frames(frames: list[TopconFAFrame]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("index\tfilename\tmodality\tlabel\telapsed_s\tacquisition")
    for frame in frames:
        elapsed = "" if frame.elapsed_seconds is None else f"{frame.elapsed_seconds:.1f}"
        print(
            f"{frame.order_index + 1}\t{frame.filename}\t{frame.modality}\t{frame.label}\t"
            f"{elapsed}\t{frame.acquisition_display}"
        )


def main() -> int:
    args = parse_args()
    input_dir, study_info, frames = load_topcon_fa_dataset(args.input_path)

    if args.dump:
        if input_dir is None or not frames:
            print("没有解析到 Topcon FA 数据。")
            return 1
        dump_study_info(study_info)
        dump_frames(frames)
        return 0

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    application = QApplication(sys.argv)
    application.setApplicationName("TopconFAQtViewer")
    application.setStyle("Fusion")

    window = TopconFAViewerWindow(args.input_path)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())
