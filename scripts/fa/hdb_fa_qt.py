from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
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
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

from hdb_fa_parser import (
    DEFAULT_INPUT_PATH,
    HeidelbergFAFrame,
    HeidelbergFAStudyInfo,
    HeidelbergViewerTrack,
    apply_image_window,
    build_heidelberg_viewer_tracks,
    clean_text,
    dump_frames,
    dump_study_info,
    frame_time_text,
    laterality_to_chinese,
    load_heidelberg_fa_dataset,
    normalize_to_uint8,
    parse_args,
    safe_slug,
    save_image_array,
    summarize_tracks_for_console,
    viewer_laterality_label,
    viewer_laterality_short_label, resolve_input_file, build_frame_metadata_text,
)

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

    raise ValueError(f"Unsupported image shape: {display.shape}")


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

if QT_AVAILABLE:

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
            self.ax_image.set_title("Heidelberg FA", color="#E5E7EB")
            self.ax_timeline.set_title("Timeline", color="#E5E7EB")
            self.draw_idle()


    class HeidelbergFAZeissViewerWindow(QMainWindow):
        def __init__(self, startup_path: str | None):
            super().__init__()
            self.settings = QSettings("OpenAI", "HeidelbergFAQtViewer")
            self.input_file: Path | None = None
            self.study_info = HeidelbergFAStudyInfo()
            self.frames: list[HeidelbergFAFrame] = []
            self.tracks: list[HeidelbergViewerTrack] = []
            self.current_track_index = 0
            self.current_frame_index = 0
            self.playing = False
            self.canvas = ViewerCanvas()
            self.play_timer = QTimer(self)
            self.play_timer.timeout.connect(self.advance_frame)

            self.setWindowTitle("Heidelberg FA Viewer")
            self.resize(1520, 940)
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
            if initial_path and resolve_input_file(initial_path) and resolve_input_file(initial_path).exists():
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

            self.eye_filter_combo = QComboBox()
            self.eye_filter_combo.addItem("全部", "ALL")
            self.eye_filter_combo.currentIndexChanged.connect(self.apply_eye_filter)

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

            self.summary_label = QLabel("Ready to load Heidelberg FA data")
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
            self.device_label = QLabel("Device: -")
            self.absolute_time_label.setWordWrap(True)
            self.elapsed_time_label.setWordWrap(True)
            self.source_info_label.setWordWrap(True)
            self.eye_label.setWordWrap(True)
            self.patient_name_label.setWordWrap(True)
            self.patient_id_label.setWordWrap(True)
            self.patient_sex_label.setWordWrap(True)
            self.patient_birth_label.setWordWrap(True)
            self.device_label.setWordWrap(True)
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
                self.device_label,
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

            save_action = QAction("Save Frame", self)
            save_action.triggered.connect(self.save_current_frame)
            toolbar.addAction(save_action)

            export_action = QAction("Export Track", self)
            export_action.triggered.connect(self.export_current_track)
            toolbar.addAction(export_action)

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
            navigation_layout.addWidget(QLabel("Eye filter"), 1, 0)
            navigation_layout.addWidget(self.eye_filter_combo, 1, 1, 1, 3)
            navigation_layout.addWidget(QLabel("Frame"), 2, 0)
            navigation_layout.addWidget(self.frame_slider, 2, 1, 1, 3)
            navigation_layout.addWidget(self.prev_button, 3, 0)
            navigation_layout.addWidget(self.frame_spin, 3, 1)
            navigation_layout.addWidget(self.next_button, 3, 2)
            navigation_layout.addWidget(self.play_button, 3, 3)
            navigation_layout.addWidget(QLabel("FPS"), 4, 0)
            navigation_layout.addWidget(self.fps_spin, 4, 1)
            navigation_layout.addWidget(self.frame_info_label, 4, 2, 1, 2)

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
            patient_layout.addRow("Device", self.device_label)

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
            QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_frame)

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
            self.device_label.setText("Device: -")
            self.metadata_edit.setPlainText("")
            self.summary_label.setText("Ready to load Heidelberg FA data")
            self.path_label.setText("No dataset loaded")
            self.canvas.clear_views()

        def _populate_eye_filter(self):
            current_eye = self.eye_filter_combo.currentData() or "ALL"
            eye_items = sorted(
                {frame.laterality or "UNKNOWN" for frame in self.frames},
                key=lambda value: {
                    "R": 0,
                    "OD": 0,
                    "L": 1,
                    "OS": 1,
                    "UNKNOWN": 2,
                }.get(clean_text(value).upper(), 99),
            )

            self.eye_filter_combo.blockSignals(True)
            self.eye_filter_combo.clear()
            self.eye_filter_combo.addItem("全部", "ALL")
            for eye in eye_items:
                self.eye_filter_combo.addItem(laterality_to_chinese(eye), eye)
            index = self.eye_filter_combo.findData(current_eye)
            self.eye_filter_combo.setCurrentIndex(index if index >= 0 else 0)
            self.eye_filter_combo.blockSignals(False)

        def _refresh_track_options(self, preferred_key: str | None = None):
            self.sequence_combo.blockSignals(True)
            self.sequence_combo.clear()
            for track in self.tracks:
                self.sequence_combo.addItem(track.label, track.key)
            self.sequence_combo.blockSignals(False)

            if not self.tracks:
                self.current_track_index = 0
                return

            target_index = 0
            if preferred_key:
                matched_index = self.sequence_combo.findData(preferred_key)
                if matched_index >= 0:
                    target_index = matched_index

            self.current_track_index = target_index
            self.sequence_combo.blockSignals(True)
            self.sequence_combo.setCurrentIndex(target_index)
            self.sequence_combo.blockSignals(False)

        def _update_summary_label(self):
            selected_eye = self.eye_filter_combo.currentData() or "ALL"
            summary_lines = summarize_tracks_for_console(self.tracks)
            eye_filter_text = "全部" if selected_eye == "ALL" else laterality_to_chinese(selected_eye)
            summary_text = f"Eye filter: {eye_filter_text}\nLoaded tracks:"
            if summary_lines:
                summary_text += "\n" + "\n".join(summary_lines[:8])
                if len(summary_lines) > 8:
                    summary_text += "\n..."
            else:
                summary_text += "\n-"
            summary_text += (
                f"\n\nPatient: {self.study_info.patient_name or '-'} | ID: {self.study_info.patient_id or '-'} | "
                f"Sex: {self.study_info.sex_display} | DOB: {self.study_info.birth_date or '-'}"
            )
            summary_text += (
                f"\nDevice: {self.study_info.device_display} | Study: {self.study_info.study_datetime_iso} | "
                f"TZ: {self.study_info.timezone_display}"
            )
            self.summary_label.setText(summary_text)

        def current_track(self) -> HeidelbergViewerTrack | None:
            if not self.tracks:
                return None
            return self.tracks[self.current_track_index]

        def current_frame(self) -> HeidelbergFAFrame | None:
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
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select Heidelberg E2E file",
                start_dir,
                "Heidelberg E2E (*.E2E *.e2e);;All Files (*)",
            )
            if not selected:
                return
            self.path_edit.setText(selected)
            self.load_path(selected)

        def reload_from_path(self):
            self.load_path(self.path_edit.text().strip())

        def load_path(self, path_text: str):
            if not path_text:
                QMessageBox.warning(self, "Missing path", "Please choose a Heidelberg E2E file.")
                return

            try:
                input_file, study_info, frames = load_heidelberg_fa_dataset(path_text)
            except Exception as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
                return

            if input_file is None or not frames:
                QMessageBox.warning(self, "No frames", "No Heidelberg FA frames were found.")
                return

            self.input_file = input_file
            self.study_info = study_info
            self.frames = frames
            self.current_frame_index = 0
            self.path_label.setText(str(input_file))
            self.settings.setValue("last_dir", str(input_file.parent))
            self._populate_eye_filter()
            self.apply_eye_filter()

        def apply_eye_filter(self):
            if not self.frames:
                self.tracks = []
                self._refresh_track_options()
                self._set_empty_state()
                return

            preferred_key = self.current_track().key if self.current_track() else None
            selected_eye = self.eye_filter_combo.currentData() or "ALL"
            filtered_frames = [
                frame
                for frame in self.frames
                if selected_eye == "ALL" or (frame.laterality or "UNKNOWN") == selected_eye
            ]

            self.tracks = build_heidelberg_viewer_tracks(filtered_frames)
            self.current_frame_index = 0
            self._refresh_track_options(preferred_key=preferred_key)
            self._update_summary_label()

            if not self.tracks:
                self._set_empty_state()
                self.statusBar().showMessage(f"当前眼别筛选下没有图像：{laterality_to_chinese(selected_eye)}", 5000)
                return

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
            self.patient_name_label.setText(self.study_info.patient_name or "-")
            self.patient_id_label.setText(self.study_info.patient_id or "-")
            self.patient_sex_label.setText(self.study_info.sex_display)
            self.patient_birth_label.setText(self.study_info.birth_date or "-")
            self.device_label.setText(self.study_info.device_display)
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

            if display_image.ndim == 2:
                self.canvas.ax_image.imshow(display_image, cmap="gray", origin="upper")
            else:
                self.canvas.ax_image.imshow(display_image[:, :, :3], origin="upper")
            self.canvas.ax_image.axis("off")
            frame_eye_value = frame.laterality or track.display_laterality
            self.eye_label.setText(viewer_laterality_label(frame_eye_value))
            self.canvas.ax_image.set_title(
                f"Eye: {viewer_laterality_short_label(frame_eye_value)} | Frame {self.current_frame_index + 1}/{track.frame_count} | {time_text}",
                color="#E5E7EB",
            )

            datetime_frames = [item for item in track.frames if item.acquisition_datetime_local is not None]
            has_elapsed_time = bool(datetime_frames)
            if has_elapsed_time:
                first_datetime = datetime_frames[0].acquisition_datetime_local
                x_values = [
                    (
                        (item.acquisition_datetime_local - first_datetime).total_seconds()
                        if item.acquisition_datetime_local is not None
                        else float(index)
                    )
                    for index, item in enumerate(track.frames)
                ]
            elif any(item.time_of_day_ms is not None for item in track.frames):
                base_time = min(item.time_of_day_ms for item in track.frames if item.time_of_day_ms is not None)
                x_values = [
                    (
                        (item.time_of_day_ms - base_time) / 1000.0
                        if item.time_of_day_ms is not None
                        else float(index)
                    )
                    for index, item in enumerate(track.frames)
                ]
                has_elapsed_time = True
            else:
                x_values = [float(index) for index, _item in enumerate(track.frames)]

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
                if item.series_key != current_series:
                    current_series = item.series_key
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
            self.canvas.ax_timeline.axvline(
                x_values[self.current_frame_index],
                color="#F97316",
                linewidth=1.4,
                alpha=0.9,
            )

            elapsed_text = "-"
            if has_elapsed_time:
                elapsed_text = f"{x_values[self.current_frame_index]:.3f}s"

            self.frame_info_label.setText(f"Frame: {self.current_frame_index + 1}/{track.frame_count}")
            self.time_info_label.setText(f"{time_text} | elapsed={elapsed_text}")
            self.absolute_time_label.setText(frame.acquisition_datetime_iso or frame.time_display)
            self.elapsed_time_label.setText(elapsed_text)
            self.source_info_label.setText(
                f"Series {frame.series_id} | {frame.modality_display} | {frame.acquisition_source or '-'}"
            )
            self.metadata_edit.setPlainText(
                build_frame_metadata_text(self.input_file, self.study_info, frame) if self.input_file else frame.metadata_text
            )
            self.statusBar().showMessage(
                f"{self.study_info.patient_name or '-'} | series {frame.series_id} | frame {self.current_frame_index + 1}/{track.frame_count} | {time_text}"
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

        def save_current_frame(self):
            frame = self.current_frame()
            if frame is None or self.input_file is None:
                return
            default_name = (
                f"{safe_slug(self.input_file.stem)}_"
                f"{safe_slug(frame.modality)}_"
                f"{safe_slug(frame.laterality or 'eye')}_"
                f"s{frame.series_id}_f{frame.series_frame_index + 1:03d}.png"
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

        def export_current_track(self):
            track = self.current_track()
            if track is None or self.input_file is None:
                return
            start_dir = self.settings.value("last_export_dir", str(self.input_file.parent), type=str)
            selected_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", start_dir)
            if not selected_dir:
                return
            output_dir = Path(selected_dir)
            try:
                for index, frame in enumerate(track.frames, start=1):
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
            self.statusBar().showMessage(f"已导出 {len(track.frames)} 帧到：{output_dir}", 5000)


def main() -> int:
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    application = QApplication(sys.argv)
    application.setApplicationName("HeidelbergFAQtViewer")
    application.setStyle("Fusion")

    window = HeidelbergFAZeissViewerWindow(DEFAULT_INPUT_PATH)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())
