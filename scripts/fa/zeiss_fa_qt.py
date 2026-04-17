from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .zeiss_fa_parser import (
        DEFAULT_INPUT_PATH,
        FrameRecord,
        SeriesRecord,
        clean_text,
        load_zeiss_fa_series,
        normalize_to_uint8,
    )
except ImportError:
    from zeiss_fa_parser import (
        DEFAULT_INPUT_PATH,
        FrameRecord,
        SeriesRecord,
        clean_text,
        load_zeiss_fa_series,
        normalize_to_uint8,
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
        time_suffix = f" | {first_datetime_iso.split('T')[-1]} -> {last_datetime_iso.split('T')[-1]}"
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
            grid = figure.add_gridspec(
                2,
                1,
                height_ratios=[5.6, 1.0],
                hspace=0.18,
                left=0.04,
                right=0.985,
                top=0.95,
                bottom=0.07,
            )
            self.ax_image = figure.add_subplot(grid[0, 0])
            self.ax_timeline = figure.add_subplot(grid[1, 0])
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
            self.image_view_xlim: tuple[float, float] | None = None
            self.image_view_ylim: tuple[float, float] | None = None
            self.image_full_xlim: tuple[float, float] | None = None
            self.image_full_ylim: tuple[float, float] | None = None
            self.pending_image_view_xlim: tuple[float, float] | None = None
            self.pending_image_view_ylim: tuple[float, float] | None = None
            self.image_drag_state: dict[str, Any] | None = None
            self.canvas = ViewerCanvas()
            self.play_timer = QTimer(self)
            self.play_timer.timeout.connect(self.advance_frame)
            self.image_view_timer = QTimer(self)
            self.image_view_timer.setSingleShot(True)
            self.image_view_timer.setInterval(16)
            self.image_view_timer.timeout.connect(self._flush_image_view_update)

            self.setWindowTitle("Zeiss FA Viewer")
            self.resize(1760, 1040)
            self.setMinimumSize(1440, 900)
            self.setFont(QFont("Microsoft YaHei UI", 10))
            self._apply_styles()
            self.setStatusBar(QStatusBar(self))

            self._build_controls()
            self._build_toolbar()
            self._build_layout()
            self._install_shortcuts()
            self._set_empty_state()

            self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)
            self.canvas.mpl_connect("button_press_event", self.on_canvas_button_press)
            self.canvas.mpl_connect("motion_notify_event", self.on_canvas_mouse_move)
            self.canvas.mpl_connect("button_release_event", self.on_canvas_button_release)

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

            self.reset_zoom_button = QPushButton("Reset Zoom")
            self.reset_zoom_button.clicked.connect(self.reset_image_view)

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

            reset_zoom_action = QAction("Reset Zoom", self)
            reset_zoom_action.triggered.connect(self.reset_image_view)
            toolbar.addAction(reset_zoom_action)

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
            display_layout.addRow("Image view", self.reset_zoom_button)
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
            control_widget.setMinimumWidth(360)
            control_widget.setMaximumWidth(380)
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
            splitter.setSizes([370, 1360])

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
            self.image_drag_state = None
            self.image_view_xlim = None
            self.image_view_ylim = None
            self.image_full_xlim = None
            self.image_full_ylim = None
            self.pending_image_view_xlim = None
            self.pending_image_view_ylim = None
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
            self.image_drag_state = None
            self.image_view_xlim = None
            self.image_view_ylim = None
            self.image_full_xlim = None
            self.image_full_ylim = None
            self.pending_image_view_xlim = None
            self.pending_image_view_ylim = None
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

            self.image_drag_state = None
            self.image_view_xlim = None
            self.image_view_ylim = None
            self.pending_image_view_xlim = None
            self.pending_image_view_ylim = None
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
            self._update_image_full_view(display_image)
            self.canvas.ax_image.set_title(
                f"Eye: {laterality_display_text(track.laterality)} | Frame {self.current_frame_index + 1}/{track.frame_count} | {time_text}",
                color="#E5E7EB",
            )
            self._apply_saved_image_view()

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

        def _full_image_view(self) -> tuple[tuple[float, float], tuple[float, float]]:
            if self.image_full_xlim is not None and self.image_full_ylim is not None:
                return self.image_full_xlim, self.image_full_ylim
            x_limits = tuple(float(value) for value in self.canvas.ax_image.get_xlim())
            y_limits = tuple(float(value) for value in self.canvas.ax_image.get_ylim())
            return x_limits, y_limits

        def _update_image_full_view(self, image: np.ndarray) -> None:
            height, width = image.shape[:2]
            self.image_full_xlim = (-0.5, float(width) - 0.5)
            self.image_full_ylim = (float(height) - 0.5, -0.5)

        def _clamp_axis_limits(
            self,
            limits: tuple[float, float],
            full_limits: tuple[float, float],
            *,
            min_span: float = 16.0,
        ) -> tuple[float, float]:
            orientation_forward = limits[1] >= limits[0]
            full_min = min(full_limits)
            full_max = max(full_limits)
            current_min = min(limits)
            current_max = max(limits)
            full_span = max(full_max - full_min, 1.0)
            span = max(min_span, min(current_max - current_min, full_span))
            center = (current_min + current_max) / 2.0
            lower = center - span / 2.0
            upper = center + span / 2.0
            if lower < full_min:
                upper += full_min - lower
                lower = full_min
            if upper > full_max:
                lower -= upper - full_max
                upper = full_max
            lower = max(full_min, lower)
            upper = min(full_max, upper)
            if (upper - lower) < span:
                if lower <= full_min:
                    upper = min(full_max, lower + span)
                elif upper >= full_max:
                    lower = max(full_min, upper - span)
            if orientation_forward:
                return (lower, upper)
            return (upper, lower)

        def _save_image_view(self) -> None:
            self.image_view_xlim = tuple(float(value) for value in self.canvas.ax_image.get_xlim())
            self.image_view_ylim = tuple(float(value) for value in self.canvas.ax_image.get_ylim())

        def _apply_saved_image_view(self) -> None:
            if self.image_view_xlim is None or self.image_view_ylim is None:
                self._save_image_view()
                return
            full_xlim, full_ylim = self._full_image_view()
            clamped_xlim = self._clamp_axis_limits(self.image_view_xlim, full_xlim)
            clamped_ylim = self._clamp_axis_limits(self.image_view_ylim, full_ylim)
            self.canvas.ax_image.set_xlim(clamped_xlim)
            self.canvas.ax_image.set_ylim(clamped_ylim)
            self.image_view_xlim = clamped_xlim
            self.image_view_ylim = clamped_ylim

        def _request_image_view_update(
            self,
            xlim: tuple[float, float],
            ylim: tuple[float, float],
        ) -> None:
            self.pending_image_view_xlim = tuple(float(value) for value in xlim)
            self.pending_image_view_ylim = tuple(float(value) for value in ylim)
            if not self.image_view_timer.isActive():
                self.image_view_timer.start()

        def _flush_image_view_update(self) -> None:
            if self.pending_image_view_xlim is None or self.pending_image_view_ylim is None:
                return
            self.image_view_xlim = self.pending_image_view_xlim
            self.image_view_ylim = self.pending_image_view_ylim
            self.canvas.ax_image.set_xlim(self.image_view_xlim)
            self.canvas.ax_image.set_ylim(self.image_view_ylim)
            self.pending_image_view_xlim = None
            self.pending_image_view_ylim = None
            self.canvas.draw_idle()

        def reset_image_view(self) -> None:
            self.image_drag_state = None
            self.image_view_xlim = None
            self.image_view_ylim = None
            self.pending_image_view_xlim = None
            self.pending_image_view_ylim = None
            track = self.current_track()
            frame = self.current_frame()
            if track is not None and frame is not None:
                self.redraw_views()

        def _zoom_image_view(self, *, xdata: float, ydata: float, scale_factor: float) -> None:
            current_xlim = tuple(float(value) for value in self.canvas.ax_image.get_xlim())
            current_ylim = tuple(float(value) for value in self.canvas.ax_image.get_ylim())
            new_xlim = (
                xdata + (current_xlim[0] - xdata) * scale_factor,
                xdata + (current_xlim[1] - xdata) * scale_factor,
            )
            new_ylim = (
                ydata + (current_ylim[0] - ydata) * scale_factor,
                ydata + (current_ylim[1] - ydata) * scale_factor,
            )
            full_xlim, full_ylim = self._full_image_view()
            self.image_view_xlim = self._clamp_axis_limits(new_xlim, full_xlim)
            self.image_view_ylim = self._clamp_axis_limits(new_ylim, full_ylim)
            self._request_image_view_update(self.image_view_xlim, self.image_view_ylim)

        def _pan_image_view(self, *, dx: float, dy: float) -> None:
            if self.image_drag_state is None:
                return
            press_xlim = self.image_drag_state["xlim"]
            press_ylim = self.image_drag_state["ylim"]
            full_xlim, full_ylim = self._full_image_view()
            self.image_view_xlim = self._clamp_axis_limits(
                (press_xlim[0] - dx, press_xlim[1] - dx),
                full_xlim,
            )
            self.image_view_ylim = self._clamp_axis_limits(
                (press_ylim[0] - dy, press_ylim[1] - dy),
                full_ylim,
            )
            self._request_image_view_update(self.image_view_xlim, self.image_view_ylim)

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

        def on_canvas_button_press(self, event):
            if event.inaxes != self.canvas.ax_image:
                return
            if getattr(event, "dblclick", False):
                self.reset_image_view()
                return
            if event.button != 1 or event.xdata is None or event.ydata is None:
                return
            self.image_drag_state = {
                "anchor_x": float(event.xdata),
                "anchor_y": float(event.ydata),
                "xlim": tuple(float(value) for value in self.canvas.ax_image.get_xlim()),
                "ylim": tuple(float(value) for value in self.canvas.ax_image.get_ylim()),
            }

        def on_canvas_mouse_move(self, event):
            if self.image_drag_state is None or event.inaxes != self.canvas.ax_image:
                return
            if event.xdata is None or event.ydata is None:
                return
            dx = float(event.xdata) - float(self.image_drag_state["anchor_x"])
            dy = float(event.ydata) - float(self.image_drag_state["anchor_y"])
            self._pan_image_view(dx=dx, dy=dy)

        def on_canvas_button_release(self, event):
            self.image_drag_state = None

        def on_canvas_scroll(self, event):
            if event.inaxes == self.canvas.ax_image and event.xdata is not None and event.ydata is not None:
                scale_factor = 1 / 1.2 if event.button == "up" else 1.2
                self._zoom_image_view(
                    xdata=float(event.xdata),
                    ydata=float(event.ydata),
                    scale_factor=scale_factor,
                )
                return
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
