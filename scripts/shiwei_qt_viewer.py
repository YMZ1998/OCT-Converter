import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt5.QtCore import QSettings, Qt  # noqa: E402
from PyQt5.QtGui import QFont, QKeySequence  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QAction,
    QApplication,
    QCheckBox,
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

from scripts.parse_shiwei_location import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    SEGMENTATION_COLORS,
    ensure_rgb,
    get_segmentation_curves,
    load_shiwei_data,
    normalize_to_uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Professional Shiwei OCT Qt viewer.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Shiwei DICOM files.",
    )
    parser.add_argument("--bscan-file", default=None, help="Optional structural B-scan DICOM path.")
    parser.add_argument("--fundus-file", default=None, help="Optional fundus/CSSO DICOM path.")
    parser.add_argument("--seg-file", default=None, help="Optional segmentation DICOM path.")
    return parser.parse_args()


class ViewerCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(12, 7), facecolor="#111827", tight_layout=True)
        super().__init__(self.figure)
        self.ax_fundus = self.figure.add_subplot(1, 2, 1)
        self.ax_bscan = self.figure.add_subplot(1, 2, 2)
        self._style_axes()

    def _style_axes(self):
        for ax in (self.ax_fundus, self.ax_bscan):
            ax.set_facecolor("#0B1220")
            ax.tick_params(colors="#CBD5E1")
            for spine in ax.spines.values():
                spine.set_color("#334155")

    def clear_views(self):
        self.ax_fundus.clear()
        self.ax_bscan.clear()
        self._style_axes()
        self.ax_fundus.set_title("Fundus / CSSO", color="#E5E7EB")
        self.ax_bscan.set_title("B-scan", color="#E5E7EB")
        self.draw_idle()


class ShiweiViewerWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.settings = QSettings("OpenAI", "ShiweiQtViewer")

        self.volume = None
        self.fundus = None
        self.coordinates = []
        self.angles = []
        self.segmentation_surfaces = None
        self.segmentation_orientation = None
        self.bscan_file = None
        self.fundus_file = None
        self.seg_file = None

        self.setWindowTitle("视微 OCT 查看器")
        self.resize(1560, 920)
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self._apply_styles()

        self.canvas = ViewerCanvas()
        self.setStatusBar(QStatusBar(self))

        self._build_controls()
        self._build_toolbar()
        self._build_layout()
        self._install_shortcuts()
        self._restore_inputs()
        self._set_empty_state()

        initial_input_dir = self.input_dir_edit.text().strip()
        if initial_input_dir and Path(initial_input_dir).exists():
            self.load_data()

    def _build_controls(self):
        self.open_button = QPushButton("打开目录")
        self.open_button.clicked.connect(self.open_path_dialog)

        self.load_button = QPushButton("重新加载")
        self.load_button.clicked.connect(self.load_data)

        self.snapshot_button = QPushButton("保存截图")
        self.snapshot_button.clicked.connect(self.save_snapshot)

        self.reset_override_button = QPushButton("清空覆盖")
        self.reset_override_button.clicked.connect(self._reset_file_overrides)

        self.path_label = QLabel("未打开目录")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("PathLabel")

        self.input_dir_edit = QLineEdit()
        self.bscan_file_edit = QLineEdit()
        self.fundus_file_edit = QLineEdit()
        self.seg_file_edit = QLineEdit()

        self.input_dir_browse = QPushButton("浏览")
        self.input_dir_browse.clicked.connect(self.open_path_dialog)
        self.bscan_browse = QPushButton("浏览")
        self.bscan_browse.clicked.connect(self._browse_bscan_file)
        self.fundus_browse = QPushButton("浏览")
        self.fundus_browse.clicked.connect(self._browse_fundus_file)
        self.seg_browse = QPushButton("浏览")
        self.seg_browse.clicked.connect(self._browse_seg_file)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(0)
        self.slice_spin.valueChanged.connect(self.on_slice_changed)

        self.prev_slice_button = QPushButton("上一张")
        self.prev_slice_button.clicked.connect(self.previous_slice)
        self.next_slice_button = QPushButton("下一张")
        self.next_slice_button.clicked.connect(self.next_slice)

        self.slice_info_label = QLabel("切片：-")
        self.slice_info_label.setObjectName("SliceInfo")

        self.summary_label = QLabel("等待加载视微 OCT 数据")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("SummaryLabel")

        self.show_line_checkbox = QCheckBox("显示 B-scan 定位线")
        self.show_line_checkbox.setChecked(True)
        self.show_line_checkbox.toggled.connect(self.redraw_views)

        self.show_contours_checkbox = QCheckBox("显示分割曲线")
        self.show_contours_checkbox.setChecked(True)
        self.show_contours_checkbox.toggled.connect(self.redraw_views)

        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 5)
        self.line_width_spin.setValue(1)
        self.line_width_spin.valueChanged.connect(self.redraw_views)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setSingleStep(5)
        self.contrast_slider.valueChanged.connect(self.redraw_views)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-80, 80)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setSingleStep(2)
        self.brightness_slider.valueChanged.connect(self.redraw_views)

        self.reset_view_button = QPushButton("重置显示")
        self.reset_view_button.clicked.connect(self.reset_display_controls)

        self.volume_info_value = QLabel("-")
        self.fundus_info_value = QLabel("-")
        self.seg_info_value = QLabel("-")
        self.geometry_info_value = QLabel("-")

        self.legend_label = QLabel(
            "图例：红线=当前 B-scan 在 Fundus 上的位置；彩色线=分割层；"
            "亮度/对比度仅作用于右侧 B-scan 显示。"
        )
        self.legend_label.setWordWrap(True)
        self.legend_label.setObjectName("LegendLabel")

        self.metadata_edit = QPlainTextEdit()
        self.metadata_edit.setReadOnly(True)
        self.metadata_edit.setFont(QFont("Consolas", 10))

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
            QLineEdit, QSpinBox, QPlainTextEdit {
                background: #0F172A;
                color: #E5E7EB;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #334155;
                height: 6px;
                background: #0F172A;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38BDF8;
                border: 1px solid #7DD3FC;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #93C5FD;
            }
            QLabel#PathLabel, QLabel#SummaryLabel, QLabel#LegendLabel {
                background: #0F172A;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel#SliceInfo {
                background: #0369A1;
                color: white;
                border-radius: 12px;
                padding: 6px 12px;
                font-weight: 700;
            }
            QCheckBox { spacing: 8px; }
            QStatusBar { background: #0F172A; color: #CBD5E1; }
            """
        )

    def _build_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开目录", self)
        open_action.triggered.connect(self.open_path_dialog)
        toolbar.addAction(open_action)

        reload_action = QAction("重新加载", self)
        reload_action.triggered.connect(self.load_data)
        toolbar.addAction(reload_action)

        snapshot_action = QAction("保存截图", self)
        snapshot_action.triggered.connect(self.save_snapshot)
        toolbar.addAction(snapshot_action)

        toolbar.addSeparator()

        prev_slice_action = QAction("上一张", self)
        prev_slice_action.triggered.connect(self.previous_slice)
        toolbar.addAction(prev_slice_action)

        next_slice_action = QAction("下一张", self)
        next_slice_action.triggered.connect(self.next_slice)
        toolbar.addAction(next_slice_action)

        toolbar.addSeparator()

        reset_action = QAction("重置显示", self)
        reset_action.triggered.connect(self.reset_display_controls)
        toolbar.addAction(reset_action)

    def _make_info_row(self, title, value_widget):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #93C5FD; font-weight: 600;")
        value_widget.setWordWrap(True)
        layout.addWidget(title_label, 0)
        layout.addWidget(value_widget, 1)
        return row

    def _build_file_row(self, title, edit, button):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel(title))
        edit_row = QHBoxLayout()
        edit_row.setContentsMargins(0, 0, 0, 0)
        edit_row.addWidget(edit, 1)
        edit_row.addWidget(button)
        layout.addLayout(edit_row)
        return container

    def _build_layout(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)

        source_group = QGroupBox("数据源")
        source_layout = QVBoxLayout(source_group)
        source_layout.addWidget(self.open_button)
        source_layout.addWidget(self.load_button)
        source_layout.addWidget(self.snapshot_button)
        source_layout.addWidget(self.path_label)
        source_layout.addWidget(self._build_file_row("目录", self.input_dir_edit, self.input_dir_browse))
        source_layout.addWidget(self._build_file_row("B-scan", self.bscan_file_edit, self.bscan_browse))
        source_layout.addWidget(self._build_file_row("Fundus / CSSO", self.fundus_file_edit, self.fundus_browse))
        source_layout.addWidget(self._build_file_row("Segmentation", self.seg_file_edit, self.seg_browse))
        source_layout.addWidget(self.reset_override_button)

        navigation_group = QGroupBox("切片导航")
        navigation_layout = QGridLayout(navigation_group)
        navigation_layout.addWidget(self.slice_info_label, 0, 0, 1, 2)
        navigation_layout.addWidget(self.prev_slice_button, 1, 0)
        navigation_layout.addWidget(self.next_slice_button, 1, 1)
        navigation_layout.addWidget(self.slice_slider, 2, 0, 1, 2)
        navigation_layout.addWidget(self.slice_spin, 3, 0, 1, 2)

        display_group = QGroupBox("显示设置")
        display_layout = QFormLayout(display_group)
        display_layout.addRow(self.show_line_checkbox)
        display_layout.addRow(self.show_contours_checkbox)
        display_layout.addRow("曲线线宽", self.line_width_spin)
        display_layout.addRow("B-scan 对比度", self.contrast_slider)
        display_layout.addRow("B-scan 亮度", self.brightness_slider)
        display_layout.addRow(self.reset_view_button)

        info_group = QGroupBox("数据摘要")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self._make_info_row("体数据", self.volume_info_value))
        info_layout.addWidget(self._make_info_row("Fundus", self.fundus_info_value))
        info_layout.addWidget(self._make_info_row("分割", self.seg_info_value))
        info_layout.addWidget(self._make_info_row("几何", self.geometry_info_value))
        info_layout.addWidget(self.summary_label)
        info_layout.addWidget(self.legend_label)

        control_layout.addWidget(source_group)
        control_layout.addWidget(navigation_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(info_group)
        control_layout.addStretch(1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas, stretch=4)
        metadata_title = QLabel("元数据")
        metadata_title.setStyleSheet("font-weight: 700; color: #93C5FD; padding: 4px 0;")
        right_layout.addWidget(metadata_title)
        right_layout.addWidget(self.metadata_edit, stretch=2)

        splitter = QSplitter()
        splitter.addWidget(control_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1180])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self.previous_slice)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_slice)
        QShortcut(QKeySequence(Qt.Key_A), self, self.previous_slice)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_slice)
        QShortcut(QKeySequence(Qt.Key_Home), self, lambda: self.set_slice_index(0))
        QShortcut(QKeySequence(Qt.Key_End), self, self.jump_to_last_slice)
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_path_dialog)
        QShortcut(QKeySequence("Ctrl+R"), self, self.load_data)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_snapshot)

    def _restore_inputs(self):
        self.input_dir_edit.setText(self.args.input_dir or self.settings.value("last_dir", "", type=str))
        self.bscan_file_edit.setText(self.args.bscan_file or "")
        self.fundus_file_edit.setText(self.args.fundus_file or "")
        self.seg_file_edit.setText(self.args.seg_file or "")

    def _set_empty_state(self):
        self.volume = None
        self.fundus = None
        self.coordinates = []
        self.angles = []
        self.segmentation_surfaces = None
        self.segmentation_orientation = None
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(0)
        self.slice_spin.setValue(0)
        self.slice_spin.blockSignals(False)
        self.slice_info_label.setText("切片：-")
        self.volume_info_value.setText("-")
        self.fundus_info_value.setText("-")
        self.seg_info_value.setText("-")
        self.geometry_info_value.setText("-")
        self.summary_label.setText("等待加载视微 OCT 数据")
        self.metadata_edit.setPlainText("")
        self.canvas.clear_views()
        self.statusBar().showMessage("未加载数据。")

    def remember_path(self, path):
        remember_dir = path if Path(path).is_dir() else str(Path(path).parent)
        self.settings.setValue("last_dir", remember_dir)

    def get_last_open_dir(self):
        path = self.settings.value("last_dir", "", type=str)
        return path if path and Path(path).exists() else ""

    def open_path_dialog(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 DICOM 目录",
            self.input_dir_edit.text().strip() or self.get_last_open_dir(),
        )
        if directory:
            self.input_dir_edit.setText(directory)
            self.path_label.setText(directory)
            self.remember_path(directory)

    def _browse_file_into(self, target_edit, title):
        start_dir = self.input_dir_edit.text().strip() or self.get_last_open_dir() or "."
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            title,
            start_dir,
            "DICOM Files (*.dcm);;All Files (*)",
        )
        if filepath:
            target_edit.setText(filepath)
            self.remember_path(filepath)

    def _browse_bscan_file(self):
        self._browse_file_into(self.bscan_file_edit, "选择结构 B-scan DICOM")

    def _browse_fundus_file(self):
        self._browse_file_into(self.fundus_file_edit, "选择 Fundus/CSSO DICOM")

    def _browse_seg_file(self):
        self._browse_file_into(self.seg_file_edit, "选择 Segmentation DICOM")

    def _reset_file_overrides(self):
        self.bscan_file_edit.clear()
        self.fundus_file_edit.clear()
        self.seg_file_edit.clear()
        self.statusBar().showMessage("已清空手动文件覆盖。")

    def reset_display_controls(self):
        self.show_line_checkbox.setChecked(True)
        self.show_contours_checkbox.setChecked(True)
        self.line_width_spin.setValue(1)
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)
        self.redraw_views()

    def apply_bscan_window(self, image):
        image = normalize_to_uint8(image).astype(np.float32)
        contrast = self.contrast_slider.value() / 100.0
        brightness = float(self.brightness_slider.value())
        image = (image - 127.5) * contrast + 127.5 + brightness
        return np.clip(image, 0, 255).astype(np.uint8)

    def set_slice_index(self, index):
        if self.volume is None:
            return
        index = int(np.clip(index, 0, len(self.volume) - 1))
        self.slice_slider.blockSignals(True)
        self.slice_spin.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_spin.setValue(index)
        self.slice_slider.blockSignals(False)
        self.slice_spin.blockSignals(False)
        self.redraw_views()

    def previous_slice(self):
        self.set_slice_index(self.slice_slider.value() - 1)

    def next_slice(self):
        self.set_slice_index(self.slice_slider.value() + 1)

    def jump_to_last_slice(self):
        if self.volume is not None:
            self.set_slice_index(len(self.volume) - 1)

    def on_slice_changed(self, value):
        if self.volume is None:
            return
        self.set_slice_index(value)

    def load_data(self):
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "缺少目录", "请先选择包含 DICOM 的目录。")
            return

        try:
            (
                self.volume,
                self.fundus,
                self.coordinates,
                self.angles,
                self.segmentation_surfaces,
                self.segmentation_orientation,
                self.bscan_file,
                self.fundus_file,
                self.seg_file,
            ) = load_shiwei_data(
                input_dir=input_dir,
                bscan_file=self.bscan_file_edit.text().strip() or None,
                fundus_file=self.fundus_file_edit.text().strip() or None,
                seg_file=self.seg_file_edit.text().strip() or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            self.statusBar().showMessage("加载失败。")
            return

        self.path_label.setText(input_dir)
        self.remember_path(input_dir)

        max_index = max(0, len(self.volume) - 1)
        self.slice_slider.blockSignals(True)
        self.slice_spin.blockSignals(True)
        self.slice_slider.setMaximum(max_index)
        self.slice_spin.setMaximum(max_index)
        self.slice_slider.setValue(0)
        self.slice_spin.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_spin.blockSignals(False)

        self.volume_info_value.setText(str(tuple(self.volume.shape)))
        self.fundus_info_value.setText(str(tuple(self.fundus.shape)))
        seg_shape = "-" if self.segmentation_surfaces is None else str(tuple(self.segmentation_surfaces.shape))
        self.seg_info_value.setText(seg_shape)

        if self.coordinates:
            first = np.asarray(self.coordinates[0], dtype=float)
            last = np.asarray(self.coordinates[-1], dtype=float)
            self.geometry_info_value.setText(
                f"first={np.round(first, 2).tolist()} | last={np.round(last, 2).tolist()}"
            )
        else:
            self.geometry_info_value.setText("-")

        self.summary_label.setText(
            f"共 {len(self.volume)} 张 B-scan；"
            f"分割层数：{0 if self.segmentation_surfaces is None else len(self.segmentation_surfaces)}；"
            f"当前结构文件：{Path(self.bscan_file).name}"
        )
        self.metadata_edit.setPlainText(self.build_metadata_text())
        self.redraw_views()
        self.statusBar().showMessage(f"加载完成：{Path(self.bscan_file).name}")

    def build_metadata_text(self):
        lines = [
            f"Input dir: {self.input_dir_edit.text().strip()}",
            f"B-scan file: {self.bscan_file}",
            f"Fundus file: {self.fundus_file}",
            f"Seg file: {self.seg_file}",
            f"Volume shape: {None if self.volume is None else tuple(self.volume.shape)}",
            f"Fundus shape: {None if self.fundus is None else tuple(self.fundus.shape)}",
            f"Segmentation shape: {None if self.segmentation_surfaces is None else tuple(self.segmentation_surfaces.shape)}",
        ]
        if self.coordinates:
            lines.append(f"First coordinates: {np.asarray(self.coordinates[0], dtype=float).tolist()}")
            lines.append(f"Last coordinates: {np.asarray(self.coordinates[-1], dtype=float).tolist()}")
        return "\n".join(lines)

    def redraw_views(self):
        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()
        self.canvas._style_axes()

        if self.volume is None or self.fundus is None:
            self.canvas.ax_fundus.text(0.5, 0.5, "No data", ha="center", va="center", color="#E5E7EB")
            self.canvas.ax_bscan.text(0.5, 0.5, "No data", ha="center", va="center", color="#E5E7EB")
            self.canvas.draw_idle()
            return

        slice_idx = int(self.slice_slider.value())
        bscan_slice = self.apply_bscan_window(self.volume[slice_idx])
        display_fundus = ensure_rgb(self.fundus)

        if self.fundus.ndim == 2:
            self.canvas.ax_fundus.imshow(display_fundus[..., 0], cmap="gray", origin="upper")
        else:
            self.canvas.ax_fundus.imshow(display_fundus, origin="upper")
        self.canvas.ax_fundus.set_title("Fundus / CSSO", color="#E5E7EB")
        self.canvas.ax_fundus.axis("off")

        coords = np.asarray(self.coordinates[slice_idx], dtype=float)
        if self.show_line_checkbox.isChecked() and coords.size == 4 and np.any(coords):
            self.canvas.ax_fundus.plot(
                [coords[0], coords[2]],
                [coords[1], coords[3]],
                color="#FF4D6D",
                linewidth=2.0,
            )

        self.canvas.ax_bscan.imshow(bscan_slice, cmap="gray", origin="upper")
        self.canvas.ax_bscan.set_title(
            f"B-scan {slice_idx + 1}/{len(self.volume)} · angle={self.angles[slice_idx]:.2f}°",
            color="#E5E7EB",
        )
        self.canvas.ax_bscan.axis("off")

        if self.show_contours_checkbox.isChecked() and self.segmentation_surfaces is not None:
            curves = get_segmentation_curves(
                self.segmentation_surfaces,
                self.coordinates,
                slice_idx,
                bscan_slice.shape[1],
                orientation=self.segmentation_orientation,
            )
            for layer_index, curve in enumerate(curves):
                self.canvas.ax_bscan.plot(
                    np.arange(len(curve)),
                    curve,
                    color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
                    linewidth=self.line_width_spin.value(),
                    alpha=0.95,
                )

        self.slice_info_label.setText(f"切片：{slice_idx + 1} / {len(self.volume)}")
        self.canvas.draw_idle()

    def save_snapshot(self):
        if self.volume is None:
            QMessageBox.information(self, "暂无数据", "请先加载一组 OCT 数据。")
            return

        default_dir = self.get_last_open_dir() or self.input_dir_edit.text().strip() or "."
        default_name = "shiwei_viewer_snapshot.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "保存截图",
            str(Path(default_dir) / default_name),
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
        )
        if not filepath:
            return
        self.canvas.figure.savefig(filepath, dpi=160, facecolor=self.canvas.figure.get_facecolor())
        self.statusBar().showMessage(f"截图已保存：{filepath}")
        self.remember_path(filepath)


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Shiwei Qt Viewer")
    window = ShiweiViewerWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
