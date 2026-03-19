import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


@dataclass
class VolumeViewModel:
    label: str
    volume_id: str
    laterality: str
    slices: list
    contours: Optional[dict]
    fundus: np.ndarray
    scan_segments: list
    metadata_text: str
    source_kind: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qt viewer for OCT E2E/FDA files.",
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Optional E2E/FDA file to open at startup.",
    )
    return parser.parse_args()


def normalize_to_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) * (255.0 / (max_value - min_value))
    return np.clip(image, 0, 255).astype(np.uint8)


def to_display_image(image):
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_rgb(image):
    image = to_display_image(image)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return image


def make_projection_fundus(volume_slices):
    volume_array = np.asarray(volume_slices)
    if volume_array.ndim != 3:
        return np.zeros((512, 512), dtype=np.uint8)
    projection = np.mean(volume_array, axis=1)
    return normalize_to_uint8(projection)


def partition_bscan_metadata(bscan_data, slice_counts):
    groups = []
    cursor = 0
    for count in slice_counts:
        groups.append(bscan_data[cursor : cursor + count])
        cursor += count
    return groups


def apply_affine_transform(points, transform_values):
    if not transform_values or len(transform_values) < 6:
        return None

    a, b, c, d, e, f = transform_values[:6]

    def candidate_standard(point):
        x_pos, y_pos = point
        return a * x_pos + b * y_pos + c, d * x_pos + e * y_pos + f

    def candidate_alt1(point):
        x_pos, y_pos = point
        return a * x_pos + b * y_pos + e, c * x_pos + d * y_pos + f

    def candidate_alt2(point):
        x_pos, y_pos = point
        return a * x_pos + c * y_pos + e, b * x_pos + d * y_pos + f

    return [
        [candidate(point) for point in points]
        for candidate in [candidate_standard, candidate_alt1, candidate_alt2]
    ]


def map_e2e_point_to_fundus(point, width, height, field_size_degrees, flip_x=False, flip_y=False):
    x_pos, y_pos = point
    if flip_x:
        x_pos = -x_pos
    if flip_y:
        y_pos = -y_pos

    x_scale = width / field_size_degrees
    y_scale = height / field_size_degrees
    x_pixel = width / 2.0 + x_pos * x_scale
    y_pixel = height / 2.0 + y_pos * y_scale
    return x_pixel, y_pixel


def score_projected_points(points, width, height):
    if not points:
        return -1.0

    inside = 0
    outside_penalty = 0.0
    for x_pos, y_pos in points:
        if -0.25 * width <= x_pos <= 1.25 * width and -0.25 * height <= y_pos <= 1.25 * height:
            inside += 1
        else:
            dx = max(-0.25 * width - x_pos, 0.0, x_pos - 1.25 * width)
            dy = max(-0.25 * height - y_pos, 0.0, y_pos - 1.25 * height)
            outside_penalty += dx + dy

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    spread = max(xs) - min(xs) + max(ys) - min(ys)
    spread_score = min(spread / max(width + height, 1), 2.0)
    penalty_score = outside_penalty / max(width + height, 1)
    return inside + spread_score - penalty_score


def points_to_segments(points):
    return [
        (points[index], points[index + 1])
        for index in range(0, len(points), 2)
    ]


def project_e2e_segments(group, localizer_entry, fundus_shape):
    raw_segments = []
    for item in group:
        raw_segments.append(
            (
                (float(item.get("posX1", 0.0)), float(item.get("posY1", 0.0))),
                (float(item.get("posX2", 0.0)), float(item.get("posY2", 0.0))),
            )
        )

    if not raw_segments:
        return []

    fundus_height, fundus_width = fundus_shape[:2]
    points = [point for segment in raw_segments for point in segment]
    transform_values = (localizer_entry or {}).get("transform")
    field_size_degrees = 30.0

    candidate_sets = []
    for flip_x in (False, True):
        for flip_y in (False, True):
            mapped_points = [
                map_e2e_point_to_fundus(
                    point=point,
                    width=fundus_width,
                    height=fundus_height,
                    field_size_degrees=field_size_degrees,
                    flip_x=flip_x,
                    flip_y=flip_y,
                )
                for point in points
            ]
            candidate_sets.append(mapped_points)

            affine_variants = apply_affine_transform(mapped_points, transform_values)
            if affine_variants:
                candidate_sets.extend(affine_variants)

    best_points = None
    best_score = float("-inf")
    for candidate in candidate_sets:
        score = score_projected_points(candidate, fundus_width, fundus_height)
        if score > best_score:
            best_score = score
            best_points = candidate

    if best_points is None:
        return build_parallel_segments(len(group), fundus_shape)

    return points_to_segments(best_points)


def build_parallel_segments(num_slices, fundus_shape):
    height, width = fundus_shape[:2]
    if num_slices <= 0:
        return []

    x_min = width * 0.18
    x_max = width * 0.82
    y_min = height * 0.18
    y_max = height * 0.82

    segments = []
    if num_slices == 1:
        x_pos = (x_min + x_max) / 2.0
        segments.append(((x_pos, y_min), (x_pos, y_max)))
        return segments

    for index in range(num_slices):
        ratio = index / (num_slices - 1)
        x_pos = x_min + ratio * (x_max - x_min)
        segments.append(((x_pos, y_min), (x_pos, y_max)))
    return segments


def compute_scan_bounds(segments):
    if not segments:
        return None
    xs = [point[0] for segment in segments for point in segment]
    ys = [point[1] for segment in segments for point in segment]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def format_metadata_text(path, volume, source_kind, fundus_shape, segment_mode):
    summary = {
        "file": str(path),
        "source": source_kind,
        "volume_id": volume.volume_id,
        "laterality": volume.laterality,
        "num_slices": volume.num_slices,
        "slice_shape": list(np.asarray(volume.volume[0]).shape),
        "pixel_spacing_mm": volume.pixel_spacing,
        "fundus_shape": list(fundus_shape[:2]) if fundus_shape is not None else None,
        "overlay_mode": segment_mode,
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, default=str)


def select_matching_fundus(volume, fundus_images, index):
    if not fundus_images:
        return None, "projection-fallback"

    fundus_by_id = {
        getattr(image, "image_id", None): image
        for image in fundus_images
        if getattr(image, "image_id", None)
    }
    if volume.volume_id in fundus_by_id:
        return fundus_by_id[volume.volume_id], "matched-by-id"

    if volume.laterality:
        same_laterality = [
            image for image in fundus_images if getattr(image, "laterality", None) == volume.laterality
        ]
        if len(same_laterality) == 1:
            return same_laterality[0], "matched-by-laterality"

    return fundus_images[min(index, len(fundus_images) - 1)], "matched-by-index-fallback"


def load_e2e_file(filepath):
    from oct_converter.readers.e2e import E2E

    reader = E2E(filepath)
    volumes = reader.read_oct_volume()
    fundus_images = reader.read_fundus_image()
    metadata = reader.read_all_metadata()

    if not volumes:
        raise RuntimeError("E2E 文件中没有读取到 OCT 体数据。")

    slice_counts = [volume.num_slices for volume in volumes]
    bscan_groups = partition_bscan_metadata(metadata.get("bscan_data", []), slice_counts)
    localizers = metadata.get("localizer", [])

    models = []
    for index, volume in enumerate(volumes):
        matched_fundus, fundus_match_mode = select_matching_fundus(volume, fundus_images, index)
        if matched_fundus is not None:
            fundus = matched_fundus.image
        else:
            fundus = make_projection_fundus(volume.volume)
            fundus_match_mode = "projection-fallback"

        group = bscan_groups[index] if index < len(bscan_groups) else []
        localizer_entry = localizers[min(index, len(localizers) - 1)] if localizers else None
        if group:
            scan_segments = project_e2e_segments(group, localizer_entry, fundus.shape)
            segment_mode = f"e2e-metadata/{fundus_match_mode}"
        else:
            scan_segments = build_parallel_segments(volume.num_slices, fundus.shape)
            segment_mode = f"fallback-parallel/{fundus_match_mode}"

        label = volume.volume_id or f"E2E Volume {index + 1}"
        if volume.laterality:
            label = f"{label} ({volume.laterality})"

        models.append(
            VolumeViewModel(
                label=label,
                volume_id=volume.volume_id or f"volume_{index + 1}",
                laterality=volume.laterality or "",
                slices=list(volume.volume),
                contours=volume.contours,
                fundus=to_display_image(fundus),
                scan_segments=scan_segments,
                metadata_text=format_metadata_text(filepath, volume, "E2E", fundus.shape, segment_mode),
                source_kind="E2E",
            )
        )

    return models


def load_fda_file(filepath):
    from oct_converter.readers.fda import FDA

    reader = FDA(filepath)
    volume = reader.read_oct_volume()
    if volume is None or not volume.volume:
        raise RuntimeError("FDA 文件中没有读取到 OCT 体数据。")

    fundus_image = reader.read_fundus_image()
    if fundus_image is None:
        fundus_image = reader.read_fundus_image_gray_scale()

    if fundus_image is not None:
        fundus = fundus_image.image
    else:
        fundus = make_projection_fundus(volume.volume)

    scan_segments = build_parallel_segments(volume.num_slices, fundus.shape)
    label = Path(filepath).stem
    if volume.laterality:
        label = f"{label} ({volume.laterality})"

    return [
        VolumeViewModel(
            label=label,
            volume_id=volume.volume_id or Path(filepath).stem,
            laterality=volume.laterality or "",
            slices=list(volume.volume),
            contours=volume.contours,
            fundus=to_display_image(fundus),
            scan_segments=scan_segments,
            metadata_text=format_metadata_text(filepath, volume, "FDA", fundus.shape, "parallel-approximation"),
            source_kind="FDA",
        )
    ]


def load_models(filepath):
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".e2e":
        return load_e2e_file(path)
    if suffix == ".fda":
        return load_fda_file(path)
    raise ValueError("仅支持导入 .E2E 和 .fda 文件。")


class FundusBscanCanvas(FigureCanvas):
    def __init__(self, parent=None):
        figure = Figure(figsize=(10, 5), tight_layout=True)
        self.ax_fundus = figure.add_subplot(1, 2, 1)
        self.ax_bscan = figure.add_subplot(1, 2, 2)
        super().__init__(figure)
        self.setParent(parent)


class OCTViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCT E2E/FDA Qt Viewer")
        self.resize(1360, 860)
        self.settings = QSettings("OpenAI", "OCTE2EFDAViewer")

        self.filepath = None
        self.models = []
        self.current_volume_index = 0
        self.current_slice_index = 0

        self.canvas = FundusBscanCanvas(self)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        self.open_button = QPushButton("打开文件")
        self.open_button.clicked.connect(self.open_file_dialog)

        self.file_label = QLabel("未打开文件")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.volume_combo = QComboBox()
        self.volume_combo.currentIndexChanged.connect(self.on_volume_changed)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(0)
        self.slice_spin.valueChanged.connect(self.on_slice_changed)

        self.slice_info_label = QLabel("Slice: -")
        self.summary_label = QLabel("等待导入 E2E/FDA 文件")
        self.summary_label.setWordWrap(True)

        self.metadata_edit = QPlainTextEdit()
        self.metadata_edit.setReadOnly(True)

        self._build_toolbar()
        self._build_layout()
        self._set_empty_state()

    def get_last_open_dir(self):
        last_file = self.settings.value("last_file", "", type=str)
        if last_file and Path(last_file).exists():
            return str(Path(last_file).parent)

        last_dir = self.settings.value("last_dir", "", type=str)
        if last_dir and Path(last_dir).exists():
            return last_dir

        return ""

    def get_last_file(self):
        last_file = self.settings.value("last_file", "", type=str)
        if last_file and Path(last_file).exists():
            return last_file
        return None

    def remember_file(self, filepath):
        filepath = str(Path(filepath).resolve())
        self.settings.setValue("last_file", filepath)
        self.settings.setValue("last_dir", str(Path(filepath).parent))

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("打开", self)
        open_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(open_action)

    def _build_layout(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(QLabel("Volume"))
        control_layout.addWidget(self.volume_combo)
        control_layout.addWidget(QLabel("Slice"))
        control_layout.addWidget(self.slice_slider)
        control_layout.addWidget(self.slice_spin)
        control_layout.addWidget(self.slice_info_label)
        control_layout.addWidget(self.summary_label)
        control_layout.addStretch(1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.canvas, stretch=4)
        right_layout.addWidget(QLabel("Metadata"))
        right_layout.addWidget(self.metadata_edit, stretch=2)

        splitter = QSplitter()
        splitter.addWidget(control_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _set_empty_state(self):
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        self.volume_combo.blockSignals(False)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(0)
        self.slice_spin.setValue(0)
        self.slice_spin.blockSignals(False)
        self.metadata_edit.setPlainText("")
        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()
        self.canvas.ax_fundus.set_title("Fundus")
        self.canvas.ax_bscan.set_title("B-scan")
        self.canvas.draw_idle()

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 E2E/FDA 文件",
            self.get_last_open_dir(),
            "OCT Files (*.E2E *.e2e *.FDA *.fda)",
        )
        if filename:
            self.load_file(filename)

    def load_file(self, filepath):
        try:
            models = load_models(filepath)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return

        self.filepath = str(filepath)
        self.models = models
        self.current_volume_index = 0
        self.current_slice_index = 0
        self.remember_file(filepath)

        self.file_label.setText(self.filepath)
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for model in self.models:
            self.volume_combo.addItem(model.label)
        self.volume_combo.blockSignals(False)
        self.volume_combo.setCurrentIndex(0)
        self.refresh_volume()

    def current_model(self):
        if not self.models:
            return None
        return self.models[self.current_volume_index]

    def refresh_volume(self):
        model = self.current_model()
        if model is None:
            self._set_empty_state()
            return

        max_index = max(0, len(model.slices) - 1)
        self.current_slice_index = min(self.current_slice_index, max_index)

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(max_index)
        self.slice_slider.setValue(self.current_slice_index)
        self.slice_slider.blockSignals(False)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(max_index)
        self.slice_spin.setValue(self.current_slice_index)
        self.slice_spin.blockSignals(False)

        self.metadata_edit.setPlainText(model.metadata_text)
        self.summary_label.setText(
            f"{model.source_kind} | {model.volume_id} | {len(model.slices)} slices | Laterality: {model.laterality or '-'}"
        )
        self.redraw_views()

    def redraw_views(self):
        model = self.current_model()
        if model is None:
            return

        self.canvas.ax_fundus.clear()
        self.canvas.ax_bscan.clear()

        fundus = model.fundus
        if fundus.ndim == 2:
            self.canvas.ax_fundus.imshow(fundus, cmap="gray", origin="upper")
        else:
            self.canvas.ax_fundus.imshow(fundus, origin="upper")
        self.canvas.ax_fundus.axis("off")
        self.canvas.ax_fundus.set_title(f"Fundus / {model.source_kind}")

        bounds = compute_scan_bounds(model.scan_segments)
        if bounds is not None:
            rect = Rectangle(
                (bounds[0], bounds[1]),
                max(bounds[2], 1.0),
                max(bounds[3], 1.0),
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                alpha=0.7,
            )
            self.canvas.ax_fundus.add_patch(rect)

        for index, segment in enumerate(model.scan_segments):
            color = "#66ff66" if index == self.current_slice_index else "#ffd966"
            alpha = 0.95 if index == self.current_slice_index else 0.25
            linewidth = 2.8 if index == self.current_slice_index else 1.1
            (start_x, start_y), (end_x, end_y) = segment
            self.canvas.ax_fundus.plot(
                [start_x, end_x],
                [start_y, end_y],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

        bscan = to_display_image(model.slices[self.current_slice_index])
        self.canvas.ax_bscan.imshow(bscan, cmap="gray", aspect="auto", origin="upper")
        self.canvas.ax_bscan.axis("off")
        self.canvas.ax_bscan.set_title(
            f"B-scan {self.current_slice_index + 1}/{len(model.slices)}"
        )

        contour_colors = [
            "#00ffff",
            "#ff66cc",
            "#66ff66",
            "#ff9933",
            "#66b3ff",
            "#ffffff",
        ]
        if model.contours:
            color_index = 0
            for contour_name, values in model.contours.items():
                if self.current_slice_index >= len(values):
                    continue
                contour = values[self.current_slice_index]
                if contour is None:
                    continue
                contour = np.asarray(contour)
                if contour.size == 0 or np.isnan(contour).all():
                    continue
                x_coords = np.arange(contour.shape[0])
                self.canvas.ax_bscan.plot(
                    x_coords,
                    contour,
                    color=contour_colors[color_index % len(contour_colors)],
                    linewidth=1.1,
                    alpha=0.9,
                )
                color_index += 1

        self.slice_info_label.setText(
            f"Slice: {self.current_slice_index + 1} / {len(model.slices)}"
        )
        self.canvas.draw_idle()

    def set_slice_index(self, index):
        model = self.current_model()
        if model is None:
            return

        index = max(0, min(index, len(model.slices) - 1))
        if index == self.current_slice_index and self.slice_slider.value() == index and self.slice_spin.value() == index:
            self.redraw_views()
            return

        self.current_slice_index = index
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)
        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(index)
        self.slice_spin.blockSignals(False)
        self.redraw_views()

    def on_volume_changed(self, index):
        if index < 0 or index >= len(self.models):
            return
        self.current_volume_index = index
        self.current_slice_index = 0
        self.refresh_volume()

    def on_slice_changed(self, value):
        self.set_slice_index(int(value))

    def on_canvas_click(self, event):
        model = self.current_model()
        if model is None or event.inaxes != self.canvas.ax_fundus:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not model.scan_segments:
            return

        target_x = float(event.xdata)
        target_y = float(event.ydata)
        nearest_index = 0
        nearest_distance = float("inf")
        for index, segment in enumerate(model.scan_segments):
            (start_x, start_y), (end_x, end_y) = segment
            center_x = (start_x + end_x) / 2.0
            center_y = (start_y + end_y) / 2.0
            distance = (target_x - center_x) ** 2 + (target_y - center_y) ** 2
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        self.set_slice_index(nearest_index)


def main():
    args = parse_args()
    application = QApplication(sys.argv)
    window = OCTViewerWindow()
    window.show()
    if args.filepath:
        window.load_file(args.filepath)
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
