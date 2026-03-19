from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pydicom

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
    QCheckBox,
    QComboBox,
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
)

from scripts.zeiss_dicom import ZEISSDicom


@dataclass
class FundusCandidate:
    image: np.ndarray
    laterality: str
    source_file: str
    source_kind: str
    width: int
    height: int
    score: float


@dataclass
class VolumeViewModel:
    label: str
    source_file: str
    volume_id: str
    laterality: str
    slices: np.ndarray
    fundus: np.ndarray
    scan_segments: list
    overlay_bounds: tuple[float, float, float, float] | None
    overlay_spokes: list
    metadata_text: str


@dataclass
class ExamViewModel:
    label: str
    exam_id: str
    patient_id: str
    exam_path: str
    volumes: list[VolumeViewModel]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qt viewer for Zeiss OCT DICOM exports.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional Zeiss export root / exam folder / DCM / DICOMDIR path.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).split("\x00", 1)[0]
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
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


def to_display_image(image: np.ndarray) -> np.ndarray:
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_gray_image(image: np.ndarray) -> np.ndarray:
    image = to_display_image(image)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_gray_volume(volume_array: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_array)
    if volume.ndim == 2:
        return volume[np.newaxis, ...]
    if volume.ndim == 3:
        return volume
    if volume.ndim == 4 and volume.shape[-1] >= 1:
        return volume[..., 0]
    raise ValueError(f"Unsupported volume shape: {volume.shape}")


def safe_dcmread(path: Path) -> pydicom.dataset.FileDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydicom.dcmread(path, force=True)


def classify_dicom(ds: pydicom.dataset.FileDataset) -> str:
    has_pixel_data = "PixelData" in ds
    number_of_frames = None
    try:
        number_of_frames = int(clean_text(getattr(ds, "NumberOfFrames", "")) or 0)
    except ValueError:
        number_of_frames = None
    rows = getattr(ds, "Rows", None)
    columns = getattr(ds, "Columns", None)

    if not has_pixel_data:
        return "non_image_dicom"
    if number_of_frames is not None and number_of_frames >= 32:
        return "oct_volume"
    if number_of_frames is not None and number_of_frames > 1:
        return "multi_frame_image"
    if rows is not None and columns is not None:
        return "single_frame_image"
    return "unknown_image"


def resolve_exam_dirs(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.name.upper() == "DICOMDIR" or path.suffix.upper() == ".DCM":
            return [path.parent]
        raise ValueError("请选择蔡司导出目录、DICOMDIR 或 DCM 文件。")

    if (path / "DataFiles").exists():
        return sorted(child for child in (path / "DataFiles").iterdir() if child.is_dir())

    if path.name.lower() == "datafiles":
        return sorted(child for child in path.iterdir() if child.is_dir())

    if (path / "DICOMDIR").exists() or any(path.glob("*.DCM")):
        return [path]

    exam_dirs = [child for child in path.iterdir() if child.is_dir() and (child / "DICOMDIR").exists()]
    if exam_dirs:
        return sorted(exam_dirs)

    raise ValueError("未识别到蔡司导出结构。")


def is_bscan_like(volume: np.ndarray) -> bool:
    gray_volume = to_gray_volume(volume)
    height, width = gray_volume.shape[1:]
    ratio = max(width, height) / max(min(width, height), 1)
    return ratio >= 1.35


def make_projection_fundus(volume_slices: np.ndarray) -> np.ndarray:
    volume_array = to_gray_volume(volume_slices)
    projection = np.mean(volume_array, axis=1)
    return normalize_to_uint8(projection)


def build_raster_segments(num_slices: int, fundus_shape: tuple[int, int]) -> tuple[list, tuple[float, float, float, float] | None]:
    height, width = fundus_shape[:2]
    if num_slices <= 0:
        return [], None

    box_w = width * 0.66
    box_h = height * 0.66
    x0 = (width - box_w) / 2.0
    y0 = (height - box_h) / 2.0
    x1 = x0 + box_w
    y1 = y0 + box_h

    if num_slices == 1:
        y_values = [(y0 + y1) / 2.0]
    else:
        y_values = np.linspace(y0, y1, num_slices)

    segments = [((x0, float(y_val)), (x1, float(y_val))) for y_val in y_values]
    return segments, (x0, y0, box_w, box_h)


def build_overlay_spokes(bounds: tuple[float, float, float, float] | None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if bounds is None:
        return []

    x0, y0, width, height = bounds
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    points = [
        (x0, y0),
        (x0 + width / 2.0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height / 2.0),
        (x0 + width, y0 + height),
        (x0 + width / 2.0, y0 + height),
        (x0, y0 + height),
        (x0, y0 + height / 2.0),
    ]
    return [((cx, cy), point) for point in points]


def estimate_bscan_guides(bscan_image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    gray = to_gray_image(bscan_image)
    height, width = gray.shape
    if height < 32 or width < 32:
        return None, None

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    gradient = np.diff(blurred.astype(np.float32), axis=0)

    upper_limit = max(8, int(height * 0.55))
    ilm = np.argmax(gradient[:upper_limit, :], axis=0).astype(np.float32)

    second = np.full(width, np.nan, dtype=np.float32)
    for x_pos in range(width):
        start = int(min(height - 2, ilm[x_pos] + max(5, int(height * 0.06))))
        end = int(min(height - 1, start + max(12, int(height * 0.28))))
        if end <= start:
            continue
        local_gradient = gradient[start:end, x_pos]
        if local_gradient.size == 0:
            continue
        second[x_pos] = start + int(np.argmin(local_gradient))

    kernel = np.ones(9, dtype=np.float32) / 9.0
    ilm = np.convolve(ilm, kernel, mode="same")
    valid_second = np.nan_to_num(second, nan=np.nanmean(second) if np.isfinite(np.nanmean(second)) else 0.0)
    valid_second = np.convolve(valid_second, kernel, mode="same")
    return ilm, valid_second


def score_fundus_candidate(candidate: FundusCandidate, laterality: str) -> float:
    score = candidate.score
    if candidate.laterality and laterality and candidate.laterality == laterality:
        score += 1_000_000.0
    if candidate.source_kind == "multiframe_localizer":
        score += 50_000.0
    return score


def choose_best_fundus(candidates: list[FundusCandidate], laterality: str) -> FundusCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: score_fundus_candidate(candidate, laterality))


def build_metadata_text(exam_dir: Path, patient_id: str, source_file: Path, volume: np.ndarray, fundus_candidate: FundusCandidate | None) -> str:
    gray_volume = to_gray_volume(volume)
    metadata = {
        "exam_dir": str(exam_dir),
        "patient_id": patient_id,
        "source_file": str(source_file),
        "num_slices": int(gray_volume.shape[0]),
        "slice_shape": [int(gray_volume.shape[1]), int(gray_volume.shape[2])],
        "fundus_source": fundus_candidate.source_file if fundus_candidate else "projection-fallback",
        "fundus_kind": fundus_candidate.source_kind if fundus_candidate else "projection-fallback",
        "overlay_note": "Fundus overlay is approximate for Zeiss DICOM export; not for clinical measurement.",
        "bscan_guides": "heuristic top/lower layer guides",
    }
    return json.dumps(metadata, indent=2, ensure_ascii=False)


def load_exam(exam_dir: Path) -> ExamViewModel:
    dcm_files = sorted(exam_dir.glob("*.DCM"))
    if not dcm_files:
        raise RuntimeError(f"在 {exam_dir} 下没有找到 DCM 文件。")

    dicomdir_path = exam_dir / "DICOMDIR"
    patient_id = ""
    if dicomdir_path.exists():
        ds = safe_dcmread(dicomdir_path)
        for record in getattr(ds, "DirectoryRecordSequence", []):
            patient_id = clean_text(getattr(record, "PatientID", ""))
            if patient_id:
                break

    fundus_candidates: list[FundusCandidate] = []
    pending_volumes: list[dict[str, Any]] = []

    for dcm_file in dcm_files:
        ds = safe_dcmread(dcm_file)
        classification = classify_dicom(ds)
        if classification == "non_image_dicom":
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = ZEISSDicom(dcm_file)
            oct_volumes, fundus_images = reader.read_data()

        file_laterality = clean_text(getattr(ds, "Laterality", ""))

        for image in fundus_images:
            image_array = to_display_image(image.image)
            height, width = image_array.shape[:2]
            ratio = min(height, width) / max(height, width)
            score = float(height * width) * (0.5 + 0.5 * ratio)
            fundus_candidates.append(
                FundusCandidate(
                    image=image_array,
                    laterality=clean_text(getattr(image, "laterality", "")) or file_laterality,
                    source_file=dcm_file.name,
                    source_kind="fundus",
                    width=width,
                    height=height,
                    score=score,
                )
            )

        for index, volume in enumerate(oct_volumes):
            gray_volume = to_gray_volume(volume.volume)
            height, width = gray_volume.shape[1:]

            if is_bscan_like(gray_volume):
                pending_volumes.append(
                    {
                        "source_file": dcm_file,
                        "volume_index": index,
                        "laterality": clean_text(getattr(volume, "laterality", "")) or file_laterality,
                        "volume": gray_volume,
                    }
                )
            elif gray_volume.shape[0] > 0:
                localizer_image = normalize_to_uint8(gray_volume[0])
                local_h, local_w = localizer_image.shape[:2]
                ratio = min(local_h, local_w) / max(local_h, local_w)
                score = float(local_h * local_w) * (0.5 + 0.5 * ratio)
                fundus_candidates.append(
                    FundusCandidate(
                        image=localizer_image,
                        laterality=clean_text(getattr(volume, "laterality", "")) or file_laterality,
                        source_file=dcm_file.name,
                        source_kind="multiframe_localizer",
                        width=local_w,
                        height=local_h,
                        score=score,
                    )
                )

    if not pending_volumes:
        raise RuntimeError(f"在 {exam_dir} 下没有找到可显示的 Zeiss B-scan 体数据。")

    models: list[VolumeViewModel] = []
    for item in pending_volumes:
        best_fundus = choose_best_fundus(fundus_candidates, item["laterality"])
        fundus_image = best_fundus.image if best_fundus is not None else make_projection_fundus(item["volume"])
        segments, bounds = build_raster_segments(item["volume"].shape[0], fundus_image.shape)
        spokes = build_overlay_spokes(bounds)
        label = item["source_file"].stem
        if item["laterality"]:
            label = f"{label} ({item['laterality']})"

        metadata_text = build_metadata_text(
            exam_dir=exam_dir,
            patient_id=patient_id,
            source_file=item["source_file"],
            volume=item["volume"],
            fundus_candidate=best_fundus,
        )

        models.append(
            VolumeViewModel(
                label=label,
                source_file=item["source_file"].name,
                volume_id=item["source_file"].stem,
                laterality=item["laterality"],
                slices=item["volume"],
                fundus=to_display_image(fundus_image),
                scan_segments=segments,
                overlay_bounds=bounds,
                overlay_spokes=spokes,
                metadata_text=metadata_text,
            )
        )

    return ExamViewModel(
        label=f"{exam_dir.name} ({patient_id or 'Unknown'})",
        exam_id=exam_dir.name,
        patient_id=patient_id,
        exam_path=str(exam_dir),
        volumes=models,
    )


def load_exams(path: str | Path) -> list[ExamViewModel]:
    exam_dirs = resolve_exam_dirs(path)
    return [load_exam(exam_dir) for exam_dir in exam_dirs]


class ViewerCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(10, 5), tight_layout=True)
        super().__init__(self.figure)
        self.ax_fundus = self.figure.add_subplot(1, 2, 1)
        self.ax_bscan = self.figure.add_subplot(1, 2, 2)


class ZeissViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zeiss OCT Viewer")
        self.resize(1400, 860)

        self.settings = QSettings("OpenAI", "ZeissOCTViewer")
        self.source_path: str | None = None
        self.exams: list[ExamViewModel] = []
        self.current_exam_index = 0
        self.current_volume_index = 0
        self.current_slice_index = 0

        self.canvas = ViewerCanvas()
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)

        self.open_button = QPushButton("打开蔡司目录")
        self.open_button.clicked.connect(self.open_path_dialog)

        self.path_label = QLabel("未打开目录")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.exam_combo = QComboBox()
        self.exam_combo.currentIndexChanged.connect(self.on_exam_changed)

        self.volume_combo = QComboBox()
        self.volume_combo.currentIndexChanged.connect(self.on_volume_changed)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(0)
        self.slice_spin.valueChanged.connect(self.on_slice_changed)

        self.slice_info_label = QLabel("Slice: -")
        self.summary_label = QLabel("等待导入蔡司 OCT 数据")
        self.summary_label.setWordWrap(True)

        self.show_spokes_checkbox = QCheckBox("显示辅助星形线")
        self.show_spokes_checkbox.setChecked(True)
        self.show_spokes_checkbox.toggled.connect(self.redraw_views)

        self.show_guides_checkbox = QCheckBox("显示 B-scan 引导线")
        self.show_guides_checkbox.setChecked(True)
        self.show_guides_checkbox.toggled.connect(self.redraw_views)

        self.metadata_edit = QPlainTextEdit()
        self.metadata_edit.setReadOnly(True)

        self._build_toolbar()
        self._build_layout()
        self._set_empty_state()

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("打开目录", self)
        open_action.triggered.connect(self.open_path_dialog)
        toolbar.addAction(open_action)

    def _build_layout(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.path_label)
        control_layout.addWidget(QLabel("Exam"))
        control_layout.addWidget(self.exam_combo)
        control_layout.addWidget(QLabel("Volume"))
        control_layout.addWidget(self.volume_combo)
        control_layout.addWidget(QLabel("Slice"))
        control_layout.addWidget(self.slice_slider)
        control_layout.addWidget(self.slice_spin)
        control_layout.addWidget(self.slice_info_label)
        control_layout.addWidget(self.show_spokes_checkbox)
        control_layout.addWidget(self.show_guides_checkbox)
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
        self.exam_combo.blockSignals(True)
        self.exam_combo.clear()
        self.exam_combo.blockSignals(False)

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
        self.canvas.ax_fundus.set_title("Fundus / Zeiss")
        self.canvas.ax_bscan.set_title("B-scan")
        self.canvas.draw_idle()

    def get_last_open_dir(self) -> str:
        last_dir = self.settings.value("last_dir", "", type=str)
        if last_dir and Path(last_dir).exists():
            return last_dir
        return ""

    def remember_path(self, path: str) -> None:
        path = str(Path(path).resolve())
        remember_dir = path if Path(path).is_dir() else str(Path(path).parent)
        self.settings.setValue("last_dir", remember_dir)

    def open_path_dialog(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择蔡司导出目录 / Exam 目录",
            self.get_last_open_dir(),
        )
        if directory:
            self.load_path(directory)

    def load_path(self, path: str):
        try:
            exams = load_exams(path)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.source_path = str(path)
        self.exams = exams
        self.current_exam_index = 0
        self.current_volume_index = 0
        self.current_slice_index = 0
        self.remember_path(path)

        self.path_label.setText(self.source_path)
        self.exam_combo.blockSignals(True)
        self.exam_combo.clear()
        for exam in exams:
            self.exam_combo.addItem(exam.label)
        self.exam_combo.blockSignals(False)
        self.exam_combo.setCurrentIndex(0)
        self.refresh_exam()

    def current_exam(self) -> ExamViewModel | None:
        if not self.exams:
            return None
        return self.exams[self.current_exam_index]

    def current_model(self) -> VolumeViewModel | None:
        exam = self.current_exam()
        if exam is None or not exam.volumes:
            return None
        return exam.volumes[self.current_volume_index]

    def refresh_exam(self):
        exam = self.current_exam()
        if exam is None:
            self._set_empty_state()
            return

        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for model in exam.volumes:
            self.volume_combo.addItem(model.label)
        self.volume_combo.blockSignals(False)

        self.current_volume_index = 0
        self.current_slice_index = 0
        self.volume_combo.setCurrentIndex(0)
        self.refresh_volume()

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

        exam = self.current_exam()
        self.metadata_edit.setPlainText(model.metadata_text)
        self.summary_label.setText(
            f"Exam {exam.exam_id} | {model.source_file} | {len(model.slices)} slices | Laterality: {model.laterality or '-'}"
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
        self.canvas.ax_fundus.set_title("Fundus / Zeiss")

        if model.overlay_bounds is not None:
            x0, y0, width, height = model.overlay_bounds
            self.canvas.ax_fundus.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.85,
                )
            )

        if self.show_spokes_checkbox.isChecked():
            for (start_x, start_y), (end_x, end_y) in model.overlay_spokes:
                self.canvas.ax_fundus.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    color="#d8b14a",
                    alpha=0.45,
                    linewidth=0.9,
                )

        for index, segment in enumerate(model.scan_segments):
            color = "#66ff66" if index == self.current_slice_index else "#d8b14a"
            alpha = 0.95 if index == self.current_slice_index else 0.38
            linewidth = 2.2 if index == self.current_slice_index else 0.8
            (start_x, start_y), (end_x, end_y) = segment
            self.canvas.ax_fundus.plot(
                [start_x, end_x],
                [start_y, end_y],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

        bscan = normalize_to_uint8(model.slices[self.current_slice_index])
        self.canvas.ax_bscan.imshow(bscan, cmap="gray", aspect="auto", origin="upper")
        self.canvas.ax_bscan.axis("off")
        self.canvas.ax_bscan.set_title(
            f"B-scan {self.current_slice_index + 1}/{len(model.slices)}"
        )

        if self.show_guides_checkbox.isChecked():
            guide_top, guide_lower = estimate_bscan_guides(bscan)
            x_coords = np.arange(bscan.shape[1])
            if guide_top is not None:
                self.canvas.ax_bscan.plot(x_coords, guide_top, color="#00ffff", linewidth=1.0, alpha=0.95)
            if guide_lower is not None:
                self.canvas.ax_bscan.plot(x_coords, guide_lower, color="#ff66cc", linewidth=1.0, alpha=0.95)

        self.slice_info_label.setText(
            f"Slice: {self.current_slice_index + 1} / {len(model.slices)}"
        )
        self.canvas.draw_idle()

    def set_slice_index(self, index: int):
        model = self.current_model()
        if model is None:
            return

        index = max(0, min(index, len(model.slices) - 1))
        self.current_slice_index = index

        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(index)
        self.slice_spin.blockSignals(False)

        self.redraw_views()

    def on_exam_changed(self, index: int):
        if index < 0 or index >= len(self.exams):
            return
        self.current_exam_index = index
        self.current_volume_index = 0
        self.current_slice_index = 0
        self.refresh_exam()

    def on_volume_changed(self, index: int):
        exam = self.current_exam()
        if exam is None or index < 0 or index >= len(exam.volumes):
            return
        self.current_volume_index = index
        self.current_slice_index = 0
        self.refresh_volume()

    def on_slice_changed(self, value: int):
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

    def on_canvas_scroll(self, event):
        if event.inaxes != self.canvas.ax_bscan:
            return
        step = 1 if event.button == "up" else -1
        self.set_slice_index(self.current_slice_index + step)


def main():
    args = parse_args()
    application = QApplication(sys.argv)
    window = ZeissViewerWindow()
    window.show()
    if args.path:
        window.load_path(args.path)
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
