from __future__ import annotations

import io
import json
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "fa"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fa_qt_viewer import (  # type: ignore[import-not-found]
    UnifiedFADataset,
    UnifiedFAFrame,
    VENDOR_AUTO,
    VENDOR_CFP,
    VENDOR_HDB,
    VENDOR_TOPCON,
    VENDOR_ZEISS,
    build_frame_metadata_text,
    build_summary_text,
    load_unified_dataset,
)


SUPPORTED_VENDORS = [
    {"value": VENDOR_AUTO, "label": "自动"},
    {"value": VENDOR_TOPCON, "label": "Topcon"},
    {"value": VENDOR_ZEISS, "label": "Zeiss"},
    {"value": VENDOR_HDB, "label": "HDB"},
    {"value": VENDOR_CFP, "label": "CFP"},
]

FILE_DIALOG_TYPES = [
    ("FA files", "*.e2e *.E2E *.dcm *.dicom *.jpg *.JPG *.jpeg *.JPEG *.png *.PNG *.bmp *.BMP DATAFILE"),
    ("DICOM", "*.dcm *.dicom"),
    ("E2E", "*.e2e *.E2E"),
    ("Images", "*.jpg *.JPG *.jpeg *.JPEG *.png *.PNG *.bmp *.BMP"),
    ("All files", "*.*"),
]

STATE_DIRECTORY = Path.home() / ".oct_converter"
STATE_FILENAME = "fa_web_viewer_state.json"
MAX_RECENT_PATHS = 8
MAX_FRAME_PNG_CACHE_ITEMS = 384


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array, dtype=np.uint8)

    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    array = (array - min_value) * (255.0 / (max_value - min_value))
    return np.clip(array, 0, 255).astype(np.uint8)


def apply_window(image: np.ndarray, contrast_percent: int, brightness_offset: int) -> np.ndarray:
    adjusted = image.astype(np.float32) * (max(1, contrast_percent) / 100.0) + float(brightness_offset)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.ndim == 2:
        pil_image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 1:
        pil_image = Image.fromarray(array[:, :, 0], mode="L")
    elif array.ndim == 3 and array.shape[2] >= 3:
        channels = 4 if array.shape[2] >= 4 else 3
        pil_image = Image.fromarray(array[:, :, :channels], mode="RGBA" if channels == 4 else "RGB")
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def summarize_groups(frames: list[UnifiedFAFrame]) -> str:
    counts: dict[str, int] = {}
    for frame in frames:
        counts[frame.group_label] = counts.get(frame.group_label, 0) + 1
    return ", ".join(f"{key} {value}" for key, value in counts.items()) or "-"


def format_time_range(frames: list[UnifiedFAFrame]) -> str:
    datetimes = [frame.acquisition_datetime for frame in frames if frame.acquisition_datetime is not None]
    if datetimes:
        earliest = min(datetimes)
        latest = max(datetimes)
        return f"{earliest:%Y-%m-%d %H:%M:%S} ~ {latest:%Y-%m-%d %H:%M:%S}"

    elapsed_seconds = [frame.elapsed_seconds for frame in frames if frame.elapsed_seconds is not None]
    if elapsed_seconds:
        return f"+{min(elapsed_seconds):.3f} s ~ +{max(elapsed_seconds):.3f} s"
    return "-"


def frame_to_payload(
    frame: UnifiedFAFrame,
    *,
    dataset: UnifiedFADataset,
    total_frames: int,
) -> dict[str, Any]:
    metadata_text = build_frame_metadata_text(
        dataset,
        frame,
        visible_index=frame.order_index,
        visible_total=total_frames,
        total_frames=total_frames,
    )
    return {
        "originalIndex": frame.order_index,
        "vendor": frame.vendor,
        "filename": frame.filename,
        "sourcePath": str(frame.source_path) if frame.source_path else "",
        "groupKey": frame.group_key,
        "groupLabel": frame.group_label,
        "lateralityKey": frame.laterality_key,
        "lateralityLabel": frame.laterality_label,
        "label": frame.label,
        "sourceDetail": frame.source_detail,
        "acquisitionDisplay": frame.acquisition_display,
        "acquisitionDatetime": to_jsonable(frame.acquisition_datetime),
        "elapsedSeconds": frame.elapsed_seconds,
        "elapsedDisplay": frame.elapsed_display,
        "width": frame.width,
        "height": frame.height,
        "sizeDisplay": frame.size_display,
        "isProofsheet": frame.is_proofsheet,
        "metadataText": metadata_text,
    }


@dataclass
class FALoadedState:
    source_path: str
    vendor_mode: str
    dataset: UnifiedFADataset


class FAViewerState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.loaded_state: FALoadedState | None = None
        self.state_path = STATE_DIRECTORY / STATE_FILENAME
        self.recent_paths: list[str] = self._load_recent_paths()
        self.frame_png_cache: OrderedDict[tuple[int, int, int], bytes] = OrderedDict()

    def load(self, filepath: str, vendor_mode: str = VENDOR_AUTO) -> dict[str, Any]:
        path = Path(filepath).expanduser().resolve()
        dataset = load_unified_dataset(path, vendor=vendor_mode)
        with self.lock:
            self.loaded_state = FALoadedState(
                source_path=str(path),
                vendor_mode=vendor_mode,
                dataset=dataset,
            )
            self.frame_png_cache.clear()
            self._remember_path(str(path))
        return self.build_state_payload()

    def pick_and_load_file(self, vendor_mode: str = VENDOR_AUTO) -> dict[str, Any]:
        filepath = self.pick_file()
        if not filepath:
            payload = self.build_state_payload()
            payload["cancelled"] = True
            return payload
        payload = self.load(filepath, vendor_mode=vendor_mode)
        payload["cancelled"] = False
        return payload

    def pick_and_load_directory(self, vendor_mode: str = VENDOR_AUTO) -> dict[str, Any]:
        directory = self.pick_directory()
        if not directory:
            payload = self.build_state_payload()
            payload["cancelled"] = True
            return payload
        payload = self.load(directory, vendor_mode=vendor_mode)
        payload["cancelled"] = False
        return payload

    def pick_file(self) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise RuntimeError("Native file dialog is not available in this Python environment.") from exc

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            initial_directory = self._get_initial_dialog_directory()
            selected = filedialog.askopenfilename(
                title="Select a FA file",
                filetypes=FILE_DIALOG_TYPES,
                initialdir=str(initial_directory),
            )
        finally:
            root.destroy()
        return selected or ""

    def pick_directory(self) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise RuntimeError("Native directory dialog is not available in this Python environment.") from exc

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            initial_directory = self._get_initial_dialog_directory()
            selected = filedialog.askdirectory(
                title="Select a FA dataset directory",
                mustexist=True,
                initialdir=str(initial_directory),
            )
        finally:
            root.destroy()
        return selected or ""

    def get_frame_png(
        self,
        frame_index: int,
        *,
        contrast_percent: int = 100,
        brightness_offset: int = 0,
    ) -> bytes:
        cache_key = (
            int(frame_index),
            int(contrast_percent),
            int(brightness_offset),
        )

        with self.lock:
            state = self.loaded_state
            if state is None:
                raise ValueError("No FA dataset loaded.")
            dataset = state.dataset
            cached_payload = self.frame_png_cache.get(cache_key)
            if cached_payload is not None:
                self.frame_png_cache.move_to_end(cache_key)
                return cached_payload

        if frame_index < 0 or frame_index >= len(dataset.frames):
            raise IndexError(f"Frame index out of range: {frame_index}")

        frame = dataset.frames[frame_index]
        image = self._frame_to_array(frame)
        display = apply_window(normalize_to_uint8(image), contrast_percent, brightness_offset)
        payload = encode_png(display)

        with self.lock:
            if self.loaded_state is state:
                self.frame_png_cache[cache_key] = payload
                self.frame_png_cache.move_to_end(cache_key)
                while len(self.frame_png_cache) > MAX_FRAME_PNG_CACHE_ITEMS:
                    self.frame_png_cache.popitem(last=False)
        return payload

    def close(self) -> None:
        return

    def build_state_payload(self) -> dict[str, Any]:
        with self.lock:
            state = self.loaded_state
            recent_paths = list(self.recent_paths)

        if state is None:
            return {
                "loaded": False,
                "sourcePath": "",
                "vendorMode": VENDOR_AUTO,
                "supportedVendors": list(SUPPORTED_VENDORS),
                "recentPaths": recent_paths,
                "groups": [],
                "lateralityOptions": [],
            }

        dataset = state.dataset
        frames_payload = [
            frame_to_payload(frame, dataset=dataset, total_frames=len(dataset.frames))
            for frame in dataset.frames
        ]
        groups = []
        seen_groups: set[str] = set()
        for frame in dataset.frames:
            if frame.group_label in seen_groups:
                continue
            seen_groups.add(frame.group_label)
            groups.append(frame.group_label)

        lateralities = []
        seen_lateralities: set[str] = set()
        for frame in dataset.frames:
            if frame.laterality_label in seen_lateralities:
                continue
            seen_lateralities.add(frame.laterality_label)
            lateralities.append(frame.laterality_label)

        return {
            "loaded": True,
            "sourcePath": state.source_path,
            "vendorMode": state.vendor_mode,
            "supportedVendors": list(SUPPORTED_VENDORS),
            "recentPaths": recent_paths,
            "dataset": {
                "inputPath": str(dataset.input_path),
                "vendor": dataset.vendor,
                "vendorLabel": dataset.vendor_label,
                "patientName": dataset.study_info.patient_name,
                "patientId": dataset.study_info.patient_id,
                "sex": dataset.study_info.sex,
                "birthDate": dataset.study_info.birth_date,
                "examDate": dataset.study_info.exam_date,
                "laterality": dataset.study_info.laterality,
                "studyCode": dataset.study_info.study_code,
                "deviceModel": dataset.study_info.device_model,
                "frameCount": len(dataset.frames),
                "groupSummary": summarize_groups(dataset.frames),
                "timeRange": format_time_range(dataset.frames),
                "summaryText": build_summary_text(dataset, dataset.frames),
            },
            "groups": groups,
            "lateralityOptions": lateralities,
            "frames": frames_payload,
        }

    def _frame_to_array(self, frame: UnifiedFAFrame) -> np.ndarray:
        if frame.image_array is not None:
            return np.asarray(frame.image_array)
        if frame.source_path is None:
            raise ValueError(f"Frame image is unavailable: {frame.filename}")
        with Image.open(frame.source_path) as image:
            return np.asarray(image)

    def _remember_path(self, path: str) -> None:
        normalized = path.strip()
        if not normalized:
            return
        self.recent_paths = [item for item in self.recent_paths if item != normalized]
        self.recent_paths.insert(0, normalized)
        self.recent_paths = self.recent_paths[:MAX_RECENT_PATHS]
        self._save_recent_paths()

    def _load_recent_paths(self) -> list[str]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception:
            return []

        recent_paths = payload.get("recent_paths", [])
        if not isinstance(recent_paths, list):
            return []

        normalized_paths: list[str] = []
        seen: set[str] = set()
        for item in recent_paths:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            normalized_paths.append(normalized)
            seen.add(normalized)
            if len(normalized_paths) >= MAX_RECENT_PATHS:
                break
        return normalized_paths

    def _save_recent_paths(self) -> None:
        payload = {
            "recent_paths": self.recent_paths[:MAX_RECENT_PATHS],
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _get_initial_dialog_directory(self) -> Path:
        candidate = ""
        if self.loaded_state is not None:
            candidate = self.loaded_state.source_path
        elif self.recent_paths:
            candidate = self.recent_paths[0]

        if not candidate:
            return Path.home()

        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
        if path.parent.exists():
            return path.parent
        return Path.home()
