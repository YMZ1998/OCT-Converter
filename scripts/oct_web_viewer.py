from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oct_converter.readers import BOCT, Dicom, E2E, FDA, FDS, IMG, POCT

SUPPORTED_EXTENSIONS = (".fds", ".fda", ".e2e", ".img", ".oct", ".OCT", ".dcm", ".dicom")
FILE_DIALOG_TYPES = [
    ("Supported OCT files", "*.fds *.fda *.e2e *.img *.oct *.OCT *.dcm *.dicom"),
    ("Topcon FDS", "*.fds"),
    ("Topcon FDA", "*.fda"),
    ("Heidelberg E2E", "*.e2e"),
    ("Zeiss IMG", "*.img"),
    ("Optovue OCT", "*.oct"),
    ("Bioptigen OCT", "*.OCT"),
    ("DICOM", "*.dcm *.dicom"),
    ("All files", "*.*"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local OCT web viewer.")
    parser.add_argument("path", nargs="?", help="Optional OCT file to load at startup.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument("--img-rows", type=int, default=1024, help="Rows for Zeiss .img files.")
    parser.add_argument("--img-cols", type=int, default=512, help="Cols for Zeiss .img files.")
    parser.add_argument(
        "--img-interlaced",
        action="store_true",
        help="Apply de-interlacing for Zeiss .img files.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
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
    adjusted = image.astype(np.float32) * (max(1, contrast_percent) / 100.0) + float(
        brightness_offset
    )
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.ndim == 2:
        ok, payload = cv2.imencode(".png", array)
    elif array.ndim == 3 and array.shape[2] == 1:
        ok, payload = cv2.imencode(".png", array[:, :, 0])
    elif array.ndim == 3 and array.shape[2] == 3:
        ok, payload = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    elif array.ndim == 3 and array.shape[2] == 4:
        ok, payload = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA))
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    if not ok:
        raise ValueError("Failed to encode PNG")
    return payload.tobytes()


def make_label(prefix: str, index: int, identifier: str | None = None, laterality: str | None = None) -> str:
    parts = [prefix, str(index + 1)]
    if identifier:
        parts.append(str(identifier))
    if laterality:
        parts.append(str(laterality))
    return " | ".join(parts)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class LoadedDataset:
    source_path: str
    reader_name: str
    volumes: list[Any]
    fundus_images: list[Any]


class ViewerState:
    def __init__(self, img_rows: int, img_cols: int, img_interlaced: bool) -> None:
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_interlaced = img_interlaced
        self.dataset: LoadedDataset | None = None
        self.lock = threading.Lock()

    def load(self, filepath: str) -> dict[str, Any]:
        path = self._resolve_path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)

        reader = self._create_reader(path)
        volumes = as_list(self._read_oct_volumes(reader))
        if not volumes:
            raise ValueError("No OCT volumes were extracted from this file.")

        fundus_images = as_list(self._safe_read_fundus(reader))
        dataset = LoadedDataset(
            source_path=str(path.resolve()),
            reader_name=reader.__class__.__name__,
            volumes=volumes,
            fundus_images=fundus_images,
        )
        with self.lock:
            self.dataset = dataset
        return self.build_state_payload()

    def pick_and_load(self) -> dict[str, Any]:
        filepath = self.pick_file()
        if not filepath:
            payload = self.build_state_payload()
            payload["cancelled"] = True
            return payload
        payload = self.load(filepath)
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
            selected = filedialog.askopenfilename(
                title="Select an OCT file",
                filetypes=FILE_DIALOG_TYPES,
            )
        finally:
            root.destroy()
        return selected or ""

    def build_state_payload(self) -> dict[str, Any]:
        with self.lock:
            dataset = self.dataset

        if dataset is None:
            return {
                "loaded": False,
                "sourcePath": "",
                "reader": "",
                "volumes": [],
                "fundusImages": [],
                "supportedExtensions": list(SUPPORTED_EXTENSIONS),
            }

        fundus_descriptions = [
            self._describe_fundus(image, index) for index, image in enumerate(dataset.fundus_images)
        ]
        return {
            "loaded": True,
            "sourcePath": dataset.source_path,
            "reader": dataset.reader_name,
            "volumes": [
                self._describe_volume(volume, index, dataset.fundus_images)
                for index, volume in enumerate(dataset.volumes)
            ],
            "fundusImages": fundus_descriptions,
            "supportedExtensions": list(SUPPORTED_EXTENSIONS),
        }

    def get_slice_png(
        self,
        volume_index: int,
        slice_index: int,
        contrast_percent: int,
        brightness_offset: int,
    ) -> bytes:
        volume = self._get_volume(volume_index)
        image = self._get_slice(volume, slice_index)
        image = normalize_to_uint8(image)
        image = apply_window(image, contrast_percent, brightness_offset)
        return encode_png(image)

    def get_projection_png(self, volume_index: int) -> bytes:
        volume = self._get_volume(volume_index)
        projection = self._get_projection_image(volume)
        return encode_png(normalize_to_uint8(np.asarray(projection)))

    def get_fundus_png(self, fundus_index: int) -> bytes:
        with self.lock:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("No file is loaded.")
        if not 0 <= fundus_index < len(dataset.fundus_images):
            raise IndexError(f"Fundus index out of range: {fundus_index}")
        image = self._prepare_display_image(np.asarray(dataset.fundus_images[fundus_index].image))
        return encode_png(normalize_to_uint8(np.asarray(image)))

    def get_contours(self, volume_index: int, slice_index: int) -> dict[str, Any]:
        volume = self._get_volume(volume_index)
        contours = getattr(volume, "contours", None) or {}
        payload: dict[str, list[float | None]] = {}
        for name, values in contours.items():
            if slice_index >= len(values):
                continue
            contour = values[slice_index]
            if contour is None:
                continue
            array = np.asarray(contour, dtype=np.float32)
            if array.size == 0 or np.isnan(array).all():
                continue
            payload[name] = [
                None if not np.isfinite(item) else float(item) for item in array.tolist()
            ]
        return {"sliceIndex": slice_index, "contours": payload}

    def _create_reader(self, path: Path) -> Any:
        suffix = path.suffix
        suffix_lower = suffix.lower()
        if suffix_lower == ".fds":
            return FDS(path)
        if suffix_lower == ".fda":
            return FDA(path)
        if suffix_lower == ".e2e":
            return E2E(path)
        if suffix == ".OCT":
            return BOCT(path)
        if suffix_lower == ".oct":
            return POCT(path)
        if suffix_lower == ".img":
            return IMG(path)
        if suffix_lower in {".dcm", ".dicom"}:
            return Dicom(path)
        raise ValueError(f"Unsupported file type: {suffix or path.name}")

    def _resolve_path(self, filepath: str) -> Path:
        path = Path(filepath.strip()).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def _read_oct_volumes(self, reader: Any) -> Any:
        if isinstance(reader, IMG):
            return reader.read_oct_volume(
                rows=self.img_rows,
                cols=self.img_cols,
                interlaced=self.img_interlaced,
            )
        return reader.read_oct_volume()

    def _safe_read_fundus(self, reader: Any) -> Any:
        if not hasattr(reader, "read_fundus_image"):
            return []
        try:
            return reader.read_fundus_image()
        except Exception:
            return []

    def _get_volume(self, volume_index: int) -> Any:
        with self.lock:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("No file is loaded.")
        if not 0 <= volume_index < len(dataset.volumes):
            raise IndexError(f"Volume index out of range: {volume_index}")
        return dataset.volumes[volume_index]

    def _describe_volume(self, volume: Any, index: int, fundus_images: list[Any]) -> dict[str, Any]:
        slice_count = self._slice_count(volume)
        first_slice = self._get_slice(volume, 0)
        height, width = first_slice.shape[:2]
        display_width_units, display_height_units = self._get_bscan_display_units(
            width=width,
            height=height,
            pixel_spacing=getattr(volume, "pixel_spacing", None),
        )
        contour_names = sorted((getattr(volume, "contours", None) or {}).keys())
        volume_id = getattr(volume, "volume_id", None)
        laterality = getattr(volume, "laterality", None)
        patient_id = getattr(volume, "patient_id", None)
        metadata = getattr(volume, "metadata", None)
        header = getattr(volume, "header", None)
        oct_header = getattr(volume, "oct_header", None)
        matched_fundus_index = self._match_fundus_index(volume, fundus_images)
        return {
            "index": index,
            "label": make_label("Volume", index, volume_id, laterality),
            "volumeId": volume_id or "",
            "laterality": laterality or "",
            "sliceCount": slice_count,
            "width": width,
            "height": height,
            "pixelSpacing": to_jsonable(getattr(volume, "pixel_spacing", None)),
            "displayWidthUnits": display_width_units,
            "displayHeightUnits": display_height_units,
            "displayAspectRatio": (
                (display_width_units / display_height_units)
                if display_width_units and display_height_units
                else (width / height if height else None)
            ),
            "patientId": patient_id or "",
            "acquisitionDate": to_jsonable(getattr(volume, "acquisition_date", None)) or "",
            "contourNames": contour_names,
            "contourCount": len(contour_names),
            "matchedFundusIndex": matched_fundus_index,
            "metadataText": self._format_metadata_text(
                label=f"Volume {index + 1}",
                details={
                    "volume_id": volume_id,
                    "laterality": laterality,
                    "slice_count": slice_count,
                    "width": width,
                    "height": height,
                    "pixel_spacing": getattr(volume, "pixel_spacing", None),
                    "patient_id": getattr(volume, "patient_id", None),
                    "acquisition_date": getattr(volume, "acquisition_date", None),
                    "metadata": metadata,
                    "header": header,
                    "oct_header": oct_header,
                },
            ),
        }

    def _describe_fundus(self, image: Any, index: int) -> dict[str, Any]:
        array = self._prepare_display_image(np.asarray(image.image))
        height, width = array.shape[:2]
        laterality = getattr(image, "laterality", None)
        image_id = getattr(image, "image_id", None)
        return {
            "index": index,
            "label": make_label("Fundus", index, image_id, laterality),
            "imageId": image_id or "",
            "laterality": laterality or "",
            "width": width,
            "height": height,
            "metadataText": self._format_metadata_text(
                label=f"Fundus {index + 1}",
                details={
                    "image_id": image_id,
                    "laterality": laterality,
                    "width": width,
                    "height": height,
                    "pixel_spacing": getattr(image, "pixel_spacing", None),
                    "patient_id": getattr(image, "patient_id", None),
                    "metadata": getattr(image, "metadata", None),
                },
            ),
        }

    def _slice_count(self, volume: Any) -> int:
        return len(self._get_slices(volume))

    def _get_slice(self, volume: Any, slice_index: int) -> np.ndarray:
        slices = self._get_slices(volume)
        if not 0 <= slice_index < len(slices):
            raise IndexError(f"Slice index out of range: {slice_index}")
        return slices[slice_index]

    def _get_slices(self, volume: Any) -> list[np.ndarray]:
        volume_data = getattr(volume, "volume")
        if isinstance(volume_data, list):
            return [self._prepare_display_image(np.asarray(item)) for item in volume_data]

        array = np.asarray(volume_data)
        if array.ndim == 0:
            return [self._prepare_display_image(array)]
        if array.ndim == 1:
            return [self._prepare_display_image(np.expand_dims(array, axis=0))]
        if array.ndim == 2:
            return [self._prepare_display_image(array)]

        slice_axis = self._infer_slice_axis(array, volume)
        volume_array = np.moveaxis(array, slice_axis, 0)
        return [self._prepare_display_image(volume_array[index]) for index in range(volume_array.shape[0])]

    def _infer_slice_axis(self, array: np.ndarray, volume: Any) -> int:
        expected = self._expected_slice_count(volume)
        candidate_axes = list(range(array.ndim))

        if array.ndim >= 4:
            non_channel_axes = [
                axis for axis in candidate_axes
                if array.shape[axis] not in {1, 3, 4}
            ]
            if len(non_channel_axes) >= 1:
                candidate_axes = non_channel_axes

        if expected and expected > 1:
            matching_axes = [axis for axis in candidate_axes if int(array.shape[axis]) == int(expected)]
            if len(matching_axes) == 1:
                return matching_axes[0]

        return min(candidate_axes, key=lambda axis: (int(array.shape[axis]), axis))

    def _expected_slice_count(self, volume: Any) -> int | None:
        candidates = [
            getattr(volume, "oct_header", {}).get("number_slices") if getattr(volume, "oct_header", None) else None,
            getattr(volume, "oct_header", {}).get("num_slices") if getattr(volume, "oct_header", None) else None,
            getattr(volume, "header", {}).get("number_slices") if getattr(volume, "header", None) else None,
        ]
        for candidate in candidates:
            try:
                if candidate is not None and int(candidate) > 0:
                    return int(candidate)
            except (TypeError, ValueError):
                continue
        return None

    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 0:
            return np.array([[array.item()]])

        array = np.squeeze(array)
        if array.ndim == 2:
            return array

        if array.ndim == 3:
            if array.shape[-1] in {1, 3, 4}:
                return array if array.shape[-1] != 1 else array[:, :, 0]
            if array.shape[0] in {1, 3, 4}:
                moved = np.moveaxis(array, 0, -1)
                return moved if moved.shape[-1] != 1 else moved[:, :, 0]
            collapse_axis = int(np.argmin(array.shape))
            return np.mean(array, axis=collapse_axis)

        while array.ndim > 3:
            collapse_axis = int(np.argmin(array.shape))
            array = np.mean(array, axis=collapse_axis)

        return self._prepare_display_image(array)

    def _get_projection_image(self, volume: Any) -> np.ndarray:
        slices = self._get_slices(volume)
        stack = np.asarray([normalize_to_uint8(slice_image) for slice_image in slices])
        if stack.ndim == 4:
            return np.mean(stack, axis=0)
        if stack.ndim == 3:
            return np.mean(stack, axis=1)
        return stack[0]

    def _get_bscan_display_units(
        self,
        *,
        width: int,
        height: int,
        pixel_spacing: Any,
    ) -> tuple[float, float]:
        width_spacing = None
        height_spacing = None

        if isinstance(pixel_spacing, (list, tuple)):
            if len(pixel_spacing) >= 3:
                width_spacing = self._safe_positive_float(pixel_spacing[0])
                candidates = [
                    self._safe_positive_float(pixel_spacing[1]),
                    self._safe_positive_float(pixel_spacing[2]),
                ]
                positive_candidates = [value for value in candidates if value is not None]
                height_spacing = min(positive_candidates) if positive_candidates else None
            elif len(pixel_spacing) == 2:
                width_spacing = self._safe_positive_float(pixel_spacing[0])
                height_spacing = self._safe_positive_float(pixel_spacing[1])

        if width_spacing is None:
            width_spacing = 1.0
        if height_spacing is None:
            height_spacing = 1.0

        return float(width) * width_spacing, float(height) * height_spacing

    def _safe_positive_float(self, value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) and parsed > 0 else None

    def _match_fundus_index(self, volume: Any, fundus_images: list[Any]) -> int:
        if not fundus_images:
            return -1

        volume_laterality = getattr(volume, "laterality", None) or ""
        volume_patient_id = getattr(volume, "patient_id", None) or ""
        best_index = -1
        best_score = -1

        for index, image in enumerate(fundus_images):
            score = 0
            image_laterality = getattr(image, "laterality", None) or ""
            image_patient_id = getattr(image, "patient_id", None) or ""
            if volume_laterality and image_laterality and volume_laterality == image_laterality:
                score += 2
            if volume_patient_id and image_patient_id and volume_patient_id == image_patient_id:
                score += 3
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _format_metadata_text(self, label: str, details: dict[str, Any]) -> str:
        pretty = json.dumps(to_jsonable(details), ensure_ascii=False, indent=2)
        return f"{label}\n{pretty}"


class ViewerRequestHandler(BaseHTTPRequestHandler):
    state: ViewerState
    html_path: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in {"/", "/index.html"}:
                self._serve_index()
                return

            if route == "/api/state":
                self._send_json(HTTPStatus.OK, self.state.build_state_payload())
                return

            if route == "/api/load":
                query = parse_qs(parsed.query)
                filepath = query.get("path", [""])[0].strip()
                if not filepath:
                    raise ValueError("Missing path query parameter.")
                payload = self.state.load(filepath)
                self._send_json(HTTPStatus.OK, payload)
                return

            if route == "/api/pick-file":
                payload = self.state.pick_and_load()
                self._send_json(HTTPStatus.OK, payload)
                return

            match = re.fullmatch(r"/api/volume/(\d+)/slice/(\d+)\.png", route)
            if match:
                query = parse_qs(parsed.query)
                contrast = int(query.get("contrast", ["100"])[0])
                brightness = int(query.get("brightness", ["0"])[0])
                payload = self.state.get_slice_png(
                    volume_index=int(match.group(1)),
                    slice_index=int(match.group(2)),
                    contrast_percent=contrast,
                    brightness_offset=brightness,
                )
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            match = re.fullmatch(r"/api/volume/(\d+)/projection\.png", route)
            if match:
                payload = self.state.get_projection_png(int(match.group(1)))
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            match = re.fullmatch(r"/api/volume/(\d+)/contours/(\d+)\.json", route)
            if match:
                payload = self.state.get_contours(
                    volume_index=int(match.group(1)),
                    slice_index=int(match.group(2)),
                )
                self._send_json(HTTPStatus.OK, payload)
                return

            match = re.fullmatch(r"/api/fundus/(\d+)\.png", route)
            if match:
                payload = self.state.get_fundus_png(int(match.group(1)))
                self._send_bytes(HTTPStatus.OK, payload, "image/png")
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_index(self) -> None:
        payload = self.html_path.read_text(encoding="utf-8").encode("utf-8")
        self._send_bytes(HTTPStatus.OK, payload, "text/html; charset=utf-8")

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, content, "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def build_handler(state: ViewerState, html_path: Path) -> type[ViewerRequestHandler]:
    class Handler(ViewerRequestHandler):
        pass

    Handler.state = state
    Handler.html_path = html_path
    return Handler


def main() -> None:
    args = parse_args()
    html_path = Path(__file__).with_name("oct_web_viewer.html")
    state = ViewerState(
        img_rows=args.img_rows,
        img_cols=args.img_cols,
        img_interlaced=args.img_interlaced,
    )

    if args.path:
        state.load(args.path)

    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, html_path))
    url = f"http://{args.host}:{args.port}"
    print(f"OCT web viewer running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
