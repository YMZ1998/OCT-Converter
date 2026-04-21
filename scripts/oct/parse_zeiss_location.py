from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_viewer.oct.zeiss_loader import (
    load_zeiss_oct_dataset,
    normalize_to_uint8,
    resolve_zeiss_exam_dirs,
    to_display_image,
)

ACTUAL_FUNDUS_PRIORITY = {
    "fundus_photo": 0,
    "fundus": 1,
    "enface": 2,
    "multiframe_localizer": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Zeiss OCT location data without launching the Qt viewer.",
    )
    parser.add_argument(
        "--path",
        default=r"E:\Data\OCT\蔡司OCT\DataFiles",
        help="Zeiss export root, exam directory, DCM file, or DICOMDIR path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent size. Use 0 for compact output.",
    )
    parser.add_argument(
        "--volume-index",
        type=int,
        default=0,
        help="Volume index to visualize.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=0,
        help="Slice index to highlight in the visualization.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display a Matplotlib visualization of the parsed location data.",
    )
    parser.add_argument(
        "--save-plot",
        default="",
        help="Optional path to save the Matplotlib visualization.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def serialize_segment(
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, list[float]]:
    start, end = segment
    return {
        "start": [float(start[0]), float(start[1])],
        "end": [float(end[0]), float(end[1])],
    }


def build_volume_summary(dataset, index: int) -> dict[str, Any]:
    volume = dataset.volumes[index]
    overlay = dataset.overlays[index] if index < len(dataset.overlays) else None

    metadata = volume.metadata or {}
    volume_array = volume.volume
    slice_shape = []
    if getattr(volume_array, "ndim", 0) >= 3:
        slice_shape = [int(volume_array.shape[1]), int(volume_array.shape[2])]

    summary = {
        "index": index,
        "volumeId": volume.volume_id,
        "patientId": volume.patient_id,
        "patientName": volume.patient_name,
        "patientBirthDate": volume.patient_dob,
        "sex": volume.sex,
        "laterality": volume.laterality,
        "deviceName": volume.device_name,
        "scanPattern": volume.scan_pattern,
        "acquisitionDate": volume.acquisition_date,
        "numSlices": getattr(volume, "num_slices", None),
        "sliceShape": slice_shape,
        "pixelSpacing": volume.pixel_spacing,
        "sourceFile": metadata.get("source_file"),
        "displaySource": metadata.get("display_source"),
        "displaySourceKind": metadata.get("display_source_kind"),
        "matchedFundusIndex": metadata.get("matched_fundus_index"),
        "matchedFundusLabel": metadata.get("matched_fundus_label"),
        "foveaCenterNorm": metadata.get("fovea_center_norm"),
        "foveaPoint": metadata.get("fovea_point"),
        "overlayBounds": metadata.get("overlay_bounds"),
        "photoOverlayBounds": metadata.get("photo_overlay_bounds"),
        "frameOfReferenceUid": metadata.get("frame_of_reference_uid"),
        "seriesInstanceUid": metadata.get("series_instance_uid"),
        "scanWidthMm": metadata.get("scan_width_mm"),
        "scanHeightMm": metadata.get("scan_height_mm"),
        "bscanHeightMm": metadata.get("bscan_height_mm"),
    }

    if overlay is not None:
        summary.update(
            {
                "fundusMatchMode": overlay.fundus_match_mode,
                "overlayMode": overlay.overlay_mode,
                "projectionMode": overlay.projection_mode,
                "localizerMode": overlay.localizer_mode,
                "warning": overlay.warning,
                "bounds": list(overlay.bounds) if overlay.bounds is not None else None,
                "segmentCount": len(overlay.scan_segments),
                "segments": [serialize_segment(segment) for segment in overlay.scan_segments],
            }
        )

    return to_jsonable(summary)


def build_fundus_summary(dataset, index: int) -> dict[str, Any]:
    fundus = dataset.fundus_images[index]
    metadata = fundus.metadata or {}
    image_shape = list(getattr(fundus.image, "shape", []))
    return to_jsonable(
        {
            "index": index,
            "imageId": fundus.image_id,
            "patientId": fundus.patient_id,
            "patientName": fundus.patient_name,
            "patientBirthDate": fundus.patient_dob,
            "sex": fundus.sex,
            "laterality": fundus.laterality,
            "deviceName": fundus.device_name,
            "scanPattern": fundus.scan_pattern,
            "acquisitionDate": fundus.acquisition_date,
            "pixelSpacing": fundus.pixel_spacing,
            "imageShape": image_shape,
            "sourceFile": metadata.get("source_file"),
            "sourceKind": metadata.get("source_kind"),
            "frameOfReferenceUid": metadata.get("frame_of_reference_uid"),
            "seriesInstanceUid": metadata.get("series_instance_uid"),
            "physicalWidthMm": metadata.get("physical_width_mm"),
            "physicalHeightMm": metadata.get("physical_height_mm"),
        }
    )


def build_dataset_summary(path: str | Path) -> dict[str, Any]:
    dataset = load_zeiss_oct_dataset(path)
    return {
        "vendor": dataset.vendor,
        "inputPath": dataset.input_path,
        "examDirs": list(dataset.exam_dirs),
        "resolvedExamDirs": [str(item) for item in resolve_zeiss_exam_dirs(path)],
        "volumeCount": len(dataset.volumes),
        "fundusCount": len(dataset.fundus_images),
        "overlayCount": len(dataset.overlays),
        "volumes": [build_volume_summary(dataset, index) for index in range(len(dataset.volumes))],
        "fundusImages": [build_fundus_summary(dataset, index) for index in range(len(dataset.fundus_images))],
    }


def clamp_index(index: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(index, size - 1))


def plot_segment(
    ax,
    segment: tuple[tuple[float, float], tuple[float, float]],
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> tuple[float, float]:
    start, end = segment
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)


def source_kind_of(fundus) -> str:
    metadata = fundus.metadata or {}
    return str(metadata.get("source_kind") or "").strip().lower()


def choose_actual_fundus_index(dataset, *, laterality: str | None) -> int:
    if not dataset.fundus_images:
        return -1

    indexed = list(enumerate(dataset.fundus_images))
    exact_laterality = [
        (index, fundus)
        for index, fundus in indexed
        if laterality and fundus.laterality and fundus.laterality == laterality
    ]
    pool = exact_laterality or indexed
    return min(
        pool,
        key=lambda item: (
            ACTUAL_FUNDUS_PRIORITY.get(source_kind_of(item[1]), 99),
            index if (index := item[0]) is not None else 0,
        ),
    )[0]


def draw_overlay_on_axis(ax, overlay, *, slice_index: int, volume_metadata: dict[str, Any]) -> None:
    if overlay is not None and overlay.bounds is not None:
        x0, y0, width, height = overlay.bounds
        outline = np.array(
            [
                [x0, y0],
                [x0 + width, y0],
                [x0 + width, y0 + height],
                [x0, y0 + height],
                [x0, y0],
            ],
            dtype=float,
        )
        ax.plot(outline[:, 0], outline[:, 1], color="#ff7f0e", linewidth=1.4, alpha=0.9)

    current_center: tuple[float, float] | None = None
    if overlay is not None:
        for index, segment in enumerate(overlay.scan_segments):
            is_current = index == slice_index
            center = plot_segment(
                ax,
                segment,
                color="#66ff66" if is_current else "#ff4d4f",
                linewidth=2.4 if is_current else 1.0,
                alpha=0.95 if is_current else 0.35,
            )
            if is_current:
                current_center = center

    fovea_point = volume_metadata.get("fovea_point")
    if isinstance(fovea_point, (list, tuple)) and len(fovea_point) >= 2:
        ax.scatter(
            [float(fovea_point[0])],
            [float(fovea_point[1])],
            color="#00ffff",
            s=42,
            marker="+",
            linewidths=1.5,
            zorder=5,
        )

    if current_center is not None:
        ax.scatter(
            [current_center[0]],
            [current_center[1]],
            color="#66ff66",
            edgecolors="white",
            s=36,
            linewidths=0.8,
            zorder=6,
        )


def render_location_plot(
    dataset,
    *,
    volume_index: int,
    slice_index: int,
):
    if not dataset.volumes:
        raise ValueError("No Zeiss volumes are available for plotting.")

    volume_index = clamp_index(volume_index, len(dataset.volumes))
    volume = dataset.volumes[volume_index]
    overlay = dataset.overlays[volume_index] if volume_index < len(dataset.overlays) else None

    reference_fundus_index = -1
    if overlay is not None:
        reference_fundus_index = overlay.matched_fundus_index
    if reference_fundus_index < 0 or reference_fundus_index >= len(dataset.fundus_images):
        reference_fundus_index = 0 if dataset.fundus_images else -1

    if reference_fundus_index < 0:
        raise ValueError("No fundus image is available for plotting.")

    actual_fundus_index = choose_actual_fundus_index(dataset, laterality=volume.laterality)
    if actual_fundus_index < 0:
        actual_fundus_index = reference_fundus_index

    actual_fundus = dataset.fundus_images[actual_fundus_index]
    reference_fundus = dataset.fundus_images[reference_fundus_index]
    actual_fundus_image = to_display_image(actual_fundus.image)
    reference_fundus_image = to_display_image(reference_fundus.image)
    volume_array = np.asarray(volume.volume)
    if volume_array.ndim < 3:
        raise ValueError(f"Unsupported volume shape for plotting: {volume_array.shape}")

    slice_index = clamp_index(slice_index, volume_array.shape[0])
    bscan = normalize_to_uint8(volume_array[slice_index])
    metadata = volume.metadata or {}

    if actual_fundus_index != reference_fundus_index:
        figure, (ax_actual_fundus, ax_reference, ax_bscan) = plt.subplots(1, 3, figsize=(19, 7))
    else:
        figure, (ax_reference, ax_bscan) = plt.subplots(1, 2, figsize=(14, 7))
        ax_actual_fundus = None

    if ax_actual_fundus is not None:
        if actual_fundus_image.ndim == 2:
            ax_actual_fundus.imshow(actual_fundus_image, cmap="gray", origin="upper")
        else:
            ax_actual_fundus.imshow(actual_fundus_image, origin="upper")
        ax_actual_fundus.set_title(
            f"Actual Fundus | {actual_fundus.image_id or actual_fundus_index} | {source_kind_of(actual_fundus)}"
        )
        ax_actual_fundus.axis("off")

    if reference_fundus_image.ndim == 2:
        ax_reference.imshow(reference_fundus_image, cmap="gray", origin="upper")
    else:
        ax_reference.imshow(reference_fundus_image, origin="upper")
    ax_reference.set_title(
        f"Location Reference | {reference_fundus.image_id or reference_fundus_index} | {source_kind_of(reference_fundus)}"
    )
    ax_reference.axis("off")
    draw_overlay_on_axis(
        ax_reference,
        overlay,
        slice_index=slice_index,
        volume_metadata=metadata,
    )

    if ax_actual_fundus is None:
        ax_reference.set_title(
            f"Fundus / Reference | {reference_fundus.image_id or reference_fundus_index} | {source_kind_of(reference_fundus)}"
        )

    ax_bscan.imshow(bscan, cmap="gray", aspect="auto", origin="upper")
    ax_bscan.set_title(f"B-scan {slice_index + 1}/{volume_array.shape[0]}")
    ax_bscan.axis("off")

    text_lines = [
        f"Volume: {volume.volume_id or volume_index}",
        f"Laterality: {volume.laterality or '-'}",
        f"Overlay: {overlay.overlay_mode if overlay is not None else '-'}",
        f"Projection: {overlay.projection_mode if overlay is not None else '-'}",
        f"Localizer: {overlay.localizer_mode if overlay is not None else '-'}",
        f"Actual fundus source: {source_kind_of(actual_fundus)}",
        f"Reference source: {source_kind_of(reference_fundus)}",
    ]
    figure.suptitle("\n".join(text_lines), fontsize=11)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    args = parse_args()
    dataset = load_zeiss_oct_dataset(args.path)
    summary = build_dataset_summary(args.path)
    indent = None if args.indent <= 0 else args.indent
    payload = json.dumps(summary, ensure_ascii=False, indent=indent)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")

    print(payload)

    figure = render_location_plot(
        dataset,
        volume_index=args.volume_index,
        slice_index=args.slice_index,
    )
    plot_path = Path(args.save_plot).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
