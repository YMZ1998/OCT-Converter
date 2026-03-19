import argparse
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import tifffile
from matplotlib.widgets import Slider


DEFAULT_INPUT_DIR = (
    "E:\\Data\\OCT2\\视微OCT\\00538_20260228005_A-031001AMD_OS_2026-03-06_11-47-06Cube 6x6 512x512\\Dicom"
)

FORMAT_ALIASES = {
    "tif": "tiff",
    "tiff": "tiff",
    "gif": "gif",
}

SEGMENTATION_COLORS = [
    "#00ffff",
    "#ff66cc",
    "#66ff66",
    "#ff9933",
    "#66b3ff",
    "#ffffff",
    "#ff5555",
    "#aa88ff",
    "#00cc99",
    "#ffcc00",
    "#ff99ff",
    "#99ccff",
    "#aaffaa",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="View Shiwei OCT location data and export the synchronized visualization.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the Shiwei DICOM files.",
    )
    parser.add_argument(
        "--bscan-file",
        default=None,
        help="Path to the structural B-scan DICOM file.",
    )
    parser.add_argument(
        "--fundus-file",
        default=None,
        help="Path to the fundus/CSSO DICOM file.",
    )
    parser.add_argument(
        "--seg-file",
        default=None,
        help="Path to the segmentation DICOM file.",
    )
    parser.add_argument(
        "--formats",
        default="",
        help="Comma-separated export formats: tiff,gif.",
    )
    parser.add_argument(
        "--output-dir",
        default="E:\\Data\\OCT2\\result",
        help="Export directory. Defaults to <input-dir>/exports when --formats is provided.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Export file basename. Defaults to the B-scan filename stem.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Frames per second for GIF export.",
    )
    parser.add_argument(
        "--seg-candidate-index",
        type=int,
        default=0,
        help="Segmentation candidate index to use manually. Default: 0.",
    )
    parser.add_argument(
        "--seg-flip-x",
        action="store_true",
        help="Flip segmentation curves horizontally.",
    )
    parser.add_argument(
        "--seg-flip-y",
        action="store_true",
        help="Flip segmentation curves vertically.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive Matplotlib viewer.",
    )
    return parser.parse_args()


def normalize_formats(formats_text):
    if not formats_text:
        return []

    normalized = []
    for item in formats_text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in FORMAT_ALIASES:
            raise ValueError(
                f"Unsupported export format: {item}. Supported values are avi, tiff, gif."
            )
        value = FORMAT_ALIASES[key]
        if value not in normalized:
            normalized.append(value)
    return normalized


def find_candidate(input_dir, keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    for path in sorted(Path(input_dir).glob("*.dcm")):
        name = path.name.lower()
        if all(keyword in name for keyword in keywords) and not any(
            keyword in name for keyword in exclude_keywords
        ):
            return str(path)
    return None


def find_by_suffix(input_dir, suffixes):
    suffixes = [suffix.lower() for suffix in suffixes]
    for path in sorted(Path(input_dir).glob("*.dcm")):
        name = path.name.lower()
        for suffix in suffixes:
            if name.endswith(suffix):
                return str(path)
    return None


def resolve_input_files(input_dir, bscan_file=None, fundus_file=None, seg_file=None):
    if bscan_file is None:
        bscan_file = find_by_suffix(
            input_dir,
            [
                "_rotatedstructural_structural.dcm",
                "_structural.dcm",
            ],
        )
        if bscan_file is None:
            bscan_file = find_candidate(input_dir, ["structural"], ["csso", "segmentation"])
        if bscan_file is None:
            bscan_file = find_candidate(input_dir, ["oct"])

    if fundus_file is None:
        fundus_file = find_by_suffix(
            input_dir,
            [
                "_rotatedstructural_csso.dcm",
                "_csso.dcm",
            ],
        )
        if fundus_file is None:
            fundus_file = find_candidate(input_dir, ["csso"])
        if fundus_file is None:
            fundus_file = find_candidate(input_dir, ["fundus"])

    if seg_file is None:
        seg_file = find_by_suffix(
            input_dir,
            [
                "_segmentation.dcm",
            ],
        )
        if seg_file is None:
            seg_file = find_candidate(input_dir, ["segmentation"])

    if bscan_file is None or not os.path.exists(bscan_file):
        raise FileNotFoundError("Could not locate the structural B-scan DICOM file.")
    if fundus_file is None or not os.path.exists(fundus_file):
        raise FileNotFoundError("Could not locate the fundus/CSSO DICOM file.")

    return bscan_file, fundus_file, seg_file


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


def ensure_rgb(image):
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def pad_to_height(image, target_height, fill_value=0):
    if image.shape[0] == target_height:
        return image

    pad_total = target_height - image.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return np.pad(
        image,
        ((pad_top, pad_bottom), (0, 0), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )


def draw_line_rgb(image, start, end, color=(255, 0, 0), thickness=2):
    x0, y0 = start
    x1, y1 = end
    length = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.round(np.linspace(x0, x1, length)).astype(int)
    ys = np.round(np.linspace(y0, y1, length)).astype(int)
    radius = max(0, thickness // 2)

    for x, y in zip(xs, ys):
        x_min = max(0, x - radius)
        x_max = min(image.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(image.shape[0], y + radius + 1)
        image[y_min:y_max, x_min:x_max] = color


def parse_segmentation_surfaces(seg_file, num_slices, bscan_height):
    if seg_file is None or not os.path.exists(seg_file):
        return None

    ds_seg = pydicom.dcmread(seg_file)
    seg_array = ds_seg.pixel_array

    if seg_array.dtype == np.uint32:
        seg_array = seg_array.view(np.float32)
    else:
        seg_array = seg_array.astype(np.float32)

    if seg_array.ndim == 2:
        seg_array = seg_array[np.newaxis, ...]

    if seg_array.ndim != 3:
        return None

    candidates = []
    if seg_array.shape[1] == num_slices:
        candidates.append(seg_array)
    if seg_array.shape[2] == num_slices:
        candidates.append(np.transpose(seg_array, (0, 2, 1)))
    if not candidates and seg_array.shape[0] == num_slices:
        candidates.append(np.transpose(seg_array, (1, 0, 2)))

    processed_candidates = []
    for candidate in candidates:
        valid_mask = (candidate >= 0.0) & (candidate <= bscan_height - 1)
        candidate = np.where(valid_mask, candidate, np.nan)

        valid_layers = []
        for layer in candidate:
            if np.isfinite(layer).sum() < num_slices:
                continue
            if np.nanmax(layer) - np.nanmin(layer) < 1e-3:
                continue
            valid_layers.append(layer)

        if valid_layers:
            processed_candidates.append(np.stack(valid_layers, axis=0))

    if not processed_candidates:
        return None

    return processed_candidates


def get_segmentation_curves(segmentation_surfaces, slice_idx, target_width, orientation=None):
    if segmentation_surfaces is None:
        return []

    actual_slice_idx = slice_idx

    source_width = segmentation_surfaces.shape[2]
    x_source = np.arange(source_width, dtype=float)
    x_target = np.linspace(0, target_width - 1, target_width, dtype=float)
    curves = []

    for layer in segmentation_surfaces:
        curve = np.asarray(layer[actual_slice_idx], dtype=float)
        valid = np.isfinite(curve)
        if np.count_nonzero(valid) < 2:
            continue
        curve_interp = np.interp(x_target, x_source[valid], curve[valid])
        if orientation is not None and orientation.get("flip_x"):
            curve_interp = curve_interp[::-1]
        if orientation is not None and orientation.get("flip_y"):
            curve_interp = orientation["bscan_height"] - 1 - curve_interp
        curves.append(curve_interp)

    return curves


def sample_curve_alignment_score(bscan_slice, curves):
    if not curves:
        return float("-inf")

    image = normalize_to_uint8(bscan_slice).astype(np.float32)
    gradient = np.abs(np.gradient(image, axis=0))
    score = 0.0
    count = 0

    for curve in curves:
        x_coords = np.arange(len(curve), dtype=int)
        y_coords = np.clip(np.round(curve).astype(int), 1, gradient.shape[0] - 2)
        score += float(np.mean(gradient[y_coords, x_coords]))
        count += 1

    if count == 0:
        return float("-inf")
    return score / count


def resolve_segmentation_orientation(volume, segmentation_candidates):
    if not segmentation_candidates:
        return None, None

    sample_slices = sorted(
        {0, len(volume) // 4, len(volume) // 2, (3 * len(volume)) // 4, len(volume) - 1}
    )
    best_score = float("-inf")
    best_surfaces = None
    best_orientation = None

    for candidate in segmentation_candidates:
        for flip_x in (False, True):
            for flip_y in (False, True):
                orientation = {
                    "reverse_slices": False,
                    "flip_x": flip_x,
                    "flip_y": flip_y,
                    "bscan_height": volume.shape[1],
                }
                score = 0.0
                valid_samples = 0

                for slice_idx in sample_slices:
                    curves = get_segmentation_curves(
                        candidate,
                        slice_idx,
                        volume[slice_idx].shape[1],
                        orientation=orientation,
                    )
                    curve_score = sample_curve_alignment_score(volume[slice_idx], curves)
                    if np.isfinite(curve_score):
                        score += curve_score
                        valid_samples += 1

                if valid_samples == 0:
                    continue

                score /= valid_samples
                if score > best_score:
                    best_score = score
                    best_surfaces = candidate
                    best_orientation = orientation

    return best_surfaces, best_orientation


def load_shiwei_data(
    input_dir,
    bscan_file=None,
    fundus_file=None,
    seg_file=None,
    seg_candidate_index=0,
    seg_flip_x=False,
    seg_flip_y=False,
):
    bscan_file, fundus_file, seg_file = resolve_input_files(
        input_dir, bscan_file, fundus_file, seg_file
    )

    ds_bscan = pydicom.dcmread(bscan_file)
    volume = ds_bscan.pixel_array
    num_slices = volume.shape[0]

    ds_fundus = pydicom.dcmread(fundus_file)
    fundus = ds_fundus.pixel_array

    angles = []
    coordinates = []

    if hasattr(ds_bscan, "PerFrameFunctionalGroupsSequence"):
        for frame in ds_bscan.PerFrameFunctionalGroupsSequence:
            if hasattr(frame, "OphthalmicFrameLocationSequence"):
                location = frame.OphthalmicFrameLocationSequence[0]
                coords = np.array(location.ReferenceCoordinates, dtype=float)
            else:
                coords = np.array([0, 0, 0, 0], dtype=float)

            coordinates.append(coords)
            angle = np.arctan2(coords[3] - coords[1], coords[2] - coords[0])
            angles.append(float(np.degrees(angle)))
    else:
        coordinates = [np.array([0, 0, 0, 0], dtype=float) for _ in range(num_slices)]
        angles = [0.0] * num_slices

    segmentation_candidates = parse_segmentation_surfaces(
        seg_file=seg_file,
        num_slices=num_slices,
        bscan_height=volume.shape[1],
    )
    segmentation_surfaces = None
    segmentation_orientation = None
    if segmentation_candidates:
        candidate_count = len(segmentation_candidates)
        if not 0 <= seg_candidate_index < candidate_count:
            raise ValueError(
                f"Invalid --seg-candidate-index={seg_candidate_index}. "
                f"Available candidates: 0 to {candidate_count - 1}."
            )
        segmentation_surfaces = segmentation_candidates[seg_candidate_index]
        segmentation_orientation = {
            "reverse_slices": False,
            "flip_x": False,
            "flip_y": False,
            "bscan_height": volume.shape[1],
        }

    return (
        volume,
        fundus,
        coordinates,
        angles,
        segmentation_surfaces,
        segmentation_orientation,
        bscan_file,
        fundus_file,
        seg_file,
    )


def render_frame(fundus, bscan_slice, coords, angle, slice_idx, total_slices):
    fundus_rgb = ensure_rgb(fundus).copy()
    bscan_rgb = ensure_rgb(bscan_slice)

    coords = np.asarray(coords, dtype=float)
    if coords.size == 4 and np.any(coords):
        x0, y0, x1, y1 = np.round(coords).astype(int)
        draw_line_rgb(fundus_rgb, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)

    panel_height = max(fundus_rgb.shape[0], bscan_rgb.shape[0])
    fundus_rgb = pad_to_height(fundus_rgb, panel_height, fill_value=0)
    bscan_rgb = pad_to_height(bscan_rgb, panel_height, fill_value=0)

    spacer = np.full((panel_height, 24, 3), 20, dtype=np.uint8)
    body = np.concatenate([fundus_rgb, spacer, bscan_rgb], axis=1)
    footer = np.full((32, body.shape[1], 3), 20, dtype=np.uint8)

    progress = int(body.shape[1] * (slice_idx + 1) / max(1, total_slices))
    footer[:, :progress] = (64, 160, 255)

    return np.concatenate([body, footer], axis=0)


def export_visualization(
    volume,
    fundus,
    coordinates,
    angles,
    segmentation_surfaces,
    segmentation_orientation,
    formats,
    output_dir,
    basename,
    fps,
):
    os.makedirs(output_dir, exist_ok=True)

    gif_writer = None
    tiff_frames = []

    try:
        if "gif" in formats:
            gif_path = os.path.join(output_dir, f"{basename}.gif")
            gif_writer = imageio.get_writer(gif_path, mode="I", duration=1.0 / fps, loop=0)
            print(f"[OK] GIF export enabled: {gif_path}")

        for index, bscan_slice in enumerate(volume):
            curves = get_segmentation_curves(
                segmentation_surfaces,
                index,
                bscan_slice.shape[1],
                orientation=segmentation_orientation,
            )
            frame = create_export_frame(
                fundus=fundus,
                bscan_slice=bscan_slice,
                coords=coordinates[index],
                angle=angles[index],
                curves=curves,
                slice_idx=index,
                total_slices=len(volume),
            )
            if gif_writer is not None:
                gif_writer.append_data(frame)
            if "tiff" in formats:
                tiff_frames.append(frame)

        if "tiff" in formats:
            tiff_path = os.path.join(output_dir, f"{basename}.tiff")
            tifffile.imwrite(tiff_path, np.stack(tiff_frames, axis=0), photometric="rgb")
            print(f"[OK] TIFF exported: {tiff_path}")
    finally:
        if gif_writer is not None:
            gif_writer.close()


def create_export_frame(fundus, bscan_slice, coords, angle, curves, slice_idx, total_slices):
    fig, (ax_fundus, ax_bscan) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.03, right=0.97)

    display_fundus = ensure_rgb(fundus)
    if fundus.ndim == 2:
        ax_fundus.imshow(display_fundus[..., 0], cmap="gray", origin="upper")
    else:
        ax_fundus.imshow(display_fundus, origin="upper")
    ax_fundus.axis("off")
    ax_fundus.set_title("Fundus")

    coords = np.asarray(coords, dtype=float)
    if coords.size == 4 and np.any(coords):
        ax_fundus.plot(
            [coords[0], coords[2]],
            [coords[1], coords[3]],
            color="red",
            linewidth=2,
        )

    display_bscan = normalize_to_uint8(bscan_slice)
    ax_bscan.imshow(display_bscan, cmap="gray", origin="upper")
    ax_bscan.axis("off")
    ax_bscan.set_title(f"B-scan slice {slice_idx}, angle {angle:.2f} deg")

    for layer_index, curve in enumerate(curves):
        ax_bscan.plot(
            np.arange(len(curve)),
            curve,
            color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
            linewidth=1.1,
            alpha=0.9,
        )

    fig.suptitle(f"Slice {slice_idx + 1}/{total_slices}")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def show_interactive_viewer(
    volume,
    fundus,
    coordinates,
    angles,
    segmentation_surfaces,
    segmentation_orientation,
):
    fig, (ax_fundus, ax_bscan) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    display_fundus = ensure_rgb(fundus)
    if fundus.ndim == 2:
        ax_fundus.imshow(display_fundus[..., 0], cmap="gray", origin="upper")
    else:
        ax_fundus.imshow(display_fundus, origin="upper")
    ax_fundus.axis("off")

    slice_idx = 0
    display_bscan = normalize_to_uint8(volume[slice_idx])
    image_bscan = ax_bscan.imshow(display_bscan, cmap="gray", origin="upper")
    ax_bscan.set_title(f"B-scan slice {slice_idx}, angle {angles[slice_idx]:.2f} deg")
    ax_bscan.axis("off")

    initial_curves = get_segmentation_curves(
        segmentation_surfaces,
        slice_idx,
        volume[slice_idx].shape[1],
        orientation=segmentation_orientation,
    )
    contour_lines = []
    for layer_index, curve in enumerate(initial_curves):
        contour_line = ax_bscan.plot(
            np.arange(len(curve)),
            curve,
            color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
            linewidth=1.1,
            alpha=0.9,
        )[0]
        contour_lines.append(contour_line)

    initial_coords = coordinates[slice_idx]
    line = ax_fundus.plot(
        [initial_coords[0], initial_coords[2]],
        [initial_coords[1], initial_coords[3]],
        color="red",
        linewidth=2,
    )[0]

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, len(volume) - 1, valinit=slice_idx, valstep=1)

    def update(_):
        index = int(slider.val)
        image_bscan.set_data(normalize_to_uint8(volume[index]))
        ax_bscan.set_title(f"B-scan slice {index}, angle {angles[index]:.2f} deg")
        line.set_data(
            [coordinates[index][0], coordinates[index][2]],
            [coordinates[index][1], coordinates[index][3]],
        )

        curves = get_segmentation_curves(
            segmentation_surfaces,
            index,
            volume[index].shape[1],
            orientation=segmentation_orientation,
        )
        while len(contour_lines) < len(curves):
            layer_index = len(contour_lines)
            contour_line = ax_bscan.plot(
                [],
                [],
                color=SEGMENTATION_COLORS[layer_index % len(SEGMENTATION_COLORS)],
                linewidth=1.1,
                alpha=0.9,
            )[0]
            contour_lines.append(contour_line)

        for layer_index, contour_line in enumerate(contour_lines):
            if layer_index < len(curves):
                curve = curves[layer_index]
                contour_line.set_data(np.arange(len(curve)), curve)
                contour_line.set_visible(True)
            else:
                contour_line.set_visible(False)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    args = parse_args()
    formats = normalize_formats(args.formats)

    (
        volume,
        fundus,
        coordinates,
        angles,
        segmentation_surfaces,
        segmentation_orientation,
        bscan_file,
        _,
        _,
    ) = load_shiwei_data(
        input_dir=args.input_dir,
        bscan_file=args.bscan_file,
        fundus_file=args.fundus_file,
        seg_file=args.seg_file,
        seg_candidate_index=args.seg_candidate_index,
        seg_flip_x=args.seg_flip_x,
        seg_flip_y=args.seg_flip_y,
    )

    basename = args.basename or Path(bscan_file).stem

    if segmentation_surfaces is not None and segmentation_orientation is not None:
        print(
            "[INFO] Manual segmentation config:",
            f"candidate={args.seg_candidate_index},",
            f"flip_x={segmentation_orientation['flip_x']},",
            f"flip_y={segmentation_orientation['flip_y']},",
            f"reverse_slices={segmentation_orientation['reverse_slices']}",
        )

    if formats:
        output_dir = args.output_dir or os.path.join(args.input_dir, "exports")
        export_visualization(
            volume=volume,
            fundus=fundus,
            coordinates=coordinates,
            angles=angles,
            segmentation_surfaces=segmentation_surfaces,
            segmentation_orientation=segmentation_orientation,
            formats=formats,
            output_dir=output_dir,
            basename=basename,
            fps=args.fps,
        )

    if not args.no_show:
        show_interactive_viewer(
            volume,
            fundus,
            coordinates,
            angles,
            segmentation_surfaces,
            segmentation_orientation,
        )


if __name__ == "__main__":
    main()
