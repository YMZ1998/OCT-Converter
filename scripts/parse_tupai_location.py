import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider

DEFAULT_INPUT_DIR = (
    "E:\\Data\\OCT2\\图湃OCT\\KH902-R10 R-096016RVO_男_26_R-096016RVO\\20260303_102557_右眼_12.00mmX12.00mm_3D黄斑"
)


def is_fundus_rc(reference_coordinates, fundus_height):
    x0, y0, x1, y1 = reference_coordinates
    return (
        np.isfinite([x0, y0, x1, y1]).all()
        and 0 <= y0 <= fundus_height
        and 0 <= y1 <= fundus_height
        and abs(y0 - 0) < 5
        and abs(y1 - fundus_height) < 5
    )


def extract_reference_coordinates(ds_bscan, num_slices):
    coordinates = []

    if hasattr(ds_bscan, "PerFrameFunctionalGroupsSequence"):
        for frame in ds_bscan.PerFrameFunctionalGroupsSequence:
            if hasattr(frame, "OphthalmicFrameLocationSequence"):
                rc = np.array(
                    frame.OphthalmicFrameLocationSequence[0].ReferenceCoordinates,
                    dtype=float,
                )
            else:
                rc = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
            coordinates.append(rc)
    else:
        coordinates = [np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)] * num_slices

    return np.array(coordinates, dtype=float)


def choose_coordinate_sequence(raw_coordinates, num_slices, fundus_height):
    candidates = []

    if len(raw_coordinates) == num_slices:
        candidates.append(("direct", raw_coordinates))

    if len(raw_coordinates) == 2 * num_slices:
        candidates.append(("even_frames", raw_coordinates[0::2]))
        candidates.append(("odd_frames", raw_coordinates[1::2]))

    valid_coordinates = np.array(
        [rc for rc in raw_coordinates if is_fundus_rc(rc, fundus_height)],
        dtype=float,
    )
    if len(valid_coordinates) == num_slices:
        candidates.append(("valid_only", valid_coordinates))

    if not candidates:
        fallback = raw_coordinates[:num_slices]
        candidates.append(("fallback", fallback))

    def score(candidate_coordinates):
        valid_count = sum(
            is_fundus_rc(reference_coordinates, fundus_height)
            for reference_coordinates in candidate_coordinates
        )
        x_values = candidate_coordinates[:, [0, 2]]
        x_span = float(np.nanmax(x_values) - np.nanmin(x_values))
        return valid_count, x_span

    best_name = None
    best_coordinates = None
    best_score = (-1, -1.0)

    for name, candidate in candidates:
        if len(candidate) != num_slices:
            continue
        candidate_score = score(candidate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_name = name
            best_coordinates = candidate

    if best_coordinates is None:
        raise RuntimeError("Could not match Tupai OCT frame locations to B-scan slices.")

    return np.array(best_coordinates, dtype=float), best_name


def split_tupai_coordinate_sequences(raw_coordinates, num_slices, fundus_height):
    full_coordinates, full_mode = choose_coordinate_sequence(
        raw_coordinates=raw_coordinates,
        num_slices=num_slices,
        fundus_height=fundus_height,
    )

    segment_candidates = []
    if len(raw_coordinates) == 2 * num_slices:
        segment_candidates.append(("even_frames", raw_coordinates[0::2]))
        segment_candidates.append(("odd_frames", raw_coordinates[1::2]))
    else:
        segment_candidates.append(("direct", raw_coordinates[:num_slices]))

    best_segment_mode = None
    best_segment_coordinates = None
    best_segment_score = (-1.0, -1.0)

    for name, candidate in segment_candidates:
        if len(candidate) != num_slices:
            continue

        lengths = np.linalg.norm(candidate[:, 2:4] - candidate[:, 0:2], axis=1)
        finite_lengths = lengths[np.isfinite(lengths)]
        if len(finite_lengths) == 0:
            continue

        median_length = float(np.median(finite_lengths))
        y_span = float(np.nanmedian(np.abs(candidate[:, 3] - candidate[:, 1])))
        score = (median_length, y_span)
        if score > best_segment_score:
            best_segment_score = score
            best_segment_mode = name
            best_segment_coordinates = candidate

    if best_segment_coordinates is None:
        best_segment_coordinates = full_coordinates
        best_segment_mode = full_mode

    return (
        np.array(full_coordinates, dtype=float),
        full_mode,
        np.array(best_segment_coordinates, dtype=float),
        best_segment_mode,
    )


def compute_scan_band_width_pixels(fundus_ds, oct_ds, segment_coordinates):
    centers = (segment_coordinates[:, 0:2] + segment_coordinates[:, 2:4]) / 2.0
    deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    finite_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(finite_deltas) > 0:
        return max(float(np.median(finite_deltas)), 1.0)

    fundus_spacing = getattr(fundus_ds, "PixelSpacing", None)
    oct_spacing = getattr(oct_ds, "PixelSpacing", None)
    if fundus_spacing is None or oct_spacing is None:
        return 2.0

    fundus_x_mm_per_pixel = float(fundus_spacing[1])
    oct_slice_spacing_mm = float(oct_spacing[1])
    if fundus_x_mm_per_pixel <= 0:
        return 2.0

    width_pixels = oct_slice_spacing_mm / fundus_x_mm_per_pixel
    return max(width_pixels, 1.0)


def make_scan_band_polygon(reference_coordinates, band_width_pixels):
    start = np.array(reference_coordinates[:2], dtype=float)
    end = np.array(reference_coordinates[2:4], dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1e-6:
        return None

    direction = direction / length
    normal = np.array([-direction[1], direction[0]], dtype=float)
    offset = normal * (band_width_pixels / 2.0)
    return np.array(
        [
            start - offset,
            start + offset,
            end + offset,
            end - offset,
        ],
        dtype=float,
    )


def load_tupai_data(input_dir):
    bscan_file = os.path.join(input_dir, "OCT.dcm")
    fundus_file = os.path.join(input_dir, "Fundus.dcm")

    ds_bscan = pydicom.dcmread(bscan_file)
    volume = ds_bscan.pixel_array
    num_slices = volume.shape[0]

    ds_fundus = pydicom.dcmread(fundus_file)
    fundus = ds_fundus.pixel_array
    fundus_height = fundus.shape[0]

    raw_coordinates = extract_reference_coordinates(ds_bscan, num_slices)
    (
        full_coordinates,
        coordinate_mode,
        segment_coordinates,
        segment_mode,
    ) = split_tupai_coordinate_sequences(
        raw_coordinates=raw_coordinates,
        num_slices=num_slices,
        fundus_height=fundus_height,
    )
    scan_band_width_pixels = compute_scan_band_width_pixels(
        ds_fundus,
        ds_bscan,
        segment_coordinates,
    )

    return (
        volume,
        fundus,
        full_coordinates,
        coordinate_mode,
        segment_coordinates,
        segment_mode,
        scan_band_width_pixels,
    )


def show_tupai_viewer(input_dir=DEFAULT_INPUT_DIR):
    (
        volume,
        fundus_image,
        coordinates,
        coordinate_mode,
        segment_coordinates,
        segment_mode,
        scan_band_width_pixels,
    ) = load_tupai_data(input_dir)
    num_slices = volume.shape[0]

    fig, (ax_fundus, ax_bscan) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)

    if fundus_image.ndim == 2:
        ax_fundus.imshow(fundus_image, cmap="gray", origin="upper")
    else:
        ax_fundus.imshow(fundus_image, origin="upper")
    ax_fundus.set_title(
        f"Fundus"
    )
    ax_fundus.axis("off")

    slice_index = 0
    image_bscan = ax_bscan.imshow(volume[slice_index], cmap="gray", origin="upper")
    ax_bscan.set_title(f"B-scan {slice_index + 1}/{num_slices}")
    ax_bscan.axis("off")

    initial_coordinates = coordinates[slice_index]
    initial_segment_coordinates = segment_coordinates[slice_index]
    line = ax_fundus.plot(
        [initial_coordinates[0], initial_coordinates[2]],
        [initial_coordinates[1], initial_coordinates[3]],
        color="red",
        linewidth=1.5,
        alpha=0.9,
    )[0]
    initial_polygon = make_scan_band_polygon(
        initial_segment_coordinates,
        band_width_pixels=scan_band_width_pixels,
    )
    band_patch = Polygon(
        initial_polygon,
        closed=True,
        facecolor="red",
        edgecolor="red",
        alpha=0.22,
        linewidth=1.0,
    )
    ax_fundus.add_patch(band_patch)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Slice",
        0,
        num_slices - 1,
        valinit=slice_index,
        valstep=1,
    )

    def update(_):
        index = int(slider.val)
        image_bscan.set_data(volume[index])
        ax_bscan.set_title(f"B-scan {index + 1}/{num_slices}")

        current_coordinates = coordinates[index]
        current_segment_coordinates = segment_coordinates[index]
        line.set_data(
            [current_coordinates[0], current_coordinates[2]],
            [current_coordinates[1], current_coordinates[3]],
        )
        polygon = make_scan_band_polygon(
            current_segment_coordinates,
            band_width_pixels=scan_band_width_pixels,
        )
        if polygon is not None:
            band_patch.set_xy(polygon)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    show_tupai_viewer()
