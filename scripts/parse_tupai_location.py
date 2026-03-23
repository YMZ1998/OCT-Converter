import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider

DEFAULT_INPUT_DIR = (
    "E:\\Data\\OCT2\\图湃OCT\\KH902-R10 R-096016RVO_男_26_R-096016RVO\\20260303_102557_右眼_12.00mmX12.00mm_3D黄斑"
)


def get_pixel_spacing(ds):
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        return float(pixel_spacing[0]), float(pixel_spacing[1])

    shared_groups = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared_groups and hasattr(shared_groups[0], "PixelMeasuresSequence"):
        pixel_spacing = getattr(shared_groups[0].PixelMeasuresSequence[0], "PixelSpacing", None)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            return float(pixel_spacing[0]), float(pixel_spacing[1])

    return None


def transpose_tupai_reference_coordinates(reference_coordinates):
    coordinates = np.array(reference_coordinates, dtype=float, copy=True)
    if coordinates.ndim == 1:
        return coordinates[[1, 0, 3, 2]]
    return coordinates[:, [1, 0, 3, 2]]


def compute_expected_bscan_length_pixels(fundus_ds, oct_ds):
    fundus_spacing = get_pixel_spacing(fundus_ds)
    oct_spacing = get_pixel_spacing(oct_ds)
    if fundus_spacing is None or oct_spacing is None:
        return None

    bscan_width_pixels = float(getattr(oct_ds, "Columns", 0))
    if bscan_width_pixels <= 0:
        return None

    bscan_width_mm = bscan_width_pixels * oct_spacing[1]
    fundus_x_mm_per_pixel = fundus_spacing[1]
    if fundus_x_mm_per_pixel <= 0:
        return None

    return bscan_width_mm / fundus_x_mm_per_pixel


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


def score_tupai_coordinate_sequence(
    candidate_coordinates,
    fundus_width,
    fundus_height,
    expected_bscan_length_pixels,
):
    finite_mask = np.isfinite(candidate_coordinates).all(axis=1)
    finite_coordinates = candidate_coordinates[finite_mask]
    if len(finite_coordinates) == 0:
        return (-1, float("-inf"), float("-inf"), float("-inf"))

    deltas = finite_coordinates[:, 2:4] - finite_coordinates[:, 0:2]
    lengths = np.linalg.norm(deltas, axis=1)
    median_length = float(np.median(lengths))

    edge_margin = np.minimum.reduce(
        [
            finite_coordinates[:, 0],
            finite_coordinates[:, 2],
            fundus_width - finite_coordinates[:, 0],
            fundus_width - finite_coordinates[:, 2],
            finite_coordinates[:, 1],
            finite_coordinates[:, 3],
            fundus_height - finite_coordinates[:, 1],
            fundus_height - finite_coordinates[:, 3],
        ]
    )
    median_edge_margin = float(np.median(edge_margin))

    if expected_bscan_length_pixels is None:
        length_score = -median_length
    else:
        length_score = -abs(median_length - expected_bscan_length_pixels)

    return (
        int(np.count_nonzero(finite_mask)),
        length_score,
        median_edge_margin,
        -float(np.std(lengths)),
    )


def choose_coordinate_sequence(
    raw_coordinates,
    num_slices,
    fundus_width,
    fundus_height,
    expected_bscan_length_pixels,
):
    candidates = []

    if len(raw_coordinates) == num_slices:
        candidates.append(("direct", raw_coordinates[:num_slices]))

    if len(raw_coordinates) >= 2 * num_slices:
        candidates.append(("even_frames", raw_coordinates[0::2][:num_slices]))
        candidates.append(("odd_frames", raw_coordinates[1::2][:num_slices]))

    if not candidates:
        fallback = raw_coordinates[:num_slices]
        candidates.append(("fallback", fallback))

    best_name = None
    best_coordinates = None
    best_score = (-1, float("-inf"), float("-inf"), float("-inf"))

    for name, candidate in candidates:
        if len(candidate) != num_slices:
            continue
        display_candidate = transpose_tupai_reference_coordinates(candidate)
        candidate_score = score_tupai_coordinate_sequence(
            candidate_coordinates=display_candidate,
            fundus_width=fundus_width,
            fundus_height=fundus_height,
            expected_bscan_length_pixels=expected_bscan_length_pixels,
        )
        if candidate_score > best_score:
            best_score = candidate_score
            best_name = name
            best_coordinates = display_candidate

    if best_coordinates is None:
        raise RuntimeError("Could not match Tupai OCT frame locations to B-scan slices.")

    return np.array(best_coordinates, dtype=float), best_name


def split_tupai_coordinate_sequences(raw_coordinates, num_slices, fundus_ds, oct_ds):
    fundus_height = int(getattr(fundus_ds, "Rows"))
    fundus_width = int(getattr(fundus_ds, "Columns"))
    expected_bscan_length_pixels = compute_expected_bscan_length_pixels(
        fundus_ds=fundus_ds,
        oct_ds=oct_ds,
    )

    coordinates, coordinate_mode = choose_coordinate_sequence(
        raw_coordinates=raw_coordinates,
        num_slices=num_slices,
        fundus_width=fundus_width,
        fundus_height=fundus_height,
        expected_bscan_length_pixels=expected_bscan_length_pixels,
    )

    return (
        np.array(coordinates, dtype=float),
        coordinate_mode,
        np.array(coordinates, dtype=float),
        coordinate_mode,
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
        fundus_ds=ds_fundus,
        oct_ds=ds_bscan,
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
