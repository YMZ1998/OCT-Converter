import matplotlib.pyplot as plt
import numpy as np

from oct_converter.readers import FDA


RADIAL_SCAN_MODE = 3


def clamp_bounds_to_shape(bounds, fundus_shape):
    height, width = fundus_shape[:2]
    left, top, right, bottom = [float(value) for value in bounds]
    left = max(0.0, min(left, float(width - 1)))
    right = max(0.0, min(right, float(width - 1)))
    top = max(0.0, min(top, float(height - 1)))
    bottom = max(0.0, min(bottom, float(height - 1)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_horizontal_segments_from_bounds(num_slices, bounds):
    if num_slices <= 0:
        return []

    left, top, right, bottom = bounds
    if num_slices == 1:
        y_pos = (top + bottom) / 2.0
        return [((left, y_pos), (right, y_pos))]

    segments = []
    for index in range(num_slices):
        ratio = index / (num_slices - 1)
        y_pos = top + ratio * (bottom - top)
        segments.append(((left, y_pos), (right, y_pos)))
    return segments


def build_repeated_line_segments(num_slices, start, end):
    return [(start, end) for _ in range(max(0, num_slices))]


def clamp_point_to_shape(point, fundus_shape):
    height, width = fundus_shape[:2]
    x_pos, y_pos = point
    return (
        max(0.0, min(float(width - 1), float(x_pos))),
        max(0.0, min(float(height - 1), float(y_pos))),
    )


def clip_infinite_line_to_rect(point, direction, bounds, eps=1e-8):
    left, top, right, bottom = bounds
    px, py = point
    dx, dy = direction
    intersections = []

    if abs(dx) > eps:
        for x_pos in (left, right):
            t_value = (x_pos - px) / dx
            y_pos = py + t_value * dy
            if top - eps <= y_pos <= bottom + eps:
                intersections.append((t_value, (float(x_pos), float(y_pos))))

    if abs(dy) > eps:
        for y_pos in (top, bottom):
            t_value = (y_pos - py) / dy
            x_pos = px + t_value * dx
            if left - eps <= x_pos <= right + eps:
                intersections.append((t_value, (float(x_pos), float(y_pos))))

    deduped = []
    seen = set()
    for t_value, point_value in intersections:
        key = (round(point_value[0], 6), round(point_value[1], 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((t_value, point_value))

    if len(deduped) < 2:
        return None

    deduped.sort(key=lambda item: item[0])
    return deduped[0][1], deduped[-1][1]


def clip_infinite_line_to_ellipse(point, direction, bounds, eps=1e-8):
    left, top, right, bottom = bounds
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    radius_x = (right - left) / 2.0
    radius_y = (bottom - top) / 2.0
    if radius_x <= eps or radius_y <= eps:
        return None

    px, py = [float(value) for value in point]
    dx, dy = [float(value) for value in direction]

    coeff_a = (dx * dx) / (radius_x * radius_x) + (dy * dy) / (radius_y * radius_y)
    if coeff_a <= eps:
        return None

    coeff_b = 2.0 * (
        ((px - center_x) * dx) / (radius_x * radius_x)
        + ((py - center_y) * dy) / (radius_y * radius_y)
    )
    coeff_c = (
        ((px - center_x) * (px - center_x)) / (radius_x * radius_x)
        + ((py - center_y) * (py - center_y)) / (radius_y * radius_y)
        - 1.0
    )
    discriminant = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c
    if discriminant < -eps:
        return None

    discriminant = max(discriminant, 0.0)
    sqrt_discriminant = float(np.sqrt(discriminant))
    t_values = [
        (-coeff_b - sqrt_discriminant) / (2.0 * coeff_a),
        (-coeff_b + sqrt_discriminant) / (2.0 * coeff_a),
    ]
    points = []
    for t_value in sorted(t_values):
        points.append((px + t_value * dx, py + t_value * dy))

    if len(points) < 2:
        return None
    return points[0], points[-1]


def build_radial_segments(
    num_slices,
    center,
    radius,
    reference_start,
    reference_end,
    fundus_shape,
    effective_bounds=None,
):
    if num_slices <= 0 or radius <= 1e-8:
        return []

    center = np.asarray(center, dtype=float)
    direction = np.asarray(reference_end, dtype=float) - np.asarray(reference_start, dtype=float)
    length = float(np.hypot(direction[0], direction[1]))
    if length <= 1e-8:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction /= length

    base_angle = float(np.arctan2(direction[1], direction[0]))
    angle_step = np.pi / float(num_slices)

    segments = []
    for index in range(num_slices):
        angle = base_angle + index * angle_step
        vector = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        segment = None
        if effective_bounds is not None:
            segment = clip_infinite_line_to_ellipse(center, vector, effective_bounds)

        if segment is None:
            start = clamp_point_to_shape(center - radius * vector, fundus_shape)
            end = clamp_point_to_shape(center + radius * vector, fundus_shape)
        else:
            start = clamp_point_to_shape(segment[0], fundus_shape)
            end = clamp_point_to_shape(segment[1], fundus_shape)
        segments.append((start, end))
    return segments


def build_rotated_parallel_segments(num_slices, axis_start, axis_end, bounds):
    if num_slices <= 0:
        return []

    start = np.asarray(axis_start, dtype=float)
    end = np.asarray(axis_end, dtype=float)
    direction = end - start
    length = float(np.hypot(direction[0], direction[1]))
    if length <= 1e-8:
        return []

    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=float)
    center = (start + end) / 2.0

    left, top, right, bottom = bounds
    corners = np.asarray(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=float,
    )
    offsets = np.dot(corners - center, normal)
    min_offset = float(np.min(offsets))
    max_offset = float(np.max(offsets))

    if num_slices == 1:
        offset_values = [0.0]
    else:
        offset_values = np.linspace(min_offset, max_offset, num_slices)

    segments = []
    for offset in offset_values:
        point = center + offset * normal
        segment = clip_infinite_line_to_rect(point, direction, bounds)
        if segment is not None:
            segments.append(segment)
    return segments


def get_motion_axis(metadata):
    motion_info = metadata.get("img_mot_comp_02") or {}
    keys = (
        "motion_start_x_pos",
        "motion_start_y_pos",
        "motion_end_x_pos",
        "motion_end_y_pos",
    )
    if not all(key in motion_info for key in keys):
        return None

    start = (
        float(motion_info["motion_start_x_pos"]),
        float(motion_info["motion_start_y_pos"]),
    )
    end = (
        float(motion_info["motion_end_x_pos"]),
        float(motion_info["motion_end_y_pos"]),
    )
    if np.hypot(end[0] - start[0], end[1] - start[1]) <= 1e-8:
        return None
    return start, end


def get_scan_mode(metadata, oct_header=None):
    candidates = [
        oct_header or {},
        metadata.get("img_mot_comp_03") or {},
        metadata.get("capture_info_02") or metadata.get("capture_info") or {},
        metadata.get("param_scan_02") or {},
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        raw_value = candidate.get("scan_mode")
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def get_radial_geometry(metadata):
    effective_bounds = None
    effective_range = metadata.get("effective_scan_range") or {}
    effective_bbox = effective_range.get("bounding_box_fundus_pixel")
    if isinstance(effective_bbox, (list, tuple)) and len(effective_bbox) == 4:
        effective_bounds = tuple(float(value) for value in effective_bbox)

    regist_info = metadata.get("regist_info") or {}
    reg_bbox = regist_info.get("bounding_box_in_fundus_pixels")
    if isinstance(reg_bbox, (list, tuple)) and len(reg_bbox) == 4:
        x0, y0, x1_or_half, y1_or_flag = [float(value) for value in reg_bbox]
        if y1_or_flag == 0.0 and x1_or_half > 0.0:
            center = (x0, y0)
            radius = x1_or_half
            return center, radius, (x0 - radius, y0), (x0 + radius, y0), effective_bounds

    if effective_bounds is not None:
        left, top, right, bottom = effective_bounds
        center = ((left + right) / 2.0, (top + bottom) / 2.0)
        radius = min(abs(right - left), abs(bottom - top)) / 2.0
        return center, radius, (center[0] - radius, center[1]), (center[0] + radius, center[1]), effective_bounds

    return None


def parse_fda_localizer_segments(metadata, fundus_shape, num_slices, oct_header=None):
    metadata = metadata or {}

    scan_params = metadata.get("param_scan_04") or metadata.get("param_scan_02") or {}
    y_dimension_mm = float(scan_params.get("y_dimension_mm") or 0.0)
    motion_axis = get_motion_axis(metadata)
    full_image_bounds = (0.0, 0.0, float(fundus_shape[1] - 1), float(fundus_shape[0] - 1))
    scan_mode = get_scan_mode(metadata, oct_header=oct_header)

    if scan_mode == RADIAL_SCAN_MODE:
        radial_geometry = get_radial_geometry(metadata)
        if radial_geometry is not None:
            center, radius, reference_start, reference_end, effective_bounds = radial_geometry
            radial_segments = build_radial_segments(
                num_slices,
                center,
                radius,
                reference_start,
                reference_end,
                fundus_shape,
                effective_bounds=effective_bounds,
            )
            if radial_segments:
                return radial_segments, "radial-regist"

    regist_info = metadata.get("regist_info") or {}
    reg_bbox = regist_info.get("bounding_box_in_fundus_pixels")
    if isinstance(reg_bbox, (list, tuple)) and len(reg_bbox) == 4:
        x0, y0, x1_or_half, y1_or_flag = [float(value) for value in reg_bbox]

        # Some Topcon files encode a single locator line as:
        # [x_center, y, half_width, 0]
        if y1_or_flag == 0.0 and x1_or_half > 0.0:
            if motion_axis is not None:
                return (
                    build_repeated_line_segments(num_slices, motion_axis[0], motion_axis[1]),
                    "motion-rotated-regist-line",
                )

            start = (x0 - x1_or_half, y0)
            end = (x0 + x1_or_half, y0)
            line_bounds = clamp_bounds_to_shape(
                (start[0], min(start[1], end[1]), end[0], max(start[1], end[1]) + 1.0),
                fundus_shape,
            )
            if line_bounds is not None:
                left, _, right, _ = line_bounds
                y_pos = max(0.0, min(float(fundus_shape[0] - 1), y0))
                return (
                    build_repeated_line_segments(num_slices, (left, y_pos), (right, y_pos)),
                    "regist-line",
                )

        rect_bounds = clamp_bounds_to_shape((x0, y0, x1_or_half, y1_or_flag), fundus_shape)
        if rect_bounds is not None:
            if motion_axis is not None:
                rotated = build_rotated_parallel_segments(
                    num_slices,
                    motion_axis[0],
                    motion_axis[1],
                    rect_bounds,
                )
                if rotated:
                    return rotated, "motion-rotated-regist-rect"

            if y_dimension_mm > 0:
                return build_horizontal_segments_from_bounds(num_slices, rect_bounds), "regist-rect"

            left, top, right, bottom = rect_bounds
            y_pos = (top + bottom) / 2.0
            return (
                build_repeated_line_segments(num_slices, (left, y_pos), (right, y_pos)),
                "regist-line-fallback",
            )

    effective_range = metadata.get("effective_scan_range") or {}
    effective_bbox = effective_range.get("bounding_box_fundus_pixel")
    if isinstance(effective_bbox, (list, tuple)) and len(effective_bbox) == 4:
        rect_bounds = clamp_bounds_to_shape(
            tuple(float(value) for value in effective_bbox),
            fundus_shape,
        )
        if rect_bounds is not None:
            if motion_axis is not None:
                rotated = build_rotated_parallel_segments(
                    num_slices,
                    motion_axis[0],
                    motion_axis[1],
                    rect_bounds,
                )
                if rotated:
                    return rotated, "motion-rotated-effective-rect"

            if y_dimension_mm > 0:
                return build_horizontal_segments_from_bounds(num_slices, rect_bounds), "effective-rect"

            left, top, right, bottom = rect_bounds
            y_pos = (top + bottom) / 2.0
            return (
                build_repeated_line_segments(num_slices, (left, y_pos), (right, y_pos)),
                "effective-line",
            )

    if motion_axis is not None:
        rotated = build_rotated_parallel_segments(
            num_slices,
            motion_axis[0],
            motion_axis[1],
            full_image_bounds,
        )
        if rotated:
            return rotated, "motion-rotated-image"

    return None, "missing"


def point_to_segment_distance(point, start, end):
    px, py = [float(value) for value in point]
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-8:
        return float(np.hypot(px - start[0], py - start[1]))

    projection = ((px - start[0]) * segment[0] + (py - start[1]) * segment[1]) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = start + projection * segment
    return float(np.hypot(px - closest[0], py - closest[1]))


def normalize_bscan_image(bscan):
    image = np.asarray(bscan)
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        return np.zeros_like(image, dtype=np.uint8)

    min_value = float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (image - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def extract_slice_contours(volume, slice_index):
    contours = getattr(volume, "contours", None) or {}
    overlay_items = []
    for layer_name, values in contours.items():
        if values is None or slice_index >= len(values):
            continue
        contour = values[slice_index]
        if contour is None:
            continue
        contour_array = np.asarray(contour, dtype=np.float32)
        if contour_array.size == 0 or np.isnan(contour_array).all():
            continue
        overlay_items.append((layer_name, contour_array))
    return overlay_items


def show_localizer_and_bscan(fundus_img, volume, metadata, oct_header=None):
    image = np.asarray(fundus_img)
    num_slices = int(getattr(volume, "num_slices", len(getattr(volume, "volume", []))))
    scan_mode = get_scan_mode(metadata, oct_header=oct_header)
    segments, mode = parse_fda_localizer_segments(
        metadata,
        image.shape,
        num_slices,
        oct_header=oct_header,
    )

    if not segments:
        raise ValueError("Could not parse locator lines from FDA metadata.")

    if not getattr(volume, "volume", None):
        raise ValueError("OCT volume does not contain any B-scans.")

    figure, (fundus_ax, bscan_ax) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("FDA localizer and B-scan viewer")

    fundus_ax.imshow(image)
    fundus_ax.axis("off")
    show_reference_overlays = scan_mode == RADIAL_SCAN_MODE

    line_artists = []
    label_artists = []
    for index, (start, end) in enumerate(segments):
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        color = "red" if index == 0 else "lime"
        linewidth = 2.0 if index == 0 else 1.0
        alpha = 1.0 if index == 0 else (0.65 if show_reference_overlays else 0.0)
        (artist,) = fundus_ax.plot(xs, ys, linewidth=linewidth, color=color, alpha=alpha)
        line_artists.append(artist)
        label_artist = fundus_ax.text(
            end[0] + 3,
            end[1],
            str(index),
            color="yellow",
            fontsize=8,
            va="center",
            visible=show_reference_overlays,
        )
        label_artists.append(label_artist)

    initial_bscan = normalize_bscan_image(volume.volume[0])
    bscan_artist = bscan_ax.imshow(initial_bscan, cmap="gray", aspect="auto")
    bscan_ax.set_xlabel("A-scan")
    bscan_ax.set_ylabel("Depth")
    contour_artists = []
    contour_colors = plt.cm.get_cmap("tab10", 10)

    state = {"index": 0}

    def update_selected_slice(index):
        clamped_index = max(0, min(int(index), len(segments) - 1, len(volume.volume) - 1))
        state["index"] = clamped_index

        for line_index, artist in enumerate(line_artists):
            is_active = line_index == clamped_index
            artist.set_color("red" if is_active else "lime")
            artist.set_linewidth(2.2 if is_active else 1.0)
            artist.set_alpha(1.0 if is_active else (0.55 if show_reference_overlays else 0.0))

        bscan_artist.set_data(normalize_bscan_image(volume.volume[clamped_index]))
        while contour_artists:
            contour_artists.pop().remove()

        slice_contours = extract_slice_contours(volume, clamped_index)
        for contour_index, (layer_name, contour_values) in enumerate(slice_contours):
            xs = np.arange(contour_values.shape[0], dtype=float)
            (artist,) = bscan_ax.plot(
                xs,
                contour_values,
                color=contour_colors(contour_index % 10),
                linewidth=0.9,
                alpha=0.9,
                label=layer_name,
            )
            contour_artists.append(artist)

        fundus_ax.set_title(
            f"FDA localizer ({mode})\nSlice {clamped_index + 1}/{len(segments)}"
        )
        if contour_artists:
            legend_labels = [artist.get_label() for artist in contour_artists[:5]]
            extra = "" if len(contour_artists) <= 5 else " ..."
            contour_text = ", ".join(legend_labels) + extra
            bscan_ax.set_title(
                f"Corresponding B-scan {clamped_index + 1}/{len(volume.volume)}\nContours: {contour_text}"
            )
        else:
            bscan_ax.set_title(f"Corresponding B-scan {clamped_index + 1}/{len(volume.volume)}")
        figure.canvas.draw_idle()

    def on_key_press(event):
        if event.key in {"left", "down", "a"}:
            update_selected_slice(state["index"] - 1)
        elif event.key in {"right", "up", "d"}:
            update_selected_slice(state["index"] + 1)
        elif event.key == "home":
            update_selected_slice(0)
        elif event.key == "end":
            update_selected_slice(len(segments) - 1)

    def on_mouse_click(event):
        if event.inaxes != fundus_ax or event.xdata is None or event.ydata is None:
            return

        point = (event.xdata, event.ydata)
        distances = [point_to_segment_distance(point, start, end) for start, end in segments]
        if not distances:
            return
        update_selected_slice(int(np.argmin(distances)))

    figure.canvas.mpl_connect("key_press_event", on_key_press)
    figure.canvas.mpl_connect("button_press_event", on_mouse_click)

    update_selected_slice(0)
    figure.suptitle("Click a localizer line or use left/right arrow keys to switch B-scans.")
    figure.tight_layout()
    plt.show()


def run(filepath):
    reader = FDA(filepath)
    volume = reader.read_oct_volume()
    fundus = reader.read_fundus_image()

    if volume is None:
        raise ValueError("OCT volume was not found.")
    if fundus is None:
        raise ValueError("Fundus image was not found.")

    metadata = volume.metadata or {}

    print("[INFO] num_slices:", volume.num_slices)
    print("[INFO] scan_mode:", volume.oct_header.get("scan_mode") if volume.oct_header else None)
    print("[INFO] img_mot_comp_02:", metadata.get("img_mot_comp_02"))
    print("[INFO] regist_info:", metadata.get("regist_info"))
    print("[INFO] effective_scan_range:", metadata.get("effective_scan_range"))

    show_localizer_and_bscan(
        fundus.image,
        volume,
        metadata,
        oct_header=volume.oct_header,
    )

#扫描直径6mm 从xml中获取
# - `ScanWidth`：每条B-scan的A-scan数量（横向像素）
# - `ScanCount`：B-scan数量（定位线数量）
# - `ScanHeight`：每条A-scan的深度采样点数
# - `RealScanX`：扫描区域宽度(mm)
# - `RealScanY`：扫描区域高度(mm)或线间隔(mm)
if __name__ == "__main__":
    filepath = r"E:\Data\OCT\拓普康OCT\41365.fda"
    run(filepath)
