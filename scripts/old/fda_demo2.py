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


def build_radial_segments(num_slices, center, radius, reference_start, reference_end, fundus_shape):
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
        start = clamp_point_to_shape(center - radius * vector, fundus_shape)
        end = clamp_point_to_shape(center + radius * vector, fundus_shape)
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
    regist_info = metadata.get("regist_info") or {}
    reg_bbox = regist_info.get("bounding_box_in_fundus_pixels")
    if isinstance(reg_bbox, (list, tuple)) and len(reg_bbox) == 4:
        x0, y0, x1_or_half, y1_or_flag = [float(value) for value in reg_bbox]
        if y1_or_flag == 0.0 and x1_or_half > 0.0:
            center = (x0, y0)
            radius = x1_or_half
            return center, radius, (x0 - radius, y0), (x0 + radius, y0)

    effective_range = metadata.get("effective_scan_range") or {}
    effective_bbox = effective_range.get("bounding_box_fundus_pixel")
    if isinstance(effective_bbox, (list, tuple)) and len(effective_bbox) == 4:
        left, top, right, bottom = [float(value) for value in effective_bbox]
        center = ((left + right) / 2.0, (top + bottom) / 2.0)
        radius = min(abs(right - left), abs(bottom - top)) / 2.0
        return center, radius, (center[0] - radius, center[1]), (center[0] + radius, center[1])

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
            center, radius, reference_start, reference_end = radial_geometry
            radial_segments = build_radial_segments(
                num_slices,
                center,
                radius,
                reference_start,
                reference_end,
                fundus_shape,
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


def draw_localizer_lines(fundus_img, metadata, num_slices, oct_header=None):
    image = np.asarray(fundus_img)
    segments, mode = parse_fda_localizer_segments(
        metadata,
        image.shape,
        num_slices,
        oct_header=oct_header,
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(image)
    plt.axis("off")

    if not segments:
        raise ValueError("Could not parse locator lines from FDA metadata.")

    for index, (start, end) in enumerate(segments):
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        plt.plot(xs, ys, linewidth=1.0, color="lime")

        if index in (0, len(segments) - 1):
            plt.text(
                end[0] + 3,
                end[1],
                str(index),
                color="yellow",
                fontsize=8,
                va="center",
            )

    plt.title(f"FDA localizer lines ({mode})")
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

    draw_localizer_lines(
        fundus.image,
        metadata,
        volume.num_slices,
        oct_header=volume.oct_header,
    )


if __name__ == "__main__":
    filepath = r"E:\Data\OCT\拓普康OCT\41365.fda"
    run(filepath)
