import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from oct_converter.readers import E2E
from scripts.old.dir_process import remove_and_create_dir
from scripts.old.parse_e2e_time import HeidelbergTimeParser


# ================= 工具函数 =================
def compute_angle(m):
    p1 = np.array([m["posX1"], m["posY1"]])
    p2 = np.array([m["posX2"], m["posY2"]])
    v = p2 - p1
    return np.degrees(np.arctan2(v[1], v[0]))


def is_star_scan(bscans):
    centers = []
    for m in bscans:
        if "centrePosX" in m:
            centers.append([m["centrePosX"], m["centrePosY"]])
    if len(centers) < 2:
        return False
    centers = np.array(centers)
    return np.var(centers) < 1e-6


# ================= 分组 =================
def group_bscans_by_numImages(metadata):
    """
    根据 numImages + aktImage 对 B-scan 做分组
    """
    bscans = metadata.get("bscan_data", [])
    if not bscans:
        return {}

    # 按 aktImage 排序
    bscans_sorted = sorted(bscans, key=lambda x: x.get("aktImage", 0))

    groups = defaultdict(list)
    current_group = []
    current_numImages = bscans_sorted[0].get("numImages", 0)

    for bscan in bscans_sorted:
        if bscan.get("numImages") == current_numImages:
            current_group.append(bscan)
        else:
            # 保存上一个 volume
            vid = f"volume_{len(groups)}"
            groups[vid] = current_group
            # 开始新 volume
            current_group = [bscan]
            current_numImages = bscan.get("numImages")

    # 保存最后一个 volume
    if current_group:
        vid = f"volume_{len(groups)}"
        groups[vid] = current_group

    return groups


def group_bscans_by_volume(metadata):
    """
    根据 laterality + scan_pattern + numImages 分组
    """
    from scripts.old.e2e_sort_metadata import sort_metadata
    metadata = sort_metadata(metadata)
    bscans = metadata.get("bscan_data", [])
    # 先解析时间并附加到 bscan 对象
    for bscan in bscans:
        acquisitionTime = bscan.get('acquisitionTime', 0)
        parsed = HeidelbergTimeParser.parse_single(acquisitionTime)
        bscan['parsed_time'] = parsed['cst']  # 或 parsed['utc'] 根据需要

    # 按时间排序
    bscans_sorted = sorted(bscans, key=lambda x: x['parsed_time'])

    # 按 numImages 分组
    groups = defaultdict(list)
    for bscan in bscans:
        numImages = bscan.get('numImages', 0)
        if len(groups[numImages]) < numImages:
            groups[numImages].append(bscan)
        else:
            groups[str(numImages)+"_2"].append(bscan)
        print(f"{len(groups[numImages])}: {bscan['aktImage']}, {bscan['numImages']}, {bscan['parsed_time']}")

    # 打印排序后的结果
    # for numImages, scans in groups.items():
    #     print(f"Group numImages={numImages}:")
    #     for scan in scans:
    #         print(scan['aktImage'], scan['numImages'], scan['parsed_time'])

    # groups = defaultdict(list)
    # start = 0
    # for bscan in bscans:
    #     # print(bscan)
    #     aktImage = bscan.get('aktImage', 0)
    #     numImages = bscan.get('numImages', 0)
    #     acquisitionTime = bscan.get('acquisitionTime', 0)
    #     parsed = HeidelbergTimeParser.parse_single(acquisitionTime)
    #     # print("UTC 时间:", parsed['utc'])
    #     # print("CST 时间:", parsed['cst'])
    #     # print(aktImage, numImages, parsed['cst'])
    #     # 用 laterality + scan_pattern + numImages 生成唯一 volume ID
    #     vid = f"{bscan.get('laterality', 'unknown')}_{bscan.get('scan_pattern', 'unknown')}_{bscan.get('numImages', 0)}"
    #     groups[vid].append(bscan)

    return groups


# ================= 绘图 =================
def plot_each_volume(metadata, output_dir=None):
    volume_groups = group_bscans_by_volume(metadata)
    print(f"[INFO] 共 {len(volume_groups)} 个 volume")
    pass
    for vol_id, bscans in volume_groups.items():
        if len(bscans) == 0:
            continue

        plt.figure(figsize=(6, 6))

        star = is_star_scan(bscans)
        color = "red" if star else "blue"
        scan_type = "Star" if star else "Volume"

        print(f"[INFO] {vol_id}: {scan_type}, slices={len(bscans)}")

        xs, ys = [], []

        for m in bscans:
            x1, y1 = m["posX1"], m["posY1"]
            x2, y2 = m["posX2"], m["posY2"]
            plt.plot([x1, x2], [y1, y2], color=color, alpha=0.7)
            xs += [x1, x2]
            ys += [y1, y2]

        # Star Scan 中心
        if star:
            cx = bscans[0]["centrePosX"]
            cy = bscans[0]["centrePosY"]
            plt.scatter(cx, cy, c="black", s=80, marker="x")
            plt.text(cx, cy, "center", fontsize=10)

        # bounding box
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        plt.gca().add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=color,
                linestyle="--",
                linewidth=2
            )
        )

        plt.title(f"{vol_id} - {scan_type} ({len(bscans)} slices)")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.axis("equal")
        plt.grid(True)

        # 保存图片
        if output_dir:
            save_path = os.path.join(output_dir, f"{vol_id}_{scan_type}.png")
            plt.savefig(save_path, dpi=300)
            print(f"[OK] 保存: {save_path}")

        plt.show()


# ================= 导出 JSON =================
def export_geometry(metadata, out_path):
    volume_groups = group_bscans_by_numImages(metadata)
    geometry = []

    for vol_id, bscans in volume_groups.items():
        for i, m in enumerate(bscans):
            entry = {
                "volume_id": vol_id,
                "slice_id": i,
                "p1": [m["posX1"], m["posY1"]],
                "p2": [m["posX2"], m["posY2"]],
                "center": [m.get("centrePosX"), m.get("centrePosY")],
                "angle_deg": compute_angle(m)
            }
            geometry.append(entry)

    with open(out_path, "w") as f:
        json.dump(geometry, f, indent=2)

    print(f"[OK] Geometry JSON saved: {out_path}")


# ================= 主流程 =================
def run(filepath):
    result_root = r"E:\Data\OCT\Result"
    parent_dir = os.path.dirname(filepath)
    output_dir = os.path.join(
        result_root,
        os.path.basename(parent_dir),
        os.path.splitext(os.path.basename(filepath))[0]
    )
    remove_and_create_dir(output_dir)
    print(f"[INFO] 输出目录: {output_dir}")

    file = E2E(filepath)

    # ===== OCT volume TIFF =====
    oct_volumes = file.read_oct_volume()
    print(f"[INFO] OCT volumes: {len(oct_volumes)}")
    for volume in oct_volumes:
        print(f"[INFO] OCT volume: {volume.num_slices} {volume.laterality} {volume.scan_pattern}")
        out_tiff = os.path.join(output_dir, f"{volume.volume_id}_{volume.laterality}.tiff")
        volume.save(out_tiff)

    # ===== Fundus images =====
    fundus_images = file.read_fundus_image()
    print(f"[INFO] Fundus images: {len(fundus_images)}")
    for image in fundus_images:
        out_png = os.path.join(output_dir, f"{image.image_id}_{image.laterality}.png")
        image.save(out_png)

    # ===== Metadata =====
    metadata = file.read_all_metadata()
    for key, value in metadata.items():
        print(f"[INFO] {key}: {len(value)}")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"[OK] Metadata saved: {meta_path}")

    # ===== 绘图 =====
    plot_each_volume(metadata, output_dir)

    # ===== 导出几何 JSON =====
    geo_path = os.path.join(output_dir, "geometry.json")
    export_geometry(metadata, geo_path)


# ================= 入口 =================
if __name__ == "__main__":
    # filepath = r"E:\Data\OCT\海德堡\海德堡OCT.E2E"
    filepath = r"E:\Data\OCT2\海德堡\KH902-R10-007-D-007001DME-V4-OCT.E2E"
    run(filepath)
