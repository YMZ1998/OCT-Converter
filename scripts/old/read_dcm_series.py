import os
from typing import List

import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import tifffile
from pydicom.pixel_data_handlers import convert_color_space


# =========================
# DICOM 排序
# =========================
def get_all_dicom_files_sorted(input_dir: str) -> List[str]:
    """
    读取目录下所有 DICOM Series，按时间顺序拼接返回
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(input_dir)

    if not series_ids:
        raise RuntimeError("❌ 没有找到 DICOM Series")

    print(f"📦 找到 {len(series_ids)} 个 Series")

    def series_sort_key(sid):
        files = reader.GetGDCMSeriesFileNames(input_dir, sid)
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        return (
            int(getattr(ds, "SeriesNumber", 0)),
            getattr(ds, "SeriesTime", "000000"),
        )

    series_ids = sorted(series_ids, key=series_sort_key)

    all_files = []

    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(input_dir, sid)

        def file_sort_key(f):
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            return (
                int(getattr(ds, "InstanceNumber", 0)),
                getattr(ds, "AcquisitionTime", "000000"),
            )

        files = sorted(files, key=file_sort_key)
        print(f"  Series {sid}: {len(files)} frames")

        all_files.extend(files)

    print(f"🎞️ 总帧数: {len(all_files)}")
    return all_files


# =========================
# DICOM → numpy
# =========================
def dicom_to_numpy(ds, keep_16bit=True) -> np.ndarray:
    """
    DICOM PixelData → numpy
    - keep_16bit=True：FA 推荐
    """
    img = ds.pixel_array

    # ---- 彩色 ----
    if ds.SamplesPerPixel == 3:
        if ds.PhotometricInterpretation.startswith("YBR"):
            img = convert_color_space(img, "YBR_FULL_422", "RGB")
        return img.astype(np.uint8)

    # ---- 灰度 ----
    img = img.astype(np.float32)

    if keep_16bit:
        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
        return img.astype(np.uint16)
    else:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)


def dicom_to_bgr(ds) -> np.ndarray:
    """
    DICOM → OpenCV BGR（视频用）
    """
    img = ds.pixel_array

    if ds.SamplesPerPixel == 3:
        if ds.PhotometricInterpretation.startswith("YBR"):
            img = convert_color_space(img, "YBR_FULL_422", "RGB")
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# =========================
# TIFF（多页）
# =========================
def export_tiff(
    input_dir: str,
    output_tiff: str,
    fps: int = 3,
    keep_16bit: bool = True,
    preview: bool = True,
):
    files = get_all_dicom_files_sorted(input_dir)
    frames = []

    if preview:
        plt.figure(figsize=(6, 6))

    for i, f in enumerate(files):
        ds = pydicom.dcmread(f)
        img = dicom_to_numpy(ds, keep_16bit)

        if preview:
            plt.clf()
            plt.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "gray")
            title = f"Frame {i + 1}/{len(files)}"
            if "AcquisitionTime" in ds:
                title += f" | Time {ds.AcquisitionTime}"
            plt.title(title)
            plt.axis("off")
            plt.pause(1 / fps)

        frames.append(img)

    if preview:
        plt.close()

    print(f"💾 写入 TIFF: {output_tiff}")
    tifffile.imwrite(
        output_tiff,
        frames,
        photometric="rgb" if frames[0].ndim == 3 else "minisblack",
    )
    print("✅ TIFF 生成完成")


# =========================
# AVI 视频
# =========================
def export_avi(
    input_dir: str,
    output_avi: str,
    fps: int = 5,
):
    files = get_all_dicom_files_sorted(input_dir)

    first_ds = pydicom.dcmread(files[0])
    first_frame = dicom_to_bgr(first_ds)
    h, w = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_avi,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (w, h),
    )

    print(f"🎬 写入视频: {output_avi}")
    print(f"📐 分辨率: {w}×{h} | FPS: {fps}")

    for i, f in enumerate(files):
        frame = dicom_to_bgr(pydicom.dcmread(f))

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

        if (i + 1) % 10 == 0:
            print(f"  已写入 {i + 1}/{len(files)} 帧")

    writer.release()
    print("✅ AVI 生成完成")


# =========================
# main
# =========================
if __name__ == "__main__":
    # input_dir = r"E:\Data\OCT\蔡司FA\2"
    input_dir = r"E:\Data\OCT\蔡司CFP\18017 V1 FP"

    dirname = os.path.basename(os.path.dirname(input_dir))
    output_dir = r"E:\Data\OCT\Result"
    os.makedirs(output_dir, exist_ok=True)

    export_tiff(
        input_dir,
        os.path.join(output_dir, f"{dirname}.tiff"),
        fps=3,
        keep_16bit=True,
        preview=True,
    )

    export_avi(
        input_dir,
        os.path.join(output_dir, f"{dirname}.avi"),
        fps=3,
    )
