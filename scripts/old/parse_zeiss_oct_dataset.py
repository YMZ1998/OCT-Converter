from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pydicom
import tifffile

from scripts.old.dir_process import remove_and_create_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_ROOT = Path(r"E:\Data\OCT\蔡司OCT")
MODE_CHOICES = ("manifest", "preview", "export", "all")


def ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def get_zeiss_reader_class():
    ensure_repo_on_syspath()
    from scripts.old.zeiss_dicom import ZEISSDicom

    return ZEISSDicom


def ensure_zeiss_fundus_orientation(image_or_fundus: Any) -> np.ndarray:
    image = np.asarray(getattr(image_or_fundus, "image", image_or_fundus))
    if getattr(image_or_fundus, "orientation_normalized", False):
        return image

    ZEISSDicom = get_zeiss_reader_class()
    return ZEISSDicom.normalize_fundus_orientation(image)


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value).split("\x00", 1)[0]
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    text = clean_text(value)
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def safe_dcmread(path: Path, *, stop_before_pixels: bool = False) -> pydicom.dataset.FileDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydicom.dcmread(path, force=True, stop_before_pixels=stop_before_pixels)


def normalize_ref_file_id(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        parts = [clean_text(part) for part in value if clean_text(part)]
        return "/".join(parts)

    return clean_text(value).replace("\\", "/")


def write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"无法编码 PNG: {path}")
    with path.open("wb") as handle:
        buffer.tofile(handle)
    return str(path)


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)
    if out.size == 0:
        return out.astype(np.uint8)

    if out.dtype == np.uint8:
        return out

    out = out.astype(np.float32)
    min_value = float(np.min(out))
    max_value = float(np.max(out))
    if max_value <= min_value:
        return np.zeros(out.shape, dtype=np.uint8)

    out = (out - min_value) * 255.0 / (max_value - min_value)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_gray_volume(volume_array: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_array)

    if volume.ndim == 2:
        return volume[np.newaxis, ...]

    if volume.ndim == 3:
        return volume

    if volume.ndim == 4 and volume.shape[-1] >= 1:
        return volume[..., 0]

    raise ValueError(f"不支持的 OCT 体数据维度: {volume.shape}")


def save_montage(volume: np.ndarray, filepath: Path, title: str) -> str:
    filepath.parent.mkdir(parents=True, exist_ok=True)

    num_slices = int(volume.shape[0])
    montage_count = min(num_slices, 16)
    cols = min(4, montage_count)
    rows = int(math.ceil(montage_count / cols))
    indices = np.linspace(0, num_slices - 1, montage_count).astype(int)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()

    for axis, slice_index in zip(axes, indices):
        axis.imshow(normalize_uint8(volume[slice_index]), cmap="gray")
        axis.set_title(str(slice_index))
        axis.axis("off")

    for axis in axes[len(indices):]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)


def save_oct_exports(
    volume_array: np.ndarray,
    output_dir: Path,
    *,
    export_slice_pngs: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gray_volume = to_gray_volume(volume_array)
    gray_volume_uint8 = normalize_uint8(gray_volume)
    center_index = gray_volume_uint8.shape[0] // 2

    npy_path = output_dir / "volume.npy"
    np.save(npy_path, gray_volume)

    tiff_path = output_dir / "volume.tiff"
    tifffile.imwrite(tiff_path, gray_volume_uint8, photometric="minisblack")

    center_path = output_dir / "center_slice.png"
    write_png(center_path, gray_volume_uint8[center_index])

    projection = gray_volume.astype(np.float32).mean(axis=1)
    projection_path = output_dir / "projection.png"
    write_png(projection_path, normalize_uint8(projection))

    montage_path = output_dir / "montage.png"
    save_montage(gray_volume, montage_path, title=f"OCT volume ({gray_volume.shape[0]} slices)")

    slice_paths: list[str] = []
    if export_slice_pngs:
        slices_dir = output_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        for index, slice_image in enumerate(gray_volume_uint8):
            slice_path = slices_dir / f"slice_{index:03d}.png"
            write_png(slice_path, slice_image)
            slice_paths.append(str(slice_path))

    return {
        "shape": list(gray_volume.shape),
        "dtype": str(gray_volume.dtype),
        "npy_path": str(npy_path),
        "tiff_path": str(tiff_path),
        "center_slice_path": str(center_path),
        "projection_path": str(projection_path),
        "montage_path": str(montage_path),
        "slice_png_paths": slice_paths,
    }


def save_fundus_exports(image_array: np.ndarray, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = ensure_zeiss_fundus_orientation(image_array)
    image_uint8 = normalize_uint8(image)

    png_path = output_dir / "fundus.png"
    write_png(png_path, image_uint8)

    npy_path = output_dir / "fundus.npy"
    np.save(npy_path, image)

    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "png_path": str(png_path),
        "npy_path": str(npy_path),
    }


def read_dicomdir(dicomdir_path: Path) -> dict[str, Any]:
    records_out: list[dict[str, Any]] = []
    referenced_files: set[str] = set()
    patient_id = ""

    ds = safe_dcmread(dicomdir_path)

    for record in getattr(ds, "DirectoryRecordSequence", []):
        record_type = clean_text(getattr(record, "DirectoryRecordType", ""))
        ref_file = normalize_ref_file_id(getattr(record, "ReferencedFileID", None))
        record_info = {
            "record_type": record_type,
            "patient_id": clean_text(getattr(record, "PatientID", "")),
            "study_description": clean_text(getattr(record, "StudyDescription", "")),
            "series_description": clean_text(getattr(record, "SeriesDescription", "")),
            "referenced_file_id": ref_file,
        }
        records_out.append(record_info)

        if record_info["patient_id"] and not patient_id:
            patient_id = record_info["patient_id"]

        if ref_file:
            referenced_files.add(Path(ref_file).name.upper())

    return {
        "dicomdir_path": str(dicomdir_path),
        "patient_id": patient_id,
        "record_count": len(records_out),
        "referenced_files": sorted(referenced_files),
        "records": records_out,
    }


def classify_dicom(
    *,
    has_pixel_data: bool,
    number_of_frames: int | None,
    rows: int | None,
    columns: int | None,
) -> str:
    if not has_pixel_data:
        return "non_image_dicom"

    if number_of_frames is not None and number_of_frames >= 32:
        return "oct_volume"

    if number_of_frames is not None and number_of_frames > 1:
        return "multi_frame_image"

    if rows is not None and columns is not None:
        return "single_frame_image"

    return "unknown_image"


def extract_dicom_header(dcm_path: Path, referenced_files: set[str]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = safe_dcmread(dcm_path, stop_before_pixels=False)

        has_pixel_data = "PixelData" in ds
        rows = parse_int(getattr(ds, "Rows", None))
        columns = parse_int(getattr(ds, "Columns", None))
        number_of_frames = parse_int(getattr(ds, "NumberOfFrames", None))
        laterality = clean_text(getattr(ds, "Laterality", ""))
        transfer_syntax_uid = ""
        if hasattr(ds, "file_meta") and "TransferSyntaxUID" in ds.file_meta:
            transfer_syntax_uid = clean_text(ds.file_meta.TransferSyntaxUID)

        image_type_value = getattr(ds, "ImageType", None)
        if image_type_value is None:
            image_type: list[str] = []
        elif isinstance(image_type_value, (list, tuple)):
            image_type = [clean_text(item) for item in image_type_value if clean_text(item)]
        else:
            image_type = [clean_text(image_type_value)] if clean_text(image_type_value) else []

        classification = classify_dicom(
            has_pixel_data=has_pixel_data,
            number_of_frames=number_of_frames,
            rows=rows,
            columns=columns,
        )

        return {
            "file_name": dcm_path.name,
            "file_path": str(dcm_path),
            "file_size_bytes": dcm_path.stat().st_size,
            "dicomdir_listed": dcm_path.name.upper() in referenced_files,
            "classification": classification,
            "has_pixel_data": has_pixel_data,
            "sop_class_uid": clean_text(getattr(ds, "SOPClassUID", "")),
            "transfer_syntax_uid": transfer_syntax_uid,
            "modality": clean_text(getattr(ds, "Modality", "")),
            "manufacturer": clean_text(getattr(ds, "Manufacturer", "")),
            "study_description": clean_text(getattr(ds, "StudyDescription", "")),
            "series_description": clean_text(getattr(ds, "SeriesDescription", "")),
            "protocol_name": clean_text(getattr(ds, "ProtocolName", "")),
            "image_type": image_type,
            "rows": rows,
            "columns": columns,
            "number_of_frames": number_of_frames,
            "samples_per_pixel": parse_int(getattr(ds, "SamplesPerPixel", None)),
            "photometric_interpretation": clean_text(getattr(ds, "PhotometricInterpretation", "")),
            "laterality": laterality,
            "patient_id": clean_text(getattr(ds, "PatientID", "")),
            "patient_name": clean_text(getattr(ds, "PatientName", "")),
            "study_date": clean_text(getattr(ds, "StudyDate", "")),
            "study_time": clean_text(getattr(ds, "StudyTime", "")),
            "series_number": parse_int(getattr(ds, "SeriesNumber", None)),
            "instance_number": parse_int(getattr(ds, "InstanceNumber", None)),
        }


def decode_zeiss_file(dcm_path: Path) -> tuple[list[Any], list[Any]]:
    ZEISSDicom = get_zeiss_reader_class()
    reader = ZEISSDicom(dcm_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reader.read_data()


def export_decoded_outputs(
    dcm_path: Path,
    exam_id: str,
    output_dir: Path,
    *,
    mode: str,
    export_slice_pngs: bool,
) -> dict[str, Any]:
    want_preview = mode in {"preview", "all"}
    want_export = mode in {"export", "all"}

    oct_volumes, fundus_images = decode_zeiss_file(dcm_path)

    preview_paths: list[str] = []
    oct_assets: list[dict[str, Any]] = []
    fundus_assets: list[dict[str, Any]] = []

    base_stem = dcm_path.stem

    for index, volume in enumerate(oct_volumes):
        gray_volume = to_gray_volume(volume.volume)
        center_slice = normalize_uint8(gray_volume[gray_volume.shape[0] // 2])

        if want_preview:
            preview_path = (
                output_dir
                / "previews"
                / exam_id
                / f"{base_stem}_oct_{index:02d}_center.png"
            )
            preview_paths.append(write_png(preview_path, center_slice))

        if want_export:
            volume_dir = output_dir / "exports" / exam_id / base_stem / f"oct_{index:02d}"
            oct_assets.append(
                save_oct_exports(
                    volume.volume,
                    volume_dir,
                    export_slice_pngs=export_slice_pngs,
                )
            )

    for index, image in enumerate(fundus_images):
        corrected_image = ensure_zeiss_fundus_orientation(image)
        fundus_uint8 = normalize_uint8(corrected_image)

        if want_preview:
            preview_path = (
                output_dir
                / "previews"
                / exam_id
                / f"{base_stem}_fundus_{index:02d}.png"
            )
            preview_paths.append(write_png(preview_path, fundus_uint8))

        if want_export:
            fundus_dir = output_dir / "exports" / exam_id / base_stem / f"fundus_{index:02d}"
            fundus_assets.append(save_fundus_exports(corrected_image, fundus_dir))

    return {
        "decoded_oct_count": len(oct_volumes),
        "decoded_fundus_count": len(fundus_images),
        "preview_paths": preview_paths,
        "oct_exports": oct_assets,
        "fundus_exports": fundus_assets,
    }


def find_exam_dirs(data_files_dir: Path, selected_exam_ids: set[str] | None) -> list[Path]:
    exam_dirs = sorted(path for path in data_files_dir.iterdir() if path.is_dir())
    if not selected_exam_ids:
        return exam_dirs
    return [path for path in exam_dirs if path.name in selected_exam_ids]


def scan_dataset(
    input_root: Path,
    output_dir: Path,
    *,
    mode: str,
    export_slice_pngs: bool,
    selected_exam_ids: set[str] | None,
) -> dict[str, Any]:
    data_files_dir = input_root / "DataFiles"
    if not data_files_dir.exists():
        raise FileNotFoundError(f"找不到 DataFiles 目录: {data_files_dir}")

    exams_out: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    total_dicom_files = 0

    for exam_dir in find_exam_dirs(data_files_dir, selected_exam_ids):
        dicomdir_path = exam_dir / "DICOMDIR"
        dicomdir_info: dict[str, Any] | None = None
        referenced_files: set[str] = set()

        if dicomdir_path.exists():
            dicomdir_info = read_dicomdir(dicomdir_path)
            referenced_files = set(dicomdir_info["referenced_files"])

        dcm_files = sorted(exam_dir.glob("*.DCM"))
        file_infos: list[dict[str, Any]] = []
        exam_counts: Counter[str] = Counter()

        for dcm_file in dcm_files:
            file_info = extract_dicom_header(dcm_file, referenced_files)

            if mode != "manifest" and file_info["has_pixel_data"]:
                try:
                    file_info.update(
                        export_decoded_outputs(
                            dcm_file,
                            exam_dir.name,
                            output_dir,
                            mode=mode,
                            export_slice_pngs=export_slice_pngs,
                        )
                    )
                except Exception as exc:
                    file_info["decode_error"] = f"{type(exc).__name__}: {exc}"

            file_infos.append(file_info)
            exam_counts[file_info["classification"]] += 1
            global_counts[file_info["classification"]] += 1

        total_dicom_files += len(file_infos)

        exams_out.append(
            {
                "exam_id": exam_dir.name,
                "exam_path": str(exam_dir),
                "patient_id": dicomdir_info["patient_id"] if dicomdir_info else "",
                "dicomdir": dicomdir_info,
                "classification_counts": dict(exam_counts),
                "file_count": len(file_infos),
                "files": file_infos,
            }
        )

    return {
        "input_root": str(input_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "exam_count": len(exams_out),
        "file_count": total_dicom_files,
        "classification_counts": dict(global_counts),
        "exams": exams_out,
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    csv_path = output_dir / "files.csv"
    csv_columns = [
        "exam_id",
        "file_name",
        "classification",
        "dicomdir_listed",
        "file_size_bytes",
        "has_pixel_data",
        "decoded_oct_count",
        "decoded_fundus_count",
        "rows",
        "columns",
        "number_of_frames",
        "samples_per_pixel",
        "photometric_interpretation",
        "laterality",
        "patient_id",
        "patient_name",
        "study_date",
        "study_time",
        "series_number",
        "instance_number",
        "manufacturer",
        "modality",
        "study_description",
        "series_description",
        "protocol_name",
        "sop_class_uid",
        "transfer_syntax_uid",
        "decode_error",
        "file_path",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for exam in manifest["exams"]:
            for file_info in exam["files"]:
                row = {column: file_info.get(column, "") for column in csv_columns}
                row["exam_id"] = exam["exam_id"]
                writer.writerow(row)


def print_summary(manifest: dict[str, Any], output_dir: Path) -> None:
    print(f"输入目录: {manifest['input_root']}")
    print(f"输出目录: {output_dir}")
    print(f"运行模式: {manifest['mode']}")
    print(f"检查批次: {manifest['exam_count']}")
    print(f"DICOM 文件数: {manifest['file_count']}")
    print("分类统计:")
    for key, value in sorted(manifest["classification_counts"].items()):
        print(f"  - {key}: {value}")

    if manifest["mode"] in {"preview", "all"}:
        print(f"预览目录: {output_dir / 'previews'}")
    if manifest["mode"] in {"export", "all"}:
        print(f"导出目录: {output_dir / 'exports'}")

    print("批次明细:")
    for exam in manifest["exams"]:
        counts = ", ".join(
            f"{name}={count}" for name, count in sorted(exam["classification_counts"].items())
        )
        patient_id = exam.get("patient_id", "")
        print(f"  - {exam['exam_id']}: patient_id={patient_id or 'N/A'}, {counts}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量解析蔡司 OCT 导出目录，并生成清单、预览和解码导出。")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"蔡司 OCT 导出根目录，默认值: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录，默认写到 <input-root>\\parsed_output",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="all",
        help="manifest=仅清单，preview=清单+预览，export=清单+导出，all=全部",
    )
    parser.add_argument(
        "--exam-id",
        nargs="*",
        default=None,
        help="只处理指定批次，例如 --exam-id E195 E196",
    )
    parser.add_argument(
        "--export-slice-pngs",
        action="store_true",
        help="为 OCT 体数据额外导出全部 B-scan 切片 PNG。",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_root = args.input_root
    output_dir = args.output_dir or (input_root / "parsed_output")
    remove_and_create_dir(output_dir)
    selected_exam_ids = set(args.exam_id) if args.exam_id else None

    manifest = scan_dataset(
        input_root=input_root,
        output_dir=output_dir,
        mode=args.mode,
        export_slice_pngs=args.export_slice_pngs,
        selected_exam_ids=selected_exam_ids,
    )
    write_manifest(manifest, output_dir)
    print_summary(manifest, output_dir)


if __name__ == "__main__":
    main()
