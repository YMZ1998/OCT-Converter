import os
from typing import Any, Optional

import pydicom

from scripts.old.dir_process import remove_and_create_dir
from scripts.old.zeiss_dicom import ZEISSDicom

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE

dir = r'E:\Data\OCT\蔡司OCT\DataFiles\E197'
output_dir = r"E:\Data\OCT\Result\zeiss"
remove_and_create_dir(output_dir)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).split("\x00", 1)[0]
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def safe_getattr(dataset: pydicom.dataset.FileDataset, name: str) -> str:
    try:
        return clean_text(getattr(dataset, name, ""))
    except Exception:
        return ""


def describe_value(value: Any) -> str:
    if value is None:
        return "-"
    if hasattr(value, "VM") and getattr(value, "VM", 1) > 1:
        try:
            parts = [clean_text(item) for item in value]
            parts = [part for part in parts if part]
            return "[" + ", ".join(parts) + "]" if parts else "-"
        except Exception:
            pass
    text = clean_text(value)
    return text or "-"


def print_header_summary(ds: pydicom.dataset.FileDataset) -> None:
    fields = [
        ("SOPClassUID", safe_getattr(ds, "SOPClassUID")),
        ("Modality", safe_getattr(ds, "Modality")),
        ("TransferSyntaxUID", clean_text(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))),
        ("Manufacturer", safe_getattr(ds, "Manufacturer")),
        ("ManufacturerModelName", safe_getattr(ds, "ManufacturerModelName")),
        ("ProtocolName", safe_getattr(ds, "ProtocolName")),
        ("StudyDescription", safe_getattr(ds, "StudyDescription")),
        ("SeriesDescription", safe_getattr(ds, "SeriesDescription")),
        ("PatientID", safe_getattr(ds, "PatientID")),
        ("PatientName", safe_getattr(ds, "PatientName")),
        ("PatientSex", safe_getattr(ds, "PatientSex")),
        ("PatientBirthDate", safe_getattr(ds, "PatientBirthDate")),
        ("Laterality", safe_getattr(ds, "Laterality") or safe_getattr(ds, "ImageLaterality")),
        ("StudyDate", safe_getattr(ds, "StudyDate")),
        ("StudyTime", safe_getattr(ds, "StudyTime")),
        ("AcquisitionDate", safe_getattr(ds, "AcquisitionDate")),
        ("AcquisitionTime", safe_getattr(ds, "AcquisitionTime")),
        ("Rows", getattr(ds, "Rows", None)),
        ("Columns", getattr(ds, "Columns", None)),
        ("SamplesPerPixel", getattr(ds, "SamplesPerPixel", None)),
        ("PhotometricInterpretation", safe_getattr(ds, "PhotometricInterpretation")),
        ("NumberOfFrames", getattr(ds, "NumberOfFrames", None)),
        ("BitsAllocated", getattr(ds, "BitsAllocated", None)),
        ("BitsStored", getattr(ds, "BitsStored", None)),
        ("PixelRepresentation", getattr(ds, "PixelRepresentation", None)),
        ("PixelSpacing", getattr(ds, "PixelSpacing", None)),
        ("FrameOfReferenceUID", safe_getattr(ds, "FrameOfReferenceUID")),
        ("SeriesInstanceUID", safe_getattr(ds, "SeriesInstanceUID")),
    ]
    for label, value in fields:
        print(f"{label}: {describe_value(value)}")


def print_dataset_elements(ds: pydicom.dataset.FileDataset) -> None:
    print("Top-level DICOM elements:")
    for element in ds:
        keyword = element.keyword or "-"
        vr = element.VR
        tag_text = f"({element.tag.group:04X},{element.tag.elem:04X})"
        if keyword == "PixelData":
            value_text = f"<{len(element.value)} bytes>"
        elif vr == "SQ":
            value_text = f"<Sequence with {len(element.value)} item(s)>"
        else:
            value_text = describe_value(element.value)
            if len(value_text) > 160:
                value_text = value_text[:157] + "..."
        print(f"  {tag_text} {keyword} [{vr}] = {value_text}")


def try_decode_pixel_array(ds: pydicom.dataset.FileDataset) -> Optional[Any]:
    if "PixelData" not in ds:
        return None
    try:
        return ds.pixel_array
    except Exception as exc:
        print(f"pixel_array decode failed: {exc}")
        return None


def save_decoded_payloads(file_path: str, file_stem: str) -> None:
    try:
        img = ZEISSDicom(file_path)
        oct_volumes, fundus_volumes = img.read_data()
    except Exception as exc:
        print(f"ZEISSDicom.read_data failed: {exc}")
        return

    export_dir = os.path.join(output_dir, file_stem)
    os.makedirs(export_dir, exist_ok=True)
    print(f"ZEISSDicom parsed OCT volumes: {len(oct_volumes)}")
    for idx, volume in enumerate(oct_volumes):
        volume_dir = os.path.join(export_dir, f"oct_{idx}")
        os.makedirs(volume_dir, exist_ok=True)
        print(
            f"  OCT[{idx}] shape={getattr(volume.volume, 'shape', None)} "
            f"laterality={getattr(volume, 'laterality', None)}"
        )
        volume.save(os.path.join(volume_dir, "volume.tiff"))

    print(f"ZEISSDicom parsed fundus images: {len(fundus_volumes)}")
    for idx, image in enumerate(fundus_volumes):
        image_path = os.path.join(export_dir, f"fundus_{idx}.png")
        print(
            f"  Fundus[{idx}] shape={getattr(image.image, 'shape', None)} "
            f"laterality={getattr(image, 'laterality', None)}"
        )
        image.save(image_path)


for file in os.listdir(dir):
    if file.endswith('.DCM'):
        file_path = os.path.join(dir, file)
        print("-" * 40)
        print(file)
        ds = pydicom.dcmread(file_path)
        has_pixel_data = "PixelData" in ds
        print(f"HasPixelData: {has_pixel_data}")
        rows = getattr(ds, "Rows", None)
        columns = getattr(ds, "Columns", None)
        print(f"Rows/Columns: {rows} x {columns}")
        print_header_summary(ds)
        # print_dataset_elements(ds)

        # if has_pixel_data:
        #     pixel_array = try_decode_pixel_array(ds)
        #     if pixel_array is not None:
        #         print(
        #             "pixel_array: "
        #             f"shape={getattr(pixel_array, 'shape', None)} "
        #             f"dtype={getattr(pixel_array, 'dtype', None)} "
        #             f"min={pixel_array.min() if hasattr(pixel_array, 'min') else '-'} "
        #             f"max={pixel_array.max() if hasattr(pixel_array, 'max') else '-'}"
        #         )
