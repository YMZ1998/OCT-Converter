import os
from typing import Any, List

import pydicom

from utils import clean_text

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE

# input_dir = r'E:\Data\OCT\蔡司OCT\DataFiles\E195'
input_dir = r'E:\Data\OCT3\ZEISSOCT\FS006GIA_RD-0034-KH902-60601-RVO-JNYKYY-72-R-072009RVO-01-OCT\DataFiles\E827'


def describe_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if hasattr(value, "VM") and getattr(value, "VM", 1) > 1:
            return "[" + ", ".join(clean_text(v) for v in value if v) + "]"
    except Exception:
        pass

    return clean_text(value) or "-"


def extract_code_meaning(ds) -> List[str]:
    results = []
    for elem in ds.iterall():
        if elem.tag == (0x0008, 0x0104):
            text = clean_text(elem.value)
            if text:
                results.append(text)
    return results


def safe_get(ds, name: str) -> str:
    try:
        return clean_text(getattr(ds, name, ""))
    except Exception:
        return ""


def print_header_summary(ds) -> None:
    fields = [
        ("SOPClassUID", safe_get(ds, "SOPClassUID")),
        ("Modality", safe_get(ds, "Modality")),
        ("Manufacturer", safe_get(ds, "Manufacturer")),
        ("ManufacturerModelName", safe_get(ds, "ManufacturerModelName")),
        ("ProtocolName", safe_get(ds, "ProtocolName")),
        ("StudyDescription", safe_get(ds, "StudyDescription")),
        ("SeriesDescription", safe_get(ds, "SeriesDescription")),
        ("PatientID", safe_get(ds, "PatientID")),
        ("PatientName", safe_get(ds, "PatientName")),
        ("Laterality", safe_get(ds, "Laterality") or safe_get(ds, "ImageLaterality")),
        ("StudyDate", safe_get(ds, "StudyDate")),
        ("StudyTime", safe_get(ds, "StudyTime")),
        ("AcquisitionDate", safe_get(ds, "AcquisitionDate")),
        ("AcquisitionTime", safe_get(ds, "AcquisitionTime")),
        ("Rows", getattr(ds, "Rows", None)),
        ("Columns", getattr(ds, "Columns", None)),
        ("SamplesPerPixel", getattr(ds, "SamplesPerPixel", None)),
        ("BitsAllocated", getattr(ds, "BitsAllocated", None)),
        ("PixelSpacing", getattr(ds, "PixelSpacing", None)),
        ("PhotometricInterpretation", safe_get(ds, "PhotometricInterpretation")),
        ("NumberOfFrames", getattr(ds, "NumberOfFrames", None)),
        ("Zeiss_CodeMeaning", extract_code_meaning(ds) or "-"),
    ]

    for k, v in fields:
        print(f"{k}: {describe_value(v)}")


def try_decode_pixel(ds):
    if "PixelData" not in ds:
        return None
    try:
        return ds.pixel_array
    except Exception as e:
        print("pixel decode failed:", e)
        return None


for file in os.listdir(input_dir):
    if not file.endswith(".DCM"):
        continue

    file_path = os.path.join(input_dir, file)
    print("\n" + "-" * 40)
    print(file)

    ds = pydicom.dcmread(file_path)
    # print(ds)

    # 🔍 Zeiss info
    code_meaning = extract_code_meaning(ds)

    print("CodeMeaning:", code_meaning if code_meaning else "None")
    print(f"Rows/Columns: {getattr(ds, 'Rows', None)} x {getattr(ds, 'Columns', None)}")

    print_header_summary(ds)
