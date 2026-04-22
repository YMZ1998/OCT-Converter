import math
from pathlib import Path

import pydicom
from pydicom import dcmread
from pydicom.encaps import generate_pixel_data_frame
from pydicom.uid import JPEG2000, JPEG2000Lossless

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE

INPUT_DIR = Path(r"E:\Data\OCT\蔡司OCT\DataFiles\E196")
OUTPUT_DIR = Path(r"E:\Data\OCT\Result\zeiss_jp2")


def unscramble_czm(frame: bytes) -> bytearray:
    """Return an unscrambled JPEG 2000 frame from a Zeiss payload."""
    frame = bytearray(frame)
    for index in range(0, len(frame), 7):
        frame[index] ^= 0x5A

    jp2_offset = math.floor(len(frame) / 5 * 3)
    offset = frame.find(b"\x00\x00\x00\x0C")
    if offset == -1:
        raise ValueError("No JP2 header found in the scrambled pixel data")

    if jp2_offset != offset:
        print(f"JP2 header found at offset {offset} rather than the expected {jp2_offset}")
        jp2_offset = offset

    data = bytearray()
    data.extend(frame[jp2_offset:jp2_offset + 253])
    data.extend(frame[993:1016])
    data.extend(frame[276:763])
    data.extend(frame[23:276])
    data.extend(frame[1016:jp2_offset])
    data.extend(frame[:23])
    data.extend(frame[763:993])
    data.extend(frame[jp2_offset + 253:])

    assert len(data) == len(frame)
    return data


def iter_dicom_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
    )


def export_unscrambled_jp2(dcm_path: Path, output_dir: Path) -> None:
    ds = dcmread(dcm_path, force=True)
    meta = ds.file_meta

    if meta.TransferSyntaxUID not in (JPEG2000Lossless, JPEG2000):
        print(f"skip {dcm_path.name}: unsupported transfer syntax {meta.TransferSyntaxUID}")
        return

    manufacturer = str(getattr(ds, "Manufacturer", "") or "")
    if not manufacturer.startswith("Carl Zeiss Meditec"):
        print(f"skip {dcm_path.name}: not a Carl Zeiss Meditec file")
        return

    if "PixelData" not in ds:
        print(f"skip {dcm_path.name}: no PixelData")
        return

    file_output_dir = output_dir / dcm_path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    if "NumberOfFrames" in ds:
        nr_frames = ds.NumberOfFrames.split("\0")[0] if isinstance(ds.NumberOfFrames, str) else ds.NumberOfFrames
        frames = generate_pixel_data_frame(ds.PixelData, nr_frames=int(nr_frames))
        count = 0
        for index, frame in enumerate(frames):
            output_path = file_output_dir / f"{dcm_path.stem}_{index:03}.jp2"
            output_path.write_bytes(unscramble_czm(frame))
            count += 1
        print(f"{dcm_path.name}: exported {count} frame(s) -> {file_output_dir}")
        return

    output_path = file_output_dir / f"{dcm_path.stem}.jp2"
    output_path.write_bytes(unscramble_czm(ds.PixelData))
    print(f"{dcm_path.name}: exported 1 frame -> {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dcm_path in iter_dicom_files(INPUT_DIR):
        try:
            export_unscrambled_jp2(dcm_path, OUTPUT_DIR)
        except Exception as exc:
            print(f"{dcm_path.name}: failed -> {exc}")


if __name__ == "__main__":
    main()
