# unscramble.py

import argparse
import math
from pathlib import Path

from pydicom import dcmread
from pydicom.encaps import generate_pixel_data_frame
# from pydicom.encaps import generate_frames
from pydicom.uid import JPEG2000Lossless, JPEG2000


def setup_argparse() -> argparse.Namespace:
    """Setup the command line arguments"""
    parser = argparse.ArgumentParser(
        description=(
            "Read a CZM DICOM dataset and extract and unscramble the pixel "
            "data which is then written to file as one or more JPEG 2000 "
            "images"
        ),
        usage="unscramble path",
    )

    req_opts = parser.add_argument_group("Parameters")
    req_opts.add_argument(
        "path",
        help="The path to the CZM DICOM dataset",
        type=str
    )

    return parser.parse_args()


def unscramble_czm(frame: bytes) -> bytearray:
    """Return an unscrambled image frame.

    Parameters
    ----------
    frame : bytes
        The scrambled CZM JPEG 2000 data frame as found in the DICOM dataset.

    Returns
    -------
    bytearray
        The unscrambled JPEG 2000 data.
    """
    # Fix the 0x5A XORing
    frame = bytearray(frame)
    for ii in range(0, len(frame), 7):
        frame[ii] = frame[ii] ^ 0x5A

    # Offset to the start of the JP2 header - empirically determined
    jp2_offset = math.floor(len(frame) / 5 * 3)

    # Double check that our empirically determined jp2_offset is correct
    offset = frame.find(b"\x00\x00\x00\x0C")
    if offset == -1:
        raise ValueError("No JP2 header found in the scrambled pixel data")

    if jp2_offset != offset:
        print(
            f"JP2 header found at offset {offset} rather than the expected "
            f"{jp2_offset}"
        )
        jp2_offset = offset

    d = bytearray()
    d.extend(frame[jp2_offset:jp2_offset + 253])
    d.extend(frame[993:1016])
    d.extend(frame[276:763])
    d.extend(frame[23:276])
    d.extend(frame[1016:jp2_offset])
    d.extend(frame[:23])
    d.extend(frame[763:993])
    d.extend(frame[jp2_offset + 253:])

    assert len(d) == len(frame)

    return d


if __name__ == "__main__":
    # args = setup_argparse()
    # p = Path(args.path).resolve(strict=True)
    dir = r"E:\Data\OCT\蔡司OCT\DataFiles\E195\2TXU4UFY5T892H35RCF8PBV9D4KKGIER27IOLS4X2ZU.EX.DCM"
    p = Path(dir).resolve(strict=True)
    # Read and check the dataset is CZM
    ds = dcmread(dir)
    meta = ds.file_meta
    if meta.TransferSyntaxUID not in (JPEG2000Lossless, JPEG2000):
        raise ValueError(
            "Only DICOM datasets with a 'Transfer Syntax UID' of JPEG 2000 "
            "(Lossless) or JPEG 2000 are supported"
        )

    if not ds.Manufacturer.startswith("Carl Zeiss Meditec"):
        raise ValueError("Only CZM DICOM datasets are supported")

    if "PixelData" not in ds:
        raise ValueError("No 'Pixel Data' found in the DICOM dataset")

    # Iterate through the frames, unscramble and write to file
    if "NumberOfFrames" in ds:
        # Workaround horrible non-conformance in datasets :(
        if isinstance(ds.NumberOfFrames, str):
            nr_frames = ds.NumberOfFrames.split('\0')[0]
        else:
            nr_frames = ds.NumberOfFrames

        frames = generate_pixel_data_frame(ds.PixelData, nr_frames=int(nr_frames))
        for idx, frame in enumerate(frames):
            with open(f"{p.stem}_{idx:>03}.jp2", "wb") as f:
                f.write(unscramble_czm(frame))
    else:
        # CZM is non-conformant for single frames :(
        with open(f"{p.stem}.jp2", "wb") as f:
            f.write(unscramble_czm(ds.PixelData))