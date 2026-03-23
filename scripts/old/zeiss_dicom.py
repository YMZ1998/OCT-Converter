import math
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.encaps import generate_pixel_data_frame

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData


class ZEISSDicom(object):
    """Class for extracting data from Zeiss's .dcm file format.

    Attributes:
        filepath (str): Path to .dcm file for reading.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

    def find_oct_tags(self, dataset, data_element):
        if data_element.tag == (0x0407, 0x1005):
            num_frames = len(data_element.value)
            volume = []
            print(f"Found {num_frames} frames")
            for frame in data_element:
                scrambled_frame = frame[0x0407, 0x1006]
                unscrambled_frame = self.unscramble_frame(scrambled_frame)
                frame = cv2.imdecode(
                    np.frombuffer(unscrambled_frame, np.uint8), flags=1
                )
                if num_frames == 1:
                    volume = frame
                else:
                    # is grayscale so we take the first channel
                    volume.append(frame[:, :, 0])
            if num_frames == 1:
                self.all_fundus.append(volume)
            else:
                self.all_oct.append(volume)

    def read_data(self):
        """Reads OCT and fundus data."""
        ds = pydicom.dcmread(self.filepath)
        if not ds.Manufacturer.startswith("Carl Zeiss Meditec"):
            raise ValueError(
                "This does not appear to be a Zeiss DCM. You may need to read with the DCM class."
            )

        self.all_oct = []
        self.all_fundus = []
        all_oct_out = []
        all_fundus_out = []

        # pass 1
        ds.walk(self.find_oct_tags)
        for volume in self.all_oct:
            array = np.rot90(np.array(volume), axes=(1, 2), k=3)
            all_oct_out.append(
                OCTVolumeWithMetaData(
                    volume=array,
                    patient_id=ds.PatientID,
                    first_name=str(ds.PatientName).split("^")[1],
                    surname=str(ds.PatientName).split("^")[0],
                    sex=ds.PatientSex,
                    acquisition_date=ds.StudyDate,
                    laterality=ds.Laterality,
                )
            )
        for image in self.all_fundus:
            all_fundus_out.append(
                FundusImageWithMetaData(
                    image=image,
                    patient_id=ds.PatientID,
                    laterality=ds.Laterality,
                )
            )

        # pass 2
        if "PixelData" in ds:
            if "NumberOfFrames" in ds:
                if isinstance(ds.NumberOfFrames, str):
                    nr_frames = ds.NumberOfFrames.split("\0")[0]
                else:
                    nr_frames = ds.NumberOfFrames

                frames = generate_pixel_data_frame(ds.PixelData, int(nr_frames))
                all_frames = []
                for idx, scrambled_frame in enumerate(frames):
                    # try decoding without unscrambling
                    frame = cv2.imdecode(
                        np.frombuffer(scrambled_frame, np.uint8), flags=1
                    )
                    # if that fails, unscramble and decode
                    if frame is None:
                        unscrambled_frame = self.unscramble_frame(scrambled_frame)
                        frame = cv2.imdecode(
                            np.frombuffer(unscrambled_frame, np.uint8), flags=1
                        )
                    all_frames.append(frame)
                array = np.rot90(np.array(all_frames), axes=(1, 2), k=3)
                all_oct_out.append(
                    OCTVolumeWithMetaData(
                        volume=array,
                        patient_id=ds.PatientID,
                        first_name=str(ds.PatientName).split("^")[1],
                        surname=str(ds.PatientName).split("^")[0],
                        sex=ds.PatientSex,
                        acquisition_date=ds.StudyDate,
                        laterality=ds.Laterality,
                    )
                )
            else:
                unscrambled_frame = self.unscramble_frame(ds.PixelData)
                frame = cv2.imdecode(
                    np.frombuffer(unscrambled_frame, np.uint8), flags=1
                )
                all_fundus_out.append(
                    FundusImageWithMetaData(
                        image=frame,
                        patient_id=ds.PatientID,
                        laterality=ds.Laterality,
                    )
                )

        return all_oct_out, all_fundus_out

    def unscramble_frame(self, frame: bytes) -> bytearray:
        """Return an unscrambled image frame. Thanks to https://github.com/scaramallion for the code,
         as detailed in https://github.com/pydicom/pydicom/discussions/1618.

        Args
        frame (bytes): The scrambled CZM JPEG 2000 data frame as found in the DICOM dataset.

        Returns
        bytearray: The unscrambled JPEG 2000 data.
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
        d.extend(frame[jp2_offset: jp2_offset + 253])
        d.extend(frame[993:1016])
        d.extend(frame[276:763])
        d.extend(frame[23:276])
        d.extend(frame[1016:jp2_offset])
        d.extend(frame[:23])
        d.extend(frame[763:993])
        d.extend(frame[jp2_offset + 253:])

        assert len(d) == len(frame)

        return d
