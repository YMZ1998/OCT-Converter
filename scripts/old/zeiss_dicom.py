from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
from pydicom.encaps import generate_pixel_data_frame

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE

ZEISS_MANUFACTURER_PREFIX = "Carl Zeiss Meditec"
OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"
PRIVATE_FRAME_SEQUENCE_TAG = (0x0407, 0x1005)
PRIVATE_FRAME_DATA_TAG = (0x0407, 0x1006)


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


def clean_digits(value: Any) -> str:
    return "".join(character for character in clean_text(value) if character.isdigit())


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def clean_person_name(value: Any) -> tuple[str, str, str]:
    text = clean_text(value)
    if not text:
        return "", "", ""
    primary = text.split("=", 1)[0]
    parts = [part for part in primary.split("^") if part]
    surname = parts[0] if parts else ""
    first_name = parts[1] if len(parts) > 1 else ""
    patient_name = " ".join(parts).strip()
    return patient_name, first_name, surname


def parse_dicom_datetime(ds: pydicom.dataset.FileDataset) -> datetime | str | None:
    date_time_digits = clean_digits(getattr(ds, "AcquisitionDateTime", ""))
    if len(date_time_digits) >= 14:
        try:
            return datetime.strptime(date_time_digits[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass

    date_digits = clean_digits(getattr(ds, "AcquisitionDate", "") or getattr(ds, "StudyDate", ""))
    time_digits = clean_digits(getattr(ds, "AcquisitionTime", "") or getattr(ds, "StudyTime", ""))
    if len(date_digits) == 8:
        time_digits = (time_digits + "000000")[:6]
        try:
            return datetime.strptime(f"{date_digits}{time_digits}", "%Y%m%d%H%M%S")
        except ValueError:
            return date_digits
    return clean_text(getattr(ds, "StudyDate", "")) or None


def build_device_name(ds: pydicom.dataset.FileDataset) -> str:
    manufacturer = clean_text(getattr(ds, "Manufacturer", ""))
    model = clean_text(getattr(ds, "ManufacturerModelName", ""))
    if manufacturer and model and model.lower() not in manufacturer.lower():
        return f"{manufacturer} {model}"
    return manufacturer or model


class ZEISSDicom(object):
    """Extract OCT volumes and fundus images from a Zeiss DICOM file."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

    @staticmethod
    def normalize_fundus_orientation(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(f"Unsupported Zeiss fundus image shape: {array.shape}")
        return np.rot90(array, axes=(0, 1), k=3).copy()

    @staticmethod
    def normalize_oct_orientation(volume: list[np.ndarray] | np.ndarray) -> np.ndarray:
        array = np.asarray(volume)
        if array.ndim == 4 and array.shape[-1] >= 1:
            array = array[..., 0]
        elif array.ndim == 3 and array.shape[-1] in {3, 4}:
            array = array[..., 0]
            array = array[np.newaxis, ...]
        if array.ndim != 3:
            raise ValueError(f"Unsupported Zeiss OCT volume shape: {array.shape}")
        return np.rot90(array, axes=(1, 2), k=3).copy()

    @staticmethod
    def frame_to_gray(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[2] >= 1:
            return array[:, :, 0]
        raise ValueError(f"Unsupported Zeiss frame shape: {array.shape}")

    @staticmethod
    def is_volume_like(frames: list[np.ndarray]) -> bool:
        if len(frames) <= 1:
            return False
        try:
            array = ZEISSDicom.normalize_oct_orientation([ZEISSDicom.frame_to_gray(frame) for frame in frames])
        except Exception:
            return False
        height, width = array.shape[1:3]
        ratio = max(width, height) / max(min(width, height), 1)
        return len(frames) >= 8 or ratio >= 1.35

    @staticmethod
    def pixel_signature(array: np.ndarray) -> tuple[tuple[int, ...], str, str]:
        contiguous = np.ascontiguousarray(np.asarray(array))
        digest = hashlib.sha1(contiguous.tobytes()).hexdigest()
        return contiguous.shape, str(contiguous.dtype), digest

    @staticmethod
    def laterality(ds: pydicom.dataset.FileDataset) -> str:
        return clean_text(getattr(ds, "Laterality", "")) or clean_text(getattr(ds, "ImageLaterality", ""))

    def read_dataset(self) -> pydicom.dataset.FileDataset:
        with pydicom.config.disable_value_validation():
            return pydicom.dcmread(self.filepath, force=True)

    def validate_dataset(self, ds: pydicom.dataset.FileDataset) -> None:
        manufacturer = clean_text(getattr(ds, "Manufacturer", ""))
        if manufacturer.startswith(ZEISS_MANUFACTURER_PREFIX):
            return
        sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
        if sop_class_uid == OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID or "PixelData" in ds:
            return
        raise ValueError(
            "This does not appear to be a Zeiss image DICOM. You may need to read with the DCM class."
        )

    def decode_frame_payload(self, payload: bytes) -> tuple[np.ndarray | None, bool]:
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), flags=cv2.IMREAD_UNCHANGED)
        if frame is not None:
            return frame, True
        try:
            unscrambled_frame = self.unscramble_frame(payload)
        except Exception:
            return None, False
        frame = cv2.imdecode(np.frombuffer(unscrambled_frame, np.uint8), flags=cv2.IMREAD_UNCHANGED)
        return frame, True if frame is not None else False

    def normalize_fundus_frame(self, frame: np.ndarray, *, from_cv2: bool) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] >= 4:
            array = array[:, :, :4]

        if from_cv2 and array.ndim == 3:
            if array.shape[2] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            elif array.shape[2] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
                array = array[:, :, :3]
        elif array.ndim == 3 and array.shape[2] > 3:
            array = array[:, :, :3]

        return self.normalize_fundus_orientation(array)

    def build_base_metadata(self, ds: pydicom.dataset.FileDataset, *, source_kind: str) -> dict[str, Any]:
        metadata = {
            "source_file": str(self.filepath.name),
            "source_kind": source_kind,
            "manufacturer": clean_text(getattr(ds, "Manufacturer", "")),
            "manufacturer_model_name": clean_text(getattr(ds, "ManufacturerModelName", "")),
            "protocol_name": clean_text(getattr(ds, "ProtocolName", "")),
            "study_description": clean_text(getattr(ds, "StudyDescription", "")),
            "series_description": clean_text(getattr(ds, "SeriesDescription", "")),
            "sop_class_uid": clean_text(getattr(ds, "SOPClassUID", "")),
            "number_of_frames": parse_int(getattr(ds, "NumberOfFrames", None)),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
        }
        return {key: value for key, value in metadata.items() if value not in (None, "")}

    def make_oct_object(
        self,
        ds: pydicom.dataset.FileDataset,
        volume: np.ndarray,
        *,
        source_kind: str,
        volume_index: int,
    ) -> OCTVolumeWithMetaData:
        patient_name, first_name, surname = clean_person_name(getattr(ds, "PatientName", ""))
        metadata = self.build_base_metadata(ds, source_kind=source_kind)
        metadata["volume_index"] = volume_index
        return OCTVolumeWithMetaData(
            volume=volume,
            patient_id=clean_text(getattr(ds, "PatientID", "")),
            patient_name=patient_name or None,
            first_name=first_name or None,
            surname=surname or None,
            sex=clean_text(getattr(ds, "PatientSex", "")),
            patient_dob=clean_text(getattr(ds, "PatientBirthDate", "")) or None,
            acquisition_date=parse_dicom_datetime(ds),
            laterality=self.laterality(ds) or None,
            volume_id=f"{self.filepath.stem}:{source_kind}:{volume_index}",
            device_name=build_device_name(ds) or None,
            scan_pattern=clean_text(getattr(ds, "ProtocolName", "")) or clean_text(
                getattr(ds, "SeriesDescription", "")) or None,
            metadata=metadata,
        )

    def make_fundus_object(
        self,
        ds: pydicom.dataset.FileDataset,
        image: np.ndarray,
        *,
        source_kind: str,
        image_index: int,
    ) -> FundusImageWithMetaData:
        patient_name, _, _ = clean_person_name(getattr(ds, "PatientName", ""))
        metadata = self.build_base_metadata(ds, source_kind=source_kind)
        metadata["image_index"] = image_index
        fundus_image = FundusImageWithMetaData(
            image=image,
            patient_id=clean_text(getattr(ds, "PatientID", "")) or None,
            patient_name=patient_name or None,
            patient_dob=clean_text(getattr(ds, "PatientBirthDate", "")) or None,
            sex=clean_text(getattr(ds, "PatientSex", "")) or None,
            acquisition_date=parse_dicom_datetime(ds),
            laterality=self.laterality(ds) or None,
            image_id=f"{self.filepath.stem}:{source_kind}:{image_index}",
            device_name=build_device_name(ds) or None,
            scan_pattern=clean_text(getattr(ds, "ProtocolName", "")) or clean_text(
                getattr(ds, "SeriesDescription", "")) or None,
            metadata=metadata,
        )
        fundus_image.orientation_normalized = True
        return fundus_image

    def append_unique_oct(
        self,
        oct_outputs: list[OCTVolumeWithMetaData],
        seen_oct: set[tuple[tuple[int, ...], str, str]],
        ds: pydicom.dataset.FileDataset,
        volume: np.ndarray,
        *,
        source_kind: str,
    ) -> None:
        signature = self.pixel_signature(volume)
        if signature in seen_oct:
            return
        seen_oct.add(signature)
        oct_outputs.append(
            self.make_oct_object(ds, volume, source_kind=source_kind, volume_index=len(oct_outputs))
        )

    def append_unique_fundus(
        self,
        fundus_outputs: list[FundusImageWithMetaData],
        seen_fundus: set[tuple[tuple[int, ...], str, str]],
        ds: pydicom.dataset.FileDataset,
        image: np.ndarray,
        *,
        source_kind: str,
    ) -> None:
        signature = self.pixel_signature(image)
        if signature in seen_fundus:
            return
        seen_fundus.add(signature)
        fundus_outputs.append(
            self.make_fundus_object(ds, image, source_kind=source_kind, image_index=len(fundus_outputs))
        )

    def iter_private_sequences(self, ds: pydicom.dataset.FileDataset):
        for element in ds.iterall():
            if element.tag == PRIVATE_FRAME_SEQUENCE_TAG:
                value = getattr(element, "value", None)
                if isinstance(value, (list, tuple)):
                    yield value

    def decode_private_payloads(
        self,
        ds: pydicom.dataset.FileDataset,
        oct_outputs: list[OCTVolumeWithMetaData],
        fundus_outputs: list[FundusImageWithMetaData],
        seen_oct: set[tuple[tuple[int, ...], str, str]],
        seen_fundus: set[tuple[tuple[int, ...], str, str]],
    ) -> None:
        for sequence in self.iter_private_sequences(ds):
            decoded_frames: list[np.ndarray] = []
            for item in sequence:
                element = item.get(PRIVATE_FRAME_DATA_TAG)
                payload = getattr(element, "value", element)
                if payload is None:
                    continue
                frame, used_cv = self.decode_frame_payload(payload)
                if frame is None:
                    continue
                if frame.ndim == 2:
                    decoded_frames.append(frame)
                elif frame.ndim == 3:
                    decoded_frames.append(frame if not used_cv else frame)
            if not decoded_frames:
                continue

            if len(decoded_frames) == 1 or not self.is_volume_like(decoded_frames):
                for frame in decoded_frames:
                    normalized = self.normalize_fundus_frame(frame, from_cv2=True)
                    self.append_unique_fundus(
                        fundus_outputs,
                        seen_fundus,
                        ds,
                        normalized,
                        source_kind="private_tag",
                    )
                continue

            normalized_volume = self.normalize_oct_orientation(
                [self.frame_to_gray(frame) for frame in decoded_frames]
            )
            self.append_unique_oct(
                oct_outputs,
                seen_oct,
                ds,
                normalized_volume,
                source_kind="private_tag",
            )

    def decode_pixel_data(
        self,
        ds: pydicom.dataset.FileDataset,
        oct_outputs: list[OCTVolumeWithMetaData],
        fundus_outputs: list[FundusImageWithMetaData],
        seen_oct: set[tuple[tuple[int, ...], str, str]],
        seen_fundus: set[tuple[tuple[int, ...], str, str]],
    ) -> None:
        if "PixelData" not in ds:
            return

        num_frames = parse_int(getattr(ds, "NumberOfFrames", None))
        decoded_frames: list[np.ndarray] = []
        if num_frames is not None and num_frames > 0:
            try:
                payloads = list(generate_pixel_data_frame(ds.PixelData, num_frames))
            except Exception:
                payloads = []
            for payload in payloads:
                frame, _ = self.decode_frame_payload(payload)
                if frame is not None:
                    decoded_frames.append(frame)

        if decoded_frames:
            sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))
            if sop_class_uid == OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID or not self.is_volume_like(decoded_frames):
                for frame in decoded_frames:
                    normalized = self.normalize_fundus_frame(frame, from_cv2=True)
                    self.append_unique_fundus(
                        fundus_outputs,
                        seen_fundus,
                        ds,
                        normalized,
                        source_kind="pixel_data",
                    )
                return

            normalized_volume = self.normalize_oct_orientation(
                [self.frame_to_gray(frame) for frame in decoded_frames]
            )
            self.append_unique_oct(
                oct_outputs,
                seen_oct,
                ds,
                normalized_volume,
                source_kind="pixel_data",
            )
            return

        frame, _ = self.decode_frame_payload(ds.PixelData)
        if frame is not None:
            normalized = self.normalize_fundus_frame(frame, from_cv2=True)
            self.append_unique_fundus(
                fundus_outputs,
                seen_fundus,
                ds,
                normalized,
                source_kind="pixel_data",
            )
            return

        try:
            pixel_array = ds.pixel_array
        except Exception:
            return

        array = np.asarray(pixel_array)
        sop_class_uid = clean_text(getattr(ds, "SOPClassUID", ""))

        if array.ndim == 2:
            normalized = self.normalize_fundus_frame(array, from_cv2=False)
            self.append_unique_fundus(
                fundus_outputs,
                seen_fundus,
                ds,
                normalized,
                source_kind="pixel_array",
            )
            return

        if array.ndim == 3 and array.shape[-1] in {3, 4}:
            normalized = self.normalize_fundus_frame(array, from_cv2=False)
            self.append_unique_fundus(
                fundus_outputs,
                seen_fundus,
                ds,
                normalized,
                source_kind="pixel_array",
            )
            return

        if array.ndim == 3:
            frames = [array[index] for index in range(array.shape[0])]
            if sop_class_uid == OPHTHALMIC_PHOTOGRAPHY_SOP_CLASS_UID or not self.is_volume_like(frames):
                for frame in frames:
                    normalized = self.normalize_fundus_frame(frame, from_cv2=False)
                    self.append_unique_fundus(
                        fundus_outputs,
                        seen_fundus,
                        ds,
                        normalized,
                        source_kind="pixel_array",
                    )
                return

            normalized_volume = self.normalize_oct_orientation([self.frame_to_gray(frame) for frame in frames])
            self.append_unique_oct(
                oct_outputs,
                seen_oct,
                ds,
                normalized_volume,
                source_kind="pixel_array",
            )
            return

        if array.ndim == 4:
            normalized_volume = self.normalize_oct_orientation(array)
            self.append_unique_oct(
                oct_outputs,
                seen_oct,
                ds,
                normalized_volume,
                source_kind="pixel_array",
            )

    def read_data(self):
        """Read Zeiss OCT volumes and fundus images from one DICOM file."""
        ds = self.read_dataset()
        self.validate_dataset(ds)

        oct_outputs: list[OCTVolumeWithMetaData] = []
        fundus_outputs: list[FundusImageWithMetaData] = []
        seen_oct: set[tuple[tuple[int, ...], str, str]] = set()
        seen_fundus: set[tuple[tuple[int, ...], str, str]] = set()

        self.decode_private_payloads(ds, oct_outputs, fundus_outputs, seen_oct, seen_fundus)
        self.decode_pixel_data(ds, oct_outputs, fundus_outputs, seen_oct, seen_fundus)

        return oct_outputs, fundus_outputs

    def unscramble_frame(self, frame: bytes) -> bytearray:
        """Return an unscrambled image frame for Zeiss private JPEG2000 payloads."""
        frame = bytearray(frame)
        for ii in range(0, len(frame), 7):
            frame[ii] = frame[ii] ^ 0x5A

        jp2_offset = math.floor(len(frame) / 5 * 3)
        offset = frame.find(b"\x00\x00\x00\x0C")
        if offset == -1:
            raise ValueError("No JP2 header found in the scrambled pixel data")

        if jp2_offset != offset:
            print(
                f"JP2 header found at offset {offset} rather than the expected {jp2_offset}"
            )
            jp2_offset = offset

        data = bytearray()
        data.extend(frame[jp2_offset: jp2_offset + 253])
        data.extend(frame[993:1016])
        data.extend(frame[276:763])
        data.extend(frame[23:276])
        data.extend(frame[1016:jp2_offset])
        data.extend(frame[:23])
        data.extend(frame[763:993])
        data.extend(frame[jp2_offset + 253:])

        assert len(data) == len(frame)
        return data
