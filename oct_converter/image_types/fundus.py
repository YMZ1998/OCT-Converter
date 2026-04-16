from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oct_converter.image_types.metadata_types import (
    DeviceInfo,
    FundusMetadataModel,
    ImageGeometry,
    PatientInfo,
    SeriesInfo,
    SourceInfo,
)
from oct_converter.image_types.write_image import cv2_imwrite_safe

VIDEO_TYPES = [
    ".avi",
    ".mp4",
]
IMAGE_TYPES = [".png", ".bmp", ".tiff", ".jpg", ".jpeg"]


class FundusImageWithMetaData(object):
    """Class to hold a fundus image and any related metadata.

    Also provides methods for viewing and saving.

    Attributes:
        image: fundus image.
        laterality: left or right eye.
        patient_id: patient ID.
        patient_name: patient full name.
        image_id: image ID.
        DOB: patient date of birth.
        sex: patient sex.
        device_name: device / scanner name.
        scan_pattern: scan pattern or protocol label.
        metadata: all metadata parsed from the original file.
        pixel_spacing: [x, y] pixel spacing in mm
    """

    def __init__(
        self,
        image: np.array,
        laterality: str | None = None,
        patient_id: str | None = None,
        image_id: str | None = None,
        patient_dob: str | None = None,
        acquisition_date: str | None = None,
        metadata: dict | None = None,
        pixel_spacing: list[float] | None = None,
        patient_name: str | None = None,
        sex: str | None = None,
        device_name: str | None = None,
        scan_pattern: str | None = None,
        metadata_model: FundusMetadataModel | None = None,
    ) -> None:
        self.image = image
        self.meta = metadata_model or FundusMetadataModel(
            patient=PatientInfo(
                patient_id=patient_id,
                patient_name=patient_name,
                sex=sex,
                patient_dob=patient_dob,
            ),
            series=SeriesInfo(
                image_id=image_id,
                acquisition_date=acquisition_date,
                laterality=laterality,
                scan_pattern=scan_pattern,
            ),
            device=DeviceInfo(device_name=device_name),
            geometry=ImageGeometry(pixel_spacing=pixel_spacing),
            metadata=metadata,
        )

    @property
    def source(self) -> SourceInfo:
        return self.meta.source

    @property
    def patient(self) -> PatientInfo:
        return self.meta.patient

    @property
    def series(self) -> SeriesInfo:
        return self.meta.series

    @property
    def device(self) -> DeviceInfo:
        return self.meta.device

    @property
    def geometry(self) -> ImageGeometry:
        return self.meta.geometry

    @property
    def laterality(self) -> str | None:
        return self.series.laterality

    @laterality.setter
    def laterality(self, value: str | None) -> None:
        self.series.laterality = value

    @property
    def patient_id(self) -> str | None:
        return self.patient.patient_id

    @patient_id.setter
    def patient_id(self, value: str | None) -> None:
        self.patient.patient_id = value

    @property
    def patient_name(self) -> str | None:
        return self.patient.patient_name

    @patient_name.setter
    def patient_name(self, value: str | None) -> None:
        self.patient.patient_name = value

    @property
    def image_id(self) -> str | None:
        return self.series.image_id

    @image_id.setter
    def image_id(self, value: str | None) -> None:
        self.series.image_id = value

    @property
    def patient_dob(self) -> Any | None:
        return self.patient.patient_dob

    @patient_dob.setter
    def patient_dob(self, value: Any | None) -> None:
        self.patient.patient_dob = value

    @property
    def DOB(self) -> Any | None:
        return self.patient.patient_dob

    @DOB.setter
    def DOB(self, value: Any | None) -> None:
        self.patient.patient_dob = value

    @property
    def acquisition_date(self) -> Any | None:
        return self.series.acquisition_date

    @acquisition_date.setter
    def acquisition_date(self, value: Any | None) -> None:
        self.series.acquisition_date = value

    @property
    def metadata(self) -> dict | None:
        return self.meta.metadata

    @metadata.setter
    def metadata(self, value: dict | None) -> None:
        self.meta.metadata = value

    @property
    def pixel_spacing(self) -> list[float] | tuple[float, ...] | None:
        return self.geometry.pixel_spacing

    @pixel_spacing.setter
    def pixel_spacing(self, value: list[float] | tuple[float, ...] | None) -> None:
        self.geometry.pixel_spacing = value

    @property
    def sex(self) -> str | None:
        return self.patient.sex

    @sex.setter
    def sex(self, value: str | None) -> None:
        self.patient.sex = value

    @property
    def device_name(self) -> str | None:
        return self.device.device_name

    @device_name.setter
    def device_name(self, value: str | None) -> None:
        self.device.device_name = value

    @property
    def scan_pattern(self) -> str | None:
        return self.series.scan_pattern

    @scan_pattern.setter
    def scan_pattern(self, value: str | None) -> None:
        self.series.scan_pattern = value

    def save(self, filepath: str | Path) -> None:
        """Saves fundus image.

        Args:
            filepath: location to save volume to. Extension must be in IMAGE_TYPES.
        """
        extension = Path(filepath).suffix
        print(self.image.shape)
        if extension.lower() in IMAGE_TYPES:
            # change channel order from RGB to BGR and save with cv2
            image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            cv2_imwrite_safe(filepath, image)
            # cv2.imwrite(filepath, image)
        elif extension.lower() == ".npy":
            np.save(filepath, self.image)
        else:
            raise NotImplementedError(
                "Saving with file extension {} not supported".format(extension)
            )
