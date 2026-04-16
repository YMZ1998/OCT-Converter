from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from oct_converter.image_types.metadata_types import (
    DeviceInfo,
    ImageGeometry,
    OCTMetadataModel,
    PatientInfo,
    SeriesInfo,
    SourceInfo,
)
from oct_converter.image_types.write_image import cv2_imwrite_safe

VIDEO_TYPES = [
    ".avi",
    ".mp4",
]
IMAGE_TYPES = [".png", ".bmp", ".jpg", ".jpeg"]


class OCTVolumeWithMetaData(object):
    """Class to hold an OCT volume and any related metadata.

    Also provides methods for viewing and saving.

    Attributes:
        volume: all the volume's b-scans.

        patient_id: patient ID.
        patient_name: patient full name.
        first_name: patient first name.
        surname: patient second name.
        sex: patient sex.
        DOB: patient date of birth.

        volume_id: volume ID.
        acquisition_date: date image acquired.
        num_slices: number of b-scans present in volume.
        laterality: left or right eye.

        contours: contours data.
        pixel_spacing: (x, y, z) pixel spacing in mm.
        device_name: device / scanner name.
        scan_pattern: scan pattern or protocol label.
        metadata: all metadata available in the OCT scan.
    """

    def __init__(
        self,
        volume: list[np.array],
        patient_id: str | None = None,
        first_name: str | None = None,
        surname: str | None = None,
        sex: str | None = None,
        patient_dob: str | None = None,
        volume_id: str | None = None,
        acquisition_date: datetime | None = None,
        laterality: str | None = None,
        contours: dict | None = None,
        pixel_spacing: list[float] | None = None,
        metadata: dict | None = None,
        header: dict | None = None,
        oct_header: dict | None = None,
        patient_name: str | None = None,
        device_name: str | None = None,
        scan_pattern: str | None = None,
        metadata_model: OCTMetadataModel | None = None,
    ) -> None:
        # image
        self.volume = volume
        self.meta = metadata_model or OCTMetadataModel(
            patient=PatientInfo(
                patient_id=patient_id,
                patient_name=patient_name,
                first_name=first_name,
                surname=surname,
                sex=sex,
                patient_dob=patient_dob,
            ),
            series=SeriesInfo(
                volume_id=volume_id,
                acquisition_date=acquisition_date,
                laterality=laterality,
                scan_pattern=scan_pattern,
            ),
            device=DeviceInfo(device_name=device_name),
            geometry=ImageGeometry(pixel_spacing=pixel_spacing),
            metadata=metadata,
            header=header,
            oct_header=oct_header,
            contours=contours,
        )
        self.num_slices = len(self.volume)

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
    def first_name(self) -> str | None:
        return self.patient.first_name

    @first_name.setter
    def first_name(self, value: str | None) -> None:
        self.patient.first_name = value

    @property
    def surname(self) -> str | None:
        return self.patient.surname

    @surname.setter
    def surname(self, value: str | None) -> None:
        self.patient.surname = value

    @property
    def sex(self) -> str | None:
        return self.patient.sex

    @sex.setter
    def sex(self, value: str | None) -> None:
        self.patient.sex = value

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
    def volume_id(self) -> str | None:
        return self.series.volume_id

    @volume_id.setter
    def volume_id(self, value: str | None) -> None:
        self.series.volume_id = value

    @property
    def acquisition_date(self) -> datetime | Any | None:
        return self.series.acquisition_date

    @acquisition_date.setter
    def acquisition_date(self, value: datetime | Any | None) -> None:
        self.series.acquisition_date = value

    @property
    def laterality(self) -> str | None:
        return self.series.laterality

    @laterality.setter
    def laterality(self, value: str | None) -> None:
        self.series.laterality = value

    @property
    def scan_pattern(self) -> str | None:
        return self.series.scan_pattern

    @scan_pattern.setter
    def scan_pattern(self, value: str | None) -> None:
        self.series.scan_pattern = value

    @property
    def contours(self) -> dict | None:
        return self.meta.contours

    @contours.setter
    def contours(self, value: dict | None) -> None:
        self.meta.contours = value

    @property
    def pixel_spacing(self) -> list[float] | tuple[float, ...] | None:
        return self.geometry.pixel_spacing

    @pixel_spacing.setter
    def pixel_spacing(self, value: list[float] | tuple[float, ...] | None) -> None:
        self.geometry.pixel_spacing = value

    @property
    def device_name(self) -> str | None:
        return self.device.device_name

    @device_name.setter
    def device_name(self, value: str | None) -> None:
        self.device.device_name = value

    @property
    def metadata(self) -> dict | None:
        return self.meta.metadata

    @metadata.setter
    def metadata(self, value: dict | None) -> None:
        self.meta.metadata = value

    @property
    def header(self) -> dict | None:
        return self.meta.header

    @header.setter
    def header(self, value: dict | None) -> None:
        self.meta.header = value

    @property
    def oct_header(self) -> dict | None:
        return self.meta.oct_header

    @oct_header.setter
    def oct_header(self, value: dict | None) -> None:
        self.meta.oct_header = value

    def peek(
        self,
        rows: int = 5,
        cols: int = 5,
        filepath: str | Path | None = None,
        show_contours: bool | None = False,
    ) -> None:
        """Plots a montage of the OCT volume. Optionally saves the plot if a filepath is provided.

        Args:
            rows: number of rows in the plot.
            cols: number of columns in the plot.
            filepath: location to save montage to.
            show_contours: if set to ``True``, will plot contours on the OCT volume.
        """
        images = rows * cols
        x_size = rows * self.volume[0].shape[0]
        y_size = cols * self.volume[0].shape[1]
        ratio = y_size / x_size
        slices_indices = np.linspace(0, self.num_slices - 1, images).astype(np.int16)
        plt.figure(figsize=(12 * ratio, 12))
        for i in range(images):
            slice_id = slices_indices[i]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(self.volume[slice_id], cmap="gray")
            if show_contours and self.contours is not None:
                for v in self.contours.values():
                    if (
                        slice_id < len(v)
                        and v[slice_id] is not None
                        and not np.isnan(v[slice_id]).all()
                    ):
                        plt.plot(v[slice_id], color="r")
            plt.axis("off")
            plt.title("{}".format(slice_id))
        plt.suptitle("OCT volume with {} slices.".format(self.num_slices))

        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

    def save(self, filepath: str | Path) -> None:
        """Saves OCT volume as a video or stack of slices.

        Args:
            filepath: location to save volume to. Extension must be in VIDEO_TYPES or IMAGE_TYPES.
        """
        extension = Path(filepath).suffix
        if extension.lower() in VIDEO_TYPES:
            video_writer = imageio.get_writer(filepath, macro_block_size=None)
            for slice in self.volume:
                # print(slice.min(), slice.max())
                # slice *= 255.0 / slice.max()
                slice = slice.astype("uint8")
                video_writer.append_data(slice)
            video_writer.close()
        elif extension.lower() in {".tif", ".tiff"}:

            pages = []

            for slice in self.volume:
                # print(slice.min(), slice.max())
                slice = slice.astype(np.float32)
                if slice.max() > 0:
                    slice = slice * (255.0 / slice.max())

                slice = np.clip(slice, 0, 255).astype(np.uint8)

                pages.append(slice)

            pages = np.stack(pages, axis=0)

            tiff.imwrite(
                filepath,
                pages,
                photometric="minisblack"
            )

        elif extension.lower() in IMAGE_TYPES:
            base = Path(filepath).stem
            print(
                "Saving OCT as sequential slices {}_[1..{}]{}".format(
                    base, len(self.volume), extension
                )
            )
            full_base = Path(filepath).with_suffix("")
            self.volume = np.array(self.volume).astype("float64")
            self.volume *= 255.0 / self.volume.max()
            for index, slice in enumerate(self.volume):
                filename = "{}_{}{}".format(full_base, index, extension)
                cv2_imwrite_safe(filename, slice)
        elif extension.lower() == ".npy":
            np.save(filepath, self.volume)
        else:
            raise NotImplementedError(
                "Saving with file extension {} not supported".format(extension)
            )

    def get_projection(self) -> np.array:
        """Produces a 2D projection image from the volume."""
        projection = np.mean(self.volume, axis=1)
        return projection

    def save_projection(self, filepath: str | Path) -> None:
        """Save a 2D projection image from the volume.

        Args:
            filepath: location to save volume to. Extension must be in IMAGE_TYPES.
        """
        extension = Path(filepath).suffix
        if extension.lower() in IMAGE_TYPES:
            projection = self.get_projection()
            projection = 255 * projection / projection.max()
            cv2.imwrite(filepath, projection.astype(int))
        else:
            raise NotImplementedError(
                "Saving with file extension {} not supported".format(extension)
            )
