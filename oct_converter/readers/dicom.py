from __future__ import annotations

from pathlib import Path

from oct_converter.image_types import (
    DeviceInfo,
    ImageGeometry,
    OCTMetadataModel,
    OCTVolumeWithMetaData,
    SourceInfo,
)


class Dicom(object):
    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

    def read_oct_volume(self) -> OCTVolumeWithMetaData:
        """Reads OCT data.

        Returns:
            OCTVolumeWithMetaData
        """
        import pydicom

        dicom_data = pydicom.dcmread(self.filepath)
        if dicom_data.Manufacturer.startswith("Carl Zeiss Meditec"):
            raise ValueError(
                "This appears to be a Zeiss DCM. You may need to read with the ZEISSDCM class."
            )
        pixel_data = dicom_data.pixel_array
        oct_volume = OCTVolumeWithMetaData(
            volume=pixel_data,
            metadata_model=OCTMetadataModel(
                source=SourceInfo(
                    vendor=getattr(dicom_data, "Manufacturer", None),
                    file_format="DICOM",
                    filepath=self.filepath,
                ),
                device=DeviceInfo(
                    vendor=getattr(dicom_data, "Manufacturer", None),
                    device_name=getattr(dicom_data, "ManufacturerModelName", None),
                ),
                geometry=ImageGeometry(
                    pixel_spacing=list(getattr(dicom_data, "PixelSpacing", [])) or None
                ),
            ),
        )
        return oct_volume
