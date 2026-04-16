from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceInfo:
    vendor: str | None = None
    file_format: str | None = None
    filepath: Path | None = None


@dataclass
class PatientInfo:
    patient_id: str | None = None
    patient_name: str | None = None
    first_name: str | None = None
    surname: str | None = None
    sex: str | None = None
    patient_dob: Any | None = None


@dataclass
class SeriesInfo:
    volume_id: str | None = None
    image_id: str | None = None
    acquisition_date: Any | None = None
    laterality: str | None = None
    scan_pattern: str | None = None


@dataclass
class DeviceInfo:
    vendor: str | None = None
    device_name: str | None = None


@dataclass
class ImageGeometry:
    pixel_spacing: list[float] | tuple[float, ...] | None = None


@dataclass
class BaseMetadataModel:
    source: SourceInfo = field(default_factory=SourceInfo)
    patient: PatientInfo = field(default_factory=PatientInfo)
    series: SeriesInfo = field(default_factory=SeriesInfo)
    device: DeviceInfo = field(default_factory=DeviceInfo)
    geometry: ImageGeometry = field(default_factory=ImageGeometry)
    metadata: dict[str, Any] | None = None


@dataclass
class OCTMetadataModel(BaseMetadataModel):
    header: dict[str, Any] | None = None
    oct_header: dict[str, Any] | None = None
    contours: dict[str, Any] | None = None


@dataclass
class FundusMetadataModel(BaseMetadataModel):
    pass
