from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData

Point = tuple[float, float]
ScanSegment = tuple[Point, Point]
ScanBounds = tuple[float, float, float, float]


@dataclass
class VendorOverlayData:
    matched_fundus_index: int = -1
    matched_fundus_label: str = ""
    fundus_match_mode: str = ""
    overlay_mode: str = ""
    projection_mode: str = ""
    localizer_mode: str = ""
    warning: str = ""
    scan_segments: list[ScanSegment] = field(default_factory=list)
    bounds: ScanBounds | None = None


@dataclass
class VendorDatasetBase:
    vendor: str

    @property
    def volumes(self) -> list[OCTVolumeWithMetaData]:
        raise NotImplementedError

    @property
    def fundus_images(self) -> list[FundusImageWithMetaData]:
        raise NotImplementedError

    @property
    def overlays(self) -> list[VendorOverlayData]:
        return []


@dataclass
class ShiweiOCTDataset(VendorDatasetBase):
    input_dir: str
    volume: OCTVolumeWithMetaData
    fundus: FundusImageWithMetaData
    coordinates: list[np.ndarray]
    angles: list[float]
    segments: list[ScanSegment]
    segmentation_surfaces: np.ndarray | None = None
    segmentation_orientation: dict[str, Any] | None = None
    bscan_file: str | None = None
    fundus_file: str | None = None
    seg_file: str | None = None
    overlay_entries: list[VendorOverlayData] = field(default_factory=list)

    def __init__(
        self,
        input_dir: str,
        volume: OCTVolumeWithMetaData,
        fundus: FundusImageWithMetaData,
        coordinates: list[np.ndarray],
        angles: list[float],
        segments: list[ScanSegment],
        segmentation_surfaces: np.ndarray | None = None,
        segmentation_orientation: dict[str, Any] | None = None,
        bscan_file: str | None = None,
        fundus_file: str | None = None,
        seg_file: str | None = None,
        overlay_entries: list[VendorOverlayData] | None = None,
    ) -> None:
        super().__init__(vendor="shiwei")
        self.input_dir = input_dir
        self.volume = volume
        self.fundus = fundus
        self.coordinates = coordinates
        self.angles = angles
        self.segments = segments
        self.segmentation_surfaces = segmentation_surfaces
        self.segmentation_orientation = segmentation_orientation
        self.bscan_file = bscan_file
        self.fundus_file = fundus_file
        self.seg_file = seg_file
        self.overlay_entries = list(overlay_entries or [])

    @property
    def volumes(self) -> list[OCTVolumeWithMetaData]:
        return [self.volume]

    @property
    def fundus_images(self) -> list[FundusImageWithMetaData]:
        return [self.fundus]

    @property
    def overlays(self) -> list[VendorOverlayData]:
        return self.overlay_entries


@dataclass
class TupaiOCTDataset(VendorDatasetBase):
    dataset_dir: str
    volume: OCTVolumeWithMetaData
    fundus: FundusImageWithMetaData
    segments: list[ScanSegment]
    segment_coordinates: list[ScanSegment]
    coordinate_mode: str | None = None
    segment_mode: str | None = None
    scan_band_width_pixels: float | None = None
    overlay_entries: list[VendorOverlayData] = field(default_factory=list)

    def __init__(
        self,
        dataset_dir: str,
        volume: OCTVolumeWithMetaData,
        fundus: FundusImageWithMetaData,
        segments: list[ScanSegment],
        segment_coordinates: list[ScanSegment],
        coordinate_mode: str | None = None,
        segment_mode: str | None = None,
        scan_band_width_pixels: float | None = None,
        overlay_entries: list[VendorOverlayData] | None = None,
    ) -> None:
        super().__init__(vendor="tupai")
        self.dataset_dir = dataset_dir
        self.volume = volume
        self.fundus = fundus
        self.segments = segments
        self.segment_coordinates = segment_coordinates
        self.coordinate_mode = coordinate_mode
        self.segment_mode = segment_mode
        self.scan_band_width_pixels = scan_band_width_pixels
        self.overlay_entries = list(overlay_entries or [])

    @property
    def volumes(self) -> list[OCTVolumeWithMetaData]:
        return [self.volume]

    @property
    def fundus_images(self) -> list[FundusImageWithMetaData]:
        return [self.fundus]

    @property
    def overlays(self) -> list[VendorOverlayData]:
        return self.overlay_entries


@dataclass
class ZeissOCTDataset(VendorDatasetBase):
    input_path: str
    exam_dirs: list[str]
    volume_entries: list[OCTVolumeWithMetaData]
    fundus_entries: list[FundusImageWithMetaData]
    overlay_entries: list[VendorOverlayData]

    def __init__(
        self,
        input_path: str,
        exam_dirs: list[str],
        volume_entries: list[OCTVolumeWithMetaData],
        fundus_entries: list[FundusImageWithMetaData],
        overlay_entries: list[VendorOverlayData],
    ) -> None:
        super().__init__(vendor="zeiss")
        self.input_path = input_path
        self.exam_dirs = exam_dirs
        self.volume_entries = volume_entries
        self.fundus_entries = fundus_entries
        self.overlay_entries = overlay_entries

    @property
    def volumes(self) -> list[OCTVolumeWithMetaData]:
        return self.volume_entries

    @property
    def fundus_images(self) -> list[FundusImageWithMetaData]:
        return self.fundus_entries

    @property
    def overlays(self) -> list[VendorOverlayData]:
        return self.overlay_entries
