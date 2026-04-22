# -*- coding: utf-8 -*-
"""
Zeiss Cirrus OCTA DICOM to TIFF Converter
==========================================

Converts Zeiss Cirrus HD-OCT OCTA DICOM files to TIFF format for Imaris.

Features:
- Handles corrupted DICOM metadata (common in Zeiss exports)
- Automatically detects and fixes dimension errors (Columns/Frames swapped)
- Supports JPEG 2000 compressed DICOM files
- Adapts to different scan resolutions (245x245x1024, 490x490x1024, etc.)
- Generates preview images and metadata files
- Selects best quality volume based on vessel signal

Usage:
    python Zeiss_OCTA_Converter.py <folder_name>

    Example: python Zeiss_OCTA_Converter.py HenkE433

Output:
    - OCTA_<folder>.tif       : 3D TIFF file for Imaris
    - OCTA_<folder>.npy       : NumPy array
    - OCTA_<folder>_metadata.json : Scan parameters and voxel size
    - OCTA_<folder>_Preview.png  : Visualization (MIP projections)

Author: Automated conversion script for UCSF OCTA analysis
Date: 2025-11-12
"""

import pydicom
import numpy as np
from pathlib import Path
import warnings
import json
import sys

# UTF-8 output for Windows
if sys.platform == 'win32':
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings('ignore')


def fix_dicom_metadata(dcm):
    """
    Fix corrupted DICOM metadata commonly found in Zeiss Cirrus exports.

    Common issues:
    - PhotometricInterpretation contains garbage after 'MONOCHROME2'
    - NumberOfFrames contains null bytes
    - Rows/Columns contain null bytes or invalid characters
    """
    try:
        # Fix PhotometricInterpretation (critical for decompression)
        if hasattr(dcm, 'PhotometricInterpretation'):
            photo_str = str(dcm.PhotometricInterpretation)
            if 'MONOCHROME2' in photo_str and photo_str != 'MONOCHROME2':
                dcm.PhotometricInterpretation = 'MONOCHROME2'
            elif 'MONOCHROME1' in photo_str and photo_str != 'MONOCHROME1':
                dcm.PhotometricInterpretation = 'MONOCHROME1'

        # Fix NumberOfFrames
        if hasattr(dcm, 'NumberOfFrames'):
            frames_str = str(dcm.NumberOfFrames)
            if '\x00' in frames_str or not frames_str.isdigit():
                clean_number = ''.join(c for c in frames_str if c.isdigit())
                if clean_number:
                    dcm.NumberOfFrames = int(clean_number)

        # Fix Rows
        if hasattr(dcm, 'Rows'):
            rows_str = str(dcm.Rows)
            if '\x00' in rows_str or not str(rows_str).isdigit():
                clean_number = ''.join(c for c in rows_str if c.isdigit())
                if clean_number:
                    dcm.Rows = int(clean_number)

        # Fix Columns
        if hasattr(dcm, 'Columns'):
            cols_str = str(dcm.Columns)
            if '\x00' in cols_str or not str(cols_str).isdigit():
                clean_number = ''.join(c for c in cols_str if c.isdigit())
                if clean_number:
                    dcm.Columns = int(clean_number)

    except Exception as e:
        print(f"  Warning: Metadata fix error: {e}")

    return dcm


def read_dicom_robust(file_path):
    """
    Robustly read Zeiss OCTA DICOM file with error handling.

    Handles:
    - Corrupted metadata
    - JPEG 2000 compression
    - Dimension errors (Columns/Frames swapped)
    """
    try:
        dcm = pydicom.dcmread(str(file_path), force=True)

        if not hasattr(dcm, 'PixelData'):
            return None, None

        # Fix metadata before decompression
        dcm = fix_dicom_metadata(dcm)

        # Decompress (JPEG 2000)
        try:
            dcm.decompress()
            image = dcm.pixel_array
        except Exception as e:
            print(f"  Decompression failed: {e}")
            return None, None

        # Check for dimension errors (common in Zeiss exports)
        # OCTA data should be (Y, X, Z) where Z (depth) is typically 1024
        if len(image.shape) == 3:
            # Ensure depth dimension (should be ~1024) is last
            # If middle dimension is largest, transpose to move it to end
            if image.shape[1] > image.shape[2] and image.shape[1] > image.shape[0]:
                print(f"  Detected dimension error: swapping X and Z axes")
                print(f"    Original shape: {image.shape} → ", end='')
                image = np.transpose(image, (0, 2, 1))
                print(f"Fixed shape: {image.shape}")

        return image, dcm

    except Exception as e:
        print(f"  Read error: {e}")
        return None, None


def calculate_voxel_size(image_shape):
    """
    Calculate voxel size based on image dimensions.

    Assumptions for Zeiss Cirrus OCTA:
    - Depth (Z) is always 2.0 mm (typically 1024 pixels)
    - Width (X/Y) varies by scan protocol:
        - 3x3 mm for ~245x245 scans
        - 4x4 mm for ~320x320 scans
        - 5x5 mm for ~400x400 scans
        - 6x6 mm for ~490x490 scans
        - 8x8 mm for ~640x640 scans
        - 9x9 mm for ~700x700 scans
        - 12x12 mm for ~980x980 scans
    """
    rows, cols, depth_pixels = image_shape

    # Estimate scan width based on lateral resolution
    # Using approximate 12.2 µm/pixel resolution as baseline
    if rows <= 250:
        scan_width_mm = 3.0  # 245 pixels
    elif rows <= 350:
        scan_width_mm = 4.0  # ~320 pixels
    elif rows <= 450:
        scan_width_mm = 5.0  # ~400 pixels
    elif rows <= 550:
        scan_width_mm = 6.0  # ~490 pixels
    elif rows <= 700:
        scan_width_mm = 8.0  # ~640 pixels
    elif rows <= 850:
        scan_width_mm = 9.0  # ~730 pixels
    elif rows <= 1000:
        scan_width_mm = 12.0  # ~980 pixels
    else:
        # For very large scans, estimate based on ~12 µm resolution
        scan_width_mm = round((rows * 0.012) / 0.5) * 0.5  # Round to nearest 0.5mm

    scan_depth_mm = 2.0

    voxel_x = (scan_width_mm / cols) * 1000  # µm
    voxel_y = (scan_width_mm / rows) * 1000  # µm
    voxel_z = (scan_depth_mm / depth_pixels) * 1000  # µm

    return voxel_x, voxel_y, voxel_z, scan_width_mm, scan_depth_mm


def select_best_volume(all_data):
    """
    Select the best volume from multiple files.

    For OCTA angiography, look for:
    - Proper 3D volume (not 2D slices)
    - Good contrast (not too uniform)
    - Clear vessel signal
    """
    # Group by shape
    shape_groups = {}
    for img, dcm, name in all_data:
        if len(img.shape) != 3:
            continue
        shape = img.shape
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append((img, dcm, name))

    if not shape_groups:
        return None, None, None

    # Select most common shape
    target_shape, matching_files = max(shape_groups.items(), key=lambda x: len(x[1]))

    print(f"\nFound {len(matching_files)} files with shape {target_shape}")

    # Analyze each file
    print("\nAnalyzing files:")
    scores = []
    for i, (img, dcm, name) in enumerate(matching_files, 1):
        # Convert to uint8
        if img.dtype == np.int8:
            img_uint8 = img.astype(np.int16) + 128
            img_uint8 = img_uint8.astype(np.uint8)
        else:
            img_uint8 = ((img.astype(np.float32) - img.min()) /
                         (img.max() - img.min()) * 255).astype(np.uint8)

        # Calculate quality metrics
        mean_val = img.mean()
        std_val = img.std()

        # MIP analysis
        mip_z = np.max(img_uint8, axis=2)
        contrast = mip_z.std()

        # Score: prefer good contrast and reasonable mean
        score = contrast

        print(f"  File {i}: {name}")
        print(f"    Mean: {mean_val:.1f}, Std: {std_val:.1f}, Contrast: {contrast:.1f}")

        scores.append((score, img, dcm, name))

    # Select best
    scores.sort(key=lambda x: x[0], reverse=True)
    _, best_img, best_dcm, best_name = scores[0]

    print(f"\nSelected: {best_name} (highest contrast)")

    return best_img, best_dcm, best_name


def main():
    print("\n" + "=" * 80)
    print("Zeiss Cirrus OCTA DICOM to TIFF Converter")
    print("=" * 80)
    print("\nUsage: python Zeiss_OCTA_Converter.py <folder_name>")
    print("\nExample: python Zeiss_OCTA_Converter.py HenkE433")
    print("\nThe script will:")
    print("  1. Read all DICOM files in the folder")
    print("  2. Fix corrupted metadata and decompress JPEG 2000")
    print("  3. Select the best volume")
    print("  4. Export to TIFF, NPY, and generate preview")
    print("=" * 80 + "\n")

    folder_name = r"E:\Data\OCT\蔡司OCT\DataFiles\E195"

    print("\n" + "=" * 80)
    print("Zeiss Cirrus OCTA DICOM to TIFF Converter")
    print("=" * 80 + "\n")

    # Find data folder
    script_dir = Path(__file__).parent
    possible_locations = [
        script_dir / "DataFiles" / folder_name,
        script_dir.parent / "HenkOCTA_DataFiles" / folder_name,
        Path.cwd() / "HenkOCTA_DataFiles" / folder_name,
        Path.cwd() / folder_name,
    ]

    data_folder = None
    for loc in possible_locations:
        if loc.exists():
            data_folder = loc
            print(f"Data folder: {data_folder}")
            break

    if data_folder is None:
        print(f"ERROR: Folder '{folder_name}' not found in:")
        for loc in possible_locations:
            print(f"  - {loc}")
        return False

    # Read DICOM files
    dcm_files = sorted(data_folder.glob("*.DCM"))
    dcm_files = [f for f in dcm_files if f.name != "DICOMDIR"]
    print(f"Found {len(dcm_files)} DICOM files\n")

    if len(dcm_files) == 0:
        print("ERROR: No DICOM files found!")
        return False

    # Read all files
    print("Reading files...")
    all_data = []

    for i, file_path in enumerate(dcm_files, 1):
        print(f"\n[{i}/{len(dcm_files)}] {file_path.name}")

        image, dcm = read_dicom_robust(file_path)

        if image is None:
            continue

        print(f"  ✓ Shape: {image.shape}, Dtype: {image.dtype}")
        all_data.append((image, dcm, file_path.name))

    if len(all_data) == 0:
        print("\nERROR: No valid volumes could be read!")
        return False

    print(f"\n{'=' * 80}")
    print(f"Successfully read {len(all_data)} volumes")
    print('=' * 80)

    # Select best volume
    volume_3d, selected_dcm, selected_name = select_best_volume(all_data)

    if volume_3d is None:
        print("ERROR: Could not select a volume!")
        return False

    # Process volume
    print(f"\n{'=' * 80}")
    print("Processing Volume")
    print('=' * 80)
    print(f"Shape: {volume_3d.shape} (Y, X, Z)")
    print(f"Dtype: {volume_3d.dtype}")
    print(f"Range: [{volume_3d.min()}, {volume_3d.max()}]")

    # Convert to uint8
    if volume_3d.dtype == np.int8:
        volume_uint8 = volume_3d.astype(np.int16) + 128
        volume_uint8 = volume_uint8.astype(np.uint8)
    else:
        vol_normalized = volume_3d.astype(np.float32)
        vol_normalized = (vol_normalized - vol_normalized.min()) / (vol_normalized.max() - vol_normalized.min())
        volume_uint8 = (vol_normalized * 255).astype(np.uint8)

    print(f"Converted to: uint8 [0, 255]")

    # Calculate voxel size
    voxel_x, voxel_y, voxel_z, scan_width, scan_depth = calculate_voxel_size(volume_3d.shape)

    print(f"\nVoxel size (estimated):")
    print(f"  X: {voxel_x:.3f} µm")
    print(f"  Y: {voxel_y:.3f} µm")
    print(f"  Z: {voxel_z:.3f} µm")
    print(f"Scan dimensions: {scan_width}x{scan_width}x{scan_depth} mm")

    # Save files
    print(f"\n{'=' * 80}")
    print("Saving Files")
    print('=' * 80 + "\n")

    # Create output folder structure: Results/<folder_name>/
    script_dir = Path(__file__).parent
    results_base = script_dir / "Results"
    output_folder = results_base / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    base_name = f"OCTA_{folder_name}"

    print(f"Output folder: {output_folder.relative_to(script_dir)}\n")

    # 1. NumPy
    npy_path = output_folder / f"{base_name}.npy"
    np.save(npy_path, volume_uint8)
    print(f"[1] NumPy: {npy_path.name} ({npy_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Metadata
    meta_data = {
        'source_folder': folder_name,
        'source_file': selected_name,
        'shape': list(volume_3d.shape),
        'shape_description': 'Y (B-scans), X (width), Z (depth)',
        'dtype': 'uint8',
        'voxel_size_um': {'X': float(voxel_x), 'Y': float(voxel_y), 'Z': float(voxel_z)},
        'scan_dimensions_mm': {'width': scan_width, 'depth': scan_depth},
        'patient_id': str(getattr(selected_dcm, 'PatientID', 'Unknown')),
        'study_date': str(getattr(selected_dcm, 'StudyDate', 'Unknown')),
        'device': 'Zeiss Cirrus HD-OCT'
    }

    json_path = output_folder / f"{base_name}_metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2)
    print(f"[2] Metadata: {json_path.name}")

    # 3. TIFF for Imaris
    try:
        import tifffile

        tiff_path = output_folder / f"{base_name}.tif"

        # Transpose to (Z, Y, X) for ImageJ/Imaris
        volume_zyx = np.transpose(volume_uint8, (2, 0, 1))

        print(f"[3] TIFF: Transposed to {volume_zyx.shape} (Z, Y, X)")

        # Resolution in pixels per mm
        resolution_x = 1000.0 / voxel_x  # pixels per mm
        resolution_y = 1000.0 / voxel_y
        spacing_z = voxel_z / 1000  # mm

        tifffile.imwrite(
            tiff_path,
            volume_zyx,
            imagej=True,
            resolution=(resolution_y, resolution_x),
            metadata={'spacing': spacing_z, 'unit': 'um', 'axes': 'ZYX'}
        )

        file_size = tiff_path.stat().st_size / 1024 / 1024
        print(f"    Saved: {tiff_path.name} ({file_size:.2f} MB)")
        print(f"    ✓ Ready for Imaris!")

    except Exception as e:
        print(f"[3] TIFF: ERROR - {e}")

    # 4. NIfTI for medical imaging software
    try:
        import nibabel as nib

        nifti_path = output_folder / f"{base_name}.nii.gz"

        # NIfTI uses RAS+ coordinate system, we use (X, Y, Z) orientation
        # Volume is already in (Y, X, Z), transpose to (X, Y, Z) for standard neuroimaging
        volume_xyz = np.transpose(volume_uint8, (1, 0, 2))

        # Create affine matrix with voxel sizes (in mm)
        # NIfTI expects voxel sizes in mm
        affine = np.array([
            [voxel_x / 1000, 0, 0, 0],
            [0, voxel_y / 1000, 0, 0],
            [0, 0, voxel_z / 1000, 0],
            [0, 0, 0, 1]
        ])

        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume_xyz, affine)

        # Add metadata to header
        nifti_img.header['descrip'] = f'Zeiss OCTA {folder_name}'.encode('utf-8')
        nifti_img.header['xyzt_units'] = 2  # mm for spatial units

        # Save compressed NIfTI
        nib.save(nifti_img, str(nifti_path))

        file_size = nifti_path.stat().st_size / 1024 / 1024
        print(f"[4] NIfTI: {nifti_path.name} ({file_size:.2f} MB)")
        print(f"    Shape: {volume_xyz.shape} (X, Y, Z)")
        print(f"    Voxel size: {voxel_x / 1000:.4f} x {voxel_y / 1000:.4f} x {voxel_z / 1000:.4f} mm")
        print(f"    ✓ Ready for medical imaging software!")

    except ImportError:
        print(f"[4] NIfTI: Skipped (nibabel not installed)")
        print(f"    Install with: pip install nibabel")
    except Exception as e:
        print(f"[4] NIfTI: ERROR - {e}")

    # 5. Preview
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print(f"\n[5] Generating preview...")

        # Maximum intensity projections
        mip_z = np.max(volume_uint8, axis=2)
        mip_y = np.max(volume_uint8, axis=0)
        mip_x = np.max(volume_uint8, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(mip_z, cmap='hot')
        axes[0, 0].set_title(f'En Face (MIP Z)\n{folder_name}', fontsize=12, weight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mip_y, cmap='hot', aspect='auto')
        axes[0, 1].set_title('Side view (MIP Y)', fontsize=10)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(mip_x, cmap='hot', aspect='auto')
        axes[1, 0].set_title('Side view (MIP X)', fontsize=10)
        axes[1, 0].axis('off')

        # Central depth slice
        central_z = volume_uint8.shape[2] // 2
        axes[1, 1].imshow(volume_uint8[:, :, central_z], cmap='gray')
        axes[1, 1].set_title(f'Central depth slice (Z={central_z})', fontsize=10)
        axes[1, 1].axis('off')

        plt.tight_layout()
        preview_path = output_folder / f"{base_name}_Preview.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Preview: {preview_path.name}")

    except Exception as e:
        print(f"[5] Preview: ERROR - {e}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUCCESS!")
    print('=' * 80)
    print(f"\nOutput files:")
    print(f"  - {base_name}.tif (for Imaris)")
    print(f"  - {base_name}.nii.gz (for medical imaging software: ITK-SNAP, 3D Slicer, etc.)")
    print(f"  - {base_name}.npy (NumPy array for Python)")
    print(f"  - {base_name}_metadata.json (scan parameters)")
    print(f"  - {base_name}_Preview.png (visualization)")
    print(f"\nFor Imaris:")
    print(f"  1. Open {base_name}.tif")
    print(f"  2. Voxel size is embedded: X={voxel_x:.3f}, Y={voxel_y:.3f}, Z={voxel_z:.3f} µm")
    print('=' * 80 + "\n")

    return True


if __name__ == "__main__":
    main()
