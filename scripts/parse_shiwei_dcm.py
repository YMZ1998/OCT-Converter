import os.path

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pydicom

input_dir = r'E:\Data\OCT2\视微OCT\00402_20260224001_D-120001DME_OD_2026-02-11_10-11-16Cube 6x6 512x512\Dicom'
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(input_dir)
print(series_ids)
if not series_ids:
    print("❌ 没有找到任何 DICOM series。")

print(f"📦 找到 {len(series_ids)} 个 DICOM series.")

# ds = pydicom.dcmread(os.path.join(input_dir,'D-120001DME_OD_2026-02-11_10-11-16_RotatedStructural_csso.dcm'))
ds = pydicom.dcmread(os.path.join(input_dir,'D-120001DME_OD_2026-02-11_10-11-16_RotatedStructural_structural.dcm'))
# ds = pydicom.dcmread(os.path.join(input_dir, 'D-120001DME_OD_2026-02-11_10-11-16_Structural.dcm'))
# ds = pydicom.dcmread(os.path.join(input_dir, 'D-120001DME_OD_2026-02-11_10-11-16_Segmentation.dcm'))
print(ds)
print(ds.SamplesPerPixel)
print(ds.PhotometricInterpretation)
print(ds.PatientName)

img = ds.pixel_array
print(img.shape, img.dtype)

plt.imshow(img[:, :, 200], cmap="gray")
# plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()


# 体素信息
volume = ds.pixel_array  # 假设 shape = (num_slices, height, width)

# DICOM 提供的物理间距信息
# PixelSpacing: [row spacing (Δy), col spacing (Δx)]
# SliceThickness: Δz
delta_y, delta_x = [float(x) for x in ds.PixelSpacing]
delta_z = float(ds.SliceThickness)

# ImagePositionPatient: 左上角坐标 [X0, Y0, Z0]
x0, y0, z0 = [float(x) for x in ds.ImagePositionPatient]

# 计算每个体素的物理坐标
num_slices, height, width = volume.shape
X = x0 + np.arange(width) * delta_x
Y = y0 + np.arange(height) * delta_y
Z = z0 + np.arange(num_slices) * delta_z

# 构建 3D 网格
X_grid, Y_grid, Z_grid = np.meshgrid(X, Y, Z, indexing='xy')

print("体素物理坐标形状:", X_grid.shape)


