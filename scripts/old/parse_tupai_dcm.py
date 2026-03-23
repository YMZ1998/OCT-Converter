import os.path

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pydicom

# input_dir = r'E:\Data\OCT2\图湃OCT\KH902-R10 R-096016RVO_男_26_R-096016RVO\20260303_102557_右眼_12.00mmX12.00mm_3D黄斑'
input_dir = r'E:\Data\OCT2\图湃OCT\KH902-R10-D-114001DME_男_26_D-114001DME\20260127_115627_左眼_12.00mmX12.00mm_3D黄斑'
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(input_dir)
print(series_ids)
if not series_ids:
    print("❌ 没有找到任何 DICOM series。")

print(f"📦 找到 {len(series_ids)} 个 DICOM series.")

ds = pydicom.dcmread(os.path.join(input_dir, 'OCT.dcm'))
# ds = pydicom.dcmread(os.path.join(input_dir,'Enface.dcm'))
# ds = pydicom.dcmread(os.path.join(input_dir, 'Fundus.dcm'))
# print(ds)
print(ds.SamplesPerPixel)
print(ds.PhotometricInterpretation)
print(ds.PatientName)
print(ds.PixelSpacing)
if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
    for f in ds.PerFrameFunctionalGroupsSequence:
        if hasattr(f, 'OphthalmicFrameLocationSequence'):
            loc_seq = f.OphthalmicFrameLocationSequence[0]
            coords = np.array(loc_seq.ReferenceCoordinates, dtype=float)
            print("ReferenceCoordinates: ", coords)
        if hasattr(f, 'PlanePositionSequence'):
            pos = np.array(f.PlanePositionSequence[0].ImagePositionPatient, dtype=float)
            print("ImagePositionPatient:", pos)

img = ds.pixel_array
print(img.shape, img.dtype)
if (len(img.shape) > 2):
    plt.imshow(img[:, :, 0], cmap="gray")
else:
    plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
