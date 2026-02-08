import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers import convert_color_space
import SimpleITK as sitk

input_dir=r'E:\Data\OCT\蔡司FA\2'
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(input_dir)
print(series_ids)
if not series_ids:
    print("❌ 没有找到任何 DICOM series。")

print(f"📦 找到 {len(series_ids)} 个 DICOM series.")

# ds = pydicom.dcmread(r"E:\Data\OCT\蔡司OCT\DataFiles\E195\2TXU4UFY5T892H35RCF8PBV9D4KKGIER27IOLS4X2ZU.EX.DCM")
ds = pydicom.dcmread(r'E:\Data\OCT\蔡司FA\2\2000010120251027JenaOD(40011).dcm')
# ds = pydicom.dcmread( r'E:\Data\OCT\蔡司CFP\18017 V1 FP\2000010120251027JenaOD(39979).dcm')
# ds = pydicom.dcmread( r'E:\Data\OCT\蔡司CFP\18017 V1 FP\2000010120251027JenaOD(39984).dcm')
print(ds)
print(ds.SamplesPerPixel)
print(ds.PhotometricInterpretation)
print(ds.PatientName)


img = ds.pixel_array
print(img.shape)

# plt.imshow(img[:, :, 2], cmap="gray")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

img_rgb = convert_color_space(
    img,
    current="YBR_FULL_422",
    desired="RGB"
)

print(img_rgb.shape)  # (1444, 1444, 3)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()
