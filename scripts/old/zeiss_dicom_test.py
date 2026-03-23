import os

from zeiss_dicom import ZEISSDicom

filepath = r'E:\Data\OCT\蔡司OCT\DataFiles\E196\25XKIE1D16J892H35RCF8PBV9D4KKGIER27IOLS4X2ZU.EX.DCM'
# filepath = r'E:\Data\OCT\蔡司FA\2\2000010120251027JenaOD(40011).dcm'
img = ZEISSDicom(filepath)
output_dir = r"E:\Data\OCT\Result\zeiss"
os.makedirs(output_dir, exist_ok=True)
oct_volumes, fundus_volumes = img.read_data()  # returns a list OCT and fudus images
for idx, volume in enumerate(oct_volumes):
    volume.save(os.path.join(output_dir, f"zeiss_volume_{idx}.png"))  # save all volumes

for idx, image in enumerate(fundus_volumes):
    image.save(os.path.join(output_dir, f"zeiss_image_{idx}.png"))


for volume in oct_volumes:
    volume.peek(show_contours=True)  # plots a montage of the volume
    volume.save(os.path.join(output_dir, "{}_{}.avi".format(volume.volume_id, volume.laterality)))
