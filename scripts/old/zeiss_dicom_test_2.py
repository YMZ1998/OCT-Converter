import os

from scripts.old.dir_process import remove_and_create_dir
from zeiss_dicom import ZEISSDicom

dir = r'E:\Data\OCT\蔡司OCT\DataFiles\E198'
output_dir = r"E:\Data\OCT\Result\zeiss"
remove_and_create_dir(output_dir)

for file in os.listdir(dir):
    if file.endswith('.DCM'):
        file_path = os.path.join(dir, file)
        print(file)
        img = ZEISSDicom(file_path)

        oct_volumes, fundus_volumes = img.read_data()  # returns a list OCT and fudus images
        for idx, volume in enumerate(oct_volumes):
            volume.save(os.path.join(output_dir, file.split('.')[0], f"zeiss_volume_{idx}.png"))  # save all volumes

        print("Fundus volumes:", len(fundus_volumes))
        for idx, image in enumerate(fundus_volumes):
            image.save(os.path.join(output_dir, file.split('.')[0], f"zeiss_image_{idx}.png"))

        # for volume in oct_volumes:
        # volume.peek(show_contours=True)  # plots a montage of the volume
        # volume.save(os.path.join(output_dir, "{}_{}.avi".format(volume.volume_id, volume.laterality)))
