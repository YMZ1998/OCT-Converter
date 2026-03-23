import os

from scripts.dir_process import remove_and_create_dir
# from oct_converter.readers import Dicom
from zeiss_dicom import ZEISSDicom as Dicom


def test(filepath):
    print("=" * 50)
    # ================= 结果根目录 =================
    result_root = r"E:\\Data\\OCT\\Result\\蔡司OCT"

    # ================= 自动派生输出目录 =================
    parent_dir = os.path.dirname(filepath)

    output_dir = os.path.join(result_root, os.path.basename(parent_dir),
                              os.path.splitext(os.path.basename(filepath))[0])
    remove_and_create_dir(output_dir)

    print(f"[INFO] 输出目录: {output_dir}")

    file = Dicom(filepath)

    # ================= OCT → TIFF =================
    oct_volumes, fundus_volumes = file.read_data()  # returns a list OCT and fudus images
    print(f"[INFO] OCT volumes: {len(oct_volumes)}")
    print(f"[INFO] Fundus volumes: {len(fundus_volumes)}")
    for idx, volume in enumerate(oct_volumes):
        volume.save(os.path.join(output_dir, f"zeiss_volume_{idx}.png"))  # save all volumes

    for idx, image in enumerate(fundus_volumes):
        image.save(os.path.join(output_dir, f"zeiss_image_{idx}.png"))
    #
    # for volume in oct_volumes:
    #     volume.peek(show_contours=True)  # plots a montage of the volume
    #     volume.save(os.path.join(output_dir, "{}_{}.avi".format(volume.volume_id, volume.laterality)))



if __name__ == "__main__":
    # ================= 输入文件 =================
    filepath = r'E:\Data\OCT\蔡司OCT\DataFiles\E195'
    for p in os.listdir(filepath):
        if p.endswith(".DCM"):
            test(os.path.join(filepath, p))
