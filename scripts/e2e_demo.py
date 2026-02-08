import json
import os

from oct_converter.readers import E2E
from scripts.dir_process import remove_and_create_dir


def test(filepath):
    # ================= 结果根目录 =================
    result_root = r"E:\\Data\\OCT\\Result"

    # ================= 自动派生输出目录 =================
    parent_dir = os.path.dirname(filepath)

    output_dir = os.path.join(result_root, os.path.basename(parent_dir),
                              os.path.splitext(os.path.basename(filepath))[0])
    # os.makedirs(output_dir, exist_ok=True)
    remove_and_create_dir(output_dir)

    print(f"[INFO] 输出目录: {output_dir}")

    # ================= 读取 E2E =================
    file = E2E(filepath)

    # ================= OCT → TIFF =================
    oct_volumes = file.read_oct_volume()
    print(f"[INFO] OCT volumes: {len(oct_volumes)}")

    for volume in oct_volumes:
        out_tiff = os.path.join(
            output_dir,
            f"{volume.volume_id}_{volume.laterality}.tiff"
        )
        # volume.peek(show_contours=True)
        volume.save(out_tiff)
        print(f"[OK] TIFF saved: {out_tiff}")

    # ================= Fundus / FA → PNG =================
    fundus_images = file.read_fundus_image()
    print(f"[INFO] Fundus images: {len(fundus_images)}")
    for image in fundus_images:
        out_png = os.path.join(
            output_dir,
            f"{image.image_id}_{image.laterality}.png"
        )
        image.save(out_png)
        print(f"[OK] PNG saved: {out_png}")

    # ================= Metadata → JSON =================
    metadata = file.read_all_metadata()
    meta_path = os.path.join(output_dir, "metadata.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"[OK] Metadata saved: {meta_path}")

    # # ================= OCT → DICOM =================
    # from oct_converter.dicom import create_dicom_from_oct
    # dicom_dir = os.path.join(output_dir, "dicom")
    # os.makedirs(dicom_dir, exist_ok=True)
    #
    # dcm_files = create_dicom_from_oct(
    #     filepath,
    #     output_dir=dicom_dir
    # )
    # print(f"[INFO] DICOM files generated: {len(dcm_files)}")
    # for dcm in dcm_files:
    #     print(f"  └─ {dcm}")


if __name__ == "__main__":
    # ================= 输入文件 =================
    filepath1 = r"E:\Data\OCT\海德堡\海德堡FA.E2E"
    filepath2 = r"E:\Data\OCT\海德堡\海德堡OCT.E2E"
    test(filepath1)
    test(filepath2)
