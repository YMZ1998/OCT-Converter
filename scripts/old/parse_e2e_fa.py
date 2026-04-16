import json
import os

from oct_converter.readers import E2E


def run(filepath):
    file = E2E(filepath)

    fundus_images = file.read_fundus_image()
    metadata = file.read_all_metadata()
    for key, value in metadata.items():
        print(f"[INFO] {key}: {len(value)}")
    meta_path = os.path.join(r"E:\Data\OCT2\海德堡\KH902-R10-007-007003DME-V1-FFA", f"metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"[OK] Metadata saved: {meta_path}")
    # print(metadata)

    for image in fundus_images:
        print("-----------------------------")
        print(f"[INFO] OCT image id: {image.image_id, image.acquisition_date}")
        # for key, value in metadata.items():
        #     print(f"[INFO] {key}: {len(value)}")
        bscans = metadata.get("bscan_data", [])
        for bscan in bscans:
            # print(bscan)
            numImages = bscan.get('numImages', 0)
            # print(f" {bscan['aktImage']}, {bscan['numImages']}")


if __name__ == "__main__":
    # ================= 输入文件 =================
    filepath1 = r"E:\Data\OCT2\海德堡\KH902-R10-007-007003DME-V1-FFA\KH902-R10-007-007003DME-V1-FFA-OD.E2E"
    run(filepath1)

