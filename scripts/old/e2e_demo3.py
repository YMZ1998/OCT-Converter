from oct_converter.readers import E2E


def run(filepath):
    file = E2E(filepath)

    oct_volumes = file.read_oct_volume()
    print(f"[INFO] OCT volumes: {len(oct_volumes)}")
    oct_volumes = sorted(oct_volumes, key=lambda v: int(v.volume_id.split('_')[-1]))
    for volume in oct_volumes:
        print("-----------------------------")
        print(f"[INFO] OCT volume id: {volume.volume_id}")
        # volume.peek(show_contours=True)
        print(
            f"[INFO] OCT volume: {volume.num_slices} ,{volume.laterality}, {volume.scan_pattern}, {volume.acquisition_date}")
        print(len(volume.volume))
        metadata = volume.metadata
        # print(metadata)
        # for key, value in metadata.items():
        #     print(f"[INFO] {key}: {len(value)}")
        # meta_path = os.path.join( r"E:\Data\OCT\Result\海德堡", f"{volume.volume_id}_metadata.json")
        # with open(meta_path, "w", encoding="utf-8") as f:
        #     json.dump(metadata, f, indent=4, ensure_ascii=False)
        # print(f"[OK] Metadata saved: {meta_path}")
    fundus_images = file.read_fundus_image()
    for image in fundus_images:
        print("-----------------------------")
        print(f"[INFO] OCT image id: {image.image_id}")
        metadata = image.metadata
        # print(metadata)
        # for key, value in metadata.items():
        #     print(f"[INFO] {key}: {len(value)}")
        bscans = metadata.get("bscan_data", [])
        for bscan in bscans:
            # print(bscan)
            numImages = bscan.get('numImages', 0)
            # print(f" {bscan['aktImage']}, {bscan['numImages']}")

if __name__ == "__main__":
    # ================= 输入文件 =================
    filepath1 = r"E:\Data\OCT\海德堡\海德堡OCT.E2E"
    run(filepath1)
