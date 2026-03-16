import os
import sqlite3

# 数据库路径
db_path = r"E:\Data\OCT\图湃OCT.db"
# 输出目录
output_dir = r"E:\Data\OCT\extracted_files"
os.makedirs(output_dir, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 读取 Table2025111290346 所有 BLOB 文件
cursor.execute("SELECT FileName, BinaryData FROM Table2025111290346;")
rows = cursor.fetchall()

for fname, blob in rows:
    # 构造保存路径
    safe_name = fname.replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, safe_name)

    # 写入文件
    with open(path, "wb") as f:
        f.write(blob)
    print(f"✅ 已保存: {path} ({len(blob)} bytes)")

conn.close()
print("🎉 全部文件提取完成")

import numpy as np
import cv2
import struct
import os


# 读取 idx
with open(os.path.join(output_dir, "ShootData.img.idx"), "rb") as f:
    idx_data = f.read()

# 假设每帧 idx 是 12 字节: offset(4) + length(4) + w(2) + h(2)
frame_info = []
for i in range(0, len(idx_data), 12):
    offset, length, w, h = struct.unpack("<IIHH", idx_data[i:i+12])
    frame_info.append((offset, length, w, h))
print(f"Total frames: {len(frame_info)}")  # 帧信息: {frame_info}
print(f"Frame info: {frame_info}")
# 读取 img
with open(os.path.join(output_dir, "ShootData.img"), "rb") as f:
    img_data = f.read()

# 遍历帧
for i, (offset, length, w, h) in enumerate(frame_info):
    frame_bytes = img_data[offset:offset+length]
    img = np.frombuffer(frame_bytes, dtype=np.uint16)  # 16bit OCT
    img = img[:w*h].reshape((h, w))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"frame_{i}.png"), img)