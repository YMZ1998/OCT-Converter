import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.widgets import Slider

# -------------------------------
# 文件路径
# -------------------------------
input_dir = r'E:\Data\OCT2\视微OCT\00402_20260224001_D-120001DME_OD_2026-02-11_10-11-16Cube 6x6 512x512\Dicom'
bscan_file = os.path.join(input_dir, 'D-120001DME_OD_2026-02-11_10-11-16_RotatedStructural_structural.dcm')

# -------------------------------
# 读取 B-scan 体积
# -------------------------------
ds_b = pydicom.dcmread(bscan_file)
vol_b = ds_b.pixel_array  # shape = (num_slices, H, W)
num_slices, H, W = vol_b.shape

# -------------------------------
# 获取每帧旋转角度
# -------------------------------
angles = []

if hasattr(ds_b, 'PerFrameFunctionalGroupsSequence'):
    for f in ds_b.PerFrameFunctionalGroupsSequence:
        # 尝试从 OphthalmicFrameLocationSequence 获取旋转信息
        if hasattr(f, 'OphthalmicFrameLocationSequence'):
            loc_seq = f.OphthalmicFrameLocationSequence[0]
            coords = loc_seq.ReferenceCoordinates  # [x0, y0, x1, y1]
            print(coords)
            # 计算旋转角度（X-Y 平面）
            angle = np.arctan2(coords[3] - coords[1], coords[2] - coords[0])
            angles.append(np.degrees(angle))
        else:
            angles.append(0.0)  # 没有序列就默认 0

print(angles)
# -------------------------------
# 初始化 Figure
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.2)
slice_idx = 0
im = ax.imshow(vol_b[slice_idx], cmap='gray')
ax.set_title(f"B-scan slice {slice_idx + 1}/{num_slices}")
ax.axis('off')

# -------------------------------
# 滑动条
# -------------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 1, num_slices, valinit=slice_idx, valstep=1)


def update(val):
    idx = int(slider.val)
    im.set_data(vol_b[idx - 1])
    ax.set_title(f"B-scan slice {idx}/{num_slices}")
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()
