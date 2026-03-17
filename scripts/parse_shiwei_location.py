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
fundus_file = os.path.join(input_dir, 'D-120001DME_OD_2026-02-11_10-11-16_RotatedStructural_csso.dcm')

# -------------------------------
# 读取 B-scan 体积
# -------------------------------
ds_b = pydicom.dcmread(bscan_file)
vol_b = ds_b.pixel_array  # shape = (num_slices, H, W)
num_slices, H, W = vol_b.shape

# -------------------------------
# 读取眼底图像
# -------------------------------
ds_f = pydicom.dcmread(fundus_file)
fundus = ds_f.pixel_array  # shape = (Hf, Wf) 或 (Hf, Wf, 3)

# -------------------------------
# 获取每帧旋转角度和坐标
# -------------------------------
angles = []
coords_list = []

if hasattr(ds_b, 'PerFrameFunctionalGroupsSequence'):
    for f in ds_b.PerFrameFunctionalGroupsSequence:
        if hasattr(f, 'OphthalmicFrameLocationSequence'):
            loc_seq = f.OphthalmicFrameLocationSequence[0]
            coords = loc_seq.ReferenceCoordinates  # [x0, y0, x1, y1]
            coords_list.append(coords)
            angle = np.arctan2(coords[3] - coords[1], coords[2] - coords[0])
            angles.append(np.degrees(angle))
        else:
            coords_list.append([0, 0, 0, 0])
            angles.append(0.0)
else:
    for _ in range(num_slices):
        coords_list.append([0, 0, 0, 0])
        angles.append(0.0)

# -------------------------------
# 初始化 Figure
# -------------------------------
fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

# B-scan 显示
slice_idx = 0
im_b = ax_b.imshow(vol_b[slice_idx], cmap='gray')
ax_b.set_title(f"B-scan slice {slice_idx}, angle {angles[slice_idx]:.2f}°")
ax_b.axis('off')

# 眼底图显示
if fundus.ndim == 2:
    im_f = ax_f.imshow(fundus, cmap='gray')
else:
    im_f = ax_f.imshow(fundus)
ax_f.axis('off')
# 叠加参考线
line = ax_f.plot([coords_list[slice_idx][0], coords_list[slice_idx][2]],
                 [coords_list[slice_idx][1], coords_list[slice_idx][3]],
                 color='red', linewidth=2)[0]

# -------------------------------
# 滑动条
# -------------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=slice_idx, valstep=1)


def update(val):
    idx = int(slider.val)
    # 更新 B-scan
    im_b.set_data(vol_b[idx])
    ax_b.set_title(f"B-scan slice {idx}, angle {angles[idx]:.2f}°")
    # 更新眼底图上的线条
    line.set_data([coords_list[idx][0], coords_list[idx][2]],
                  [coords_list[idx][1], coords_list[idx][3]])
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
