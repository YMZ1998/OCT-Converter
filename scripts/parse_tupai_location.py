import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pydicom

# -------------------------------
# 文件路径
# -------------------------------
input_dir = r'E:\Data\OCT2\图湃OCT\KH902-R10-D-114001DME_男_26_D-114001DME\20260127_115627_左眼_12.00mmX12.00mm_3D黄斑'
bscan_file = os.path.join(input_dir, 'OCT.dcm')
fundus_file = os.path.join(input_dir, 'Fundus.dcm')

# -------------------------------
# 读取 B-scan 体积
# -------------------------------
ds_b = pydicom.dcmread(bscan_file)
vol_b = ds_b.pixel_array  # (N, H, W)
num_slices, H_b, W_b = vol_b.shape

# -------------------------------
# 读取 fundus
# -------------------------------
ds_f = pydicom.dcmread(fundus_file)
fundus = ds_f.pixel_array

if fundus.ndim == 3:
    fundus_img = fundus
else:
    fundus_img = fundus

H_f, W_f = fundus_img.shape[:2]

# -------------------------------
# 提取 ReferenceCoordinates
# -------------------------------
coords_list = []

if hasattr(ds_b, 'PerFrameFunctionalGroupsSequence'):
    for f in ds_b.PerFrameFunctionalGroupsSequence:
        if hasattr(f, 'OphthalmicFrameLocationSequence'):
            rc = np.array(
                f.OphthalmicFrameLocationSequence[0].ReferenceCoordinates,
                dtype=float
            )
        else:
            rc = np.array([0, 0, 0, 0], dtype=float)

        coords_list.append(rc)
else:
    coords_list = [np.array([0,0,0,0], dtype=float)] * num_slices

coords_list = np.array(coords_list)


# -------------------------------
# 判断是否为 fundus 坐标（关键）
# -------------------------------
def is_fundus_rc(rc, H):
    x0, y0, x1, y1 = rc
    return (
        0 <= y0 <= H and
        0 <= y1 <= H and
        abs(y0 - 0) < 5 and
        abs(y1 - H) < 5
    )


def get_valid_rc(idx):
    rc = coords_list[idx]
    if is_fundus_rc(rc, H_f):
        return rc
    else:
        return None


# -------------------------------
# 初始化显示
# -------------------------------
fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.2)

# fundus
if fundus_img.ndim == 2:
    ax_f.imshow(fundus_img, cmap='gray', origin='upper')
else:
    ax_f.imshow(fundus_img, origin='upper')
ax_f.set_title("Fundus")
ax_f.axis('off')

# B-scan
slice_idx = 0
im_b = ax_b.imshow(vol_b[slice_idx], cmap='gray', origin='upper')
ax_b.set_title(f"B-scan {slice_idx+1}/{num_slices}")
ax_b.axis('off')

# 初始线
rc0 = get_valid_rc(slice_idx)

if rc0 is not None:
    x0, y0, x1, y1 = rc0
    line, = ax_f.plot([x0, x1], [y0, y1],
                      color='red', linewidth=2, alpha=0.8)
else:
    line, = ax_f.plot([], [], color='red', linewidth=2, alpha=0.8)
    line.set_visible(False)


# -------------------------------
# Slider
# -------------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, num_slices-1,
                valinit=slice_idx, valstep=1)


# -------------------------------
# 更新函数
# -------------------------------
def update(val):
    idx = int(slider.val)

    # 更新 B-scan
    im_b.set_data(vol_b[idx])
    ax_b.set_title(f"B-scan {idx+1}/{num_slices}")

    # 更新线
    rc = get_valid_rc(idx)

    if rc is not None:
        x0, y0, x1, y1 = rc
        line.set_data([x0, x1], [y0, y1])
        line.set_visible(True)
    else:
        line.set_visible(False)

    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()