import eyepy as ep
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# 导入 OCT 数据
# ev = ep.import_heyex_e2e(r"E:\Data\OCT\海德堡\海德堡FA.E2E")
# ev = ep.import_heyex_e2e(r"E:\Data\OCT\海德堡\海德堡OCT.E2E")
# ev = ep.import_heyex_e2e(r"E:\Data\OCT2\海德堡\KH902-R10-007-D-007001DME-V4-OCT.E2E")
ev = ep.import_topcon_fda(r"E:\Data\OCT\拓普康OCT\41365.fda")
# ev = ep.import_topcon_fda(r"E:\Data\OCT\拓普康OCT\41365.fda")
# for i, v in enumerate(ev):
#     print(i, type(v))
print("------")
keys = list(ev.meta.keys())
print(keys)
for k in ev.meta:
    print("------")
    print(k)
    print(ev.meta[k])
print("------")
fundus = ev.localizer.data
# print(ev.localizer_transform)
# print(fundus.shape)
# print(ev.data.shape)

# 计算每条 B-scan 在 fundus 上的位置坐标范围
bscan_coords = []
all_x = []
all_y = []
i = 0
for m in ev.meta["bscan_meta"]:
    # print("--", i)
    # i = i + 1
    # print(m)
    x1, y1 = m["start_pos"]
    x2, y2 = m["end_pos"]
    region = (slice(0, fundus.shape[0]), slice(0, fundus.shape[1]))
    p1 = ev._pos_to_localizer_region((x1, y1), region)
    p2 = ev._pos_to_localizer_region((x2, y2), region)
    bscan_coords.append((p1, p2))
    all_x.extend([p1[0], p2[0]])
    all_y.extend([p1[1], p2[1]])
# print(bscan_coords)
# 总覆盖范围矩形
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
width_total, height_total = x_max - x_min, y_max - y_min

# 中间 B-scan
init_idx = len(ev.data) // 2
p1_init, p2_init = bscan_coords[init_idx]
bscan_init = ev.data[init_idx]

# 创建 figure
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(fundus, cmap="gray", origin='upper')
ax[0].axis('off')
ax[0].set_title("Fundus")
ax[1].imshow(bscan_init, cmap='gray', aspect='auto')
ax[1].axis('off')
ax[1].set_title("B-scan")
ax[0].set_aspect('equal')

# 绘制中间 B-scan 矩形
rect = patches.Rectangle(
    (min(p1_init[0], p2_init[0]), min(p1_init[1], p2_init[1])),
    abs(p2_init[0] - p1_init[0]),
    abs(p2_init[1] - p1_init[1]),
    linewidth=3,
    edgecolor='g',
    facecolor='green',
    alpha=1
)
ax[0].add_patch(rect)

# 绘制总覆盖范围矩形（只用红色线框表示）
rect_total = patches.Rectangle(
    (x_min, y_min),
    width_total,
    height_total,
    linewidth=3,
    edgecolor='r',
    facecolor='none'  # 不填充，只显示线框
)
ax[0].add_patch(rect_total)


# 点击事件更新矩形和 B-scan
def on_click(event):
    if event.inaxes == ax[0]:
        # 找最近的一条 B-scan
        min_dist = float('inf')
        nearest_idx = 0
        for i, (p1, p2) in enumerate(bscan_coords):
            cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            dist = np.sqrt((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 更新矩形位置
        p1, p2 = bscan_coords[nearest_idx]
        rect.set_xy((min(p1[0], p2[0]), min(p1[1], p2[1])))
        rect.set_width(abs(p2[0] - p1[0]))
        rect.set_height(abs(p2[1] - p1[1]))

        # 更新 B-scan 图
        bscan = ev.data[nearest_idx]
        ax[1].imshow(bscan, cmap='gray', aspect='auto')
        ax[1].axis('off')
        fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
