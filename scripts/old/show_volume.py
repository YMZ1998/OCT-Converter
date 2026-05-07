import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def show_volume_slider(volume: np.ndarray, axis: int = 0, cmap: str = "gray"):
    """
    可交互浏览3D体数据（带滑动条）

    参数：
    ----------
    volume : np.ndarray
        3D数组，例如 (Z, H, W)
    axis : int
        切片方向：
            0 -> Z方向（B-scan，默认）
            1 -> Y方向
            2 -> X方向
    cmap : str
        显示颜色映射
    """

    # assert volume.ndim == 3, "只支持3D数据"

    # 统一把目标轴换到第0维
    vol = np.moveaxis(volume, axis, 0)

    num_slices = vol.shape[0]
    slice_idx = num_slices // 2  # 默认中间

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)

    # 显示初始图像
    im = ax.imshow(vol[slice_idx], cmap=cmap,
                   vmin=vol.min(), vmax=vol.max())
    ax.set_title(f"Slice {slice_idx + 1}/{num_slices}")
    ax.axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        'Slice',
        1,
        num_slices,
        valinit=slice_idx + 1,
        valstep=1
    )

    # 更新函数
    def update(val):
        idx = int(slider.val) - 1
        im.set_data(vol[idx])
        ax.set_title(f"Slice {idx + 1}/{num_slices} (axis={axis})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
