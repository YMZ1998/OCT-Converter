import struct
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# 1. 基础工具
# =========================

def find_block(data: bytes, tag: bytes):
    """查找块起始位置"""
    idx = data.find(tag)
    if idx == -1:
        raise ValueError(f"Block {tag} not found")
    return idx + len(tag)


def dump_hex(data: bytes, start=0, length=128):
    """16进制打印"""
    chunk = data[start:start + length]
    for i in range(0, len(chunk), 16):
        line = chunk[i:i + 16]
        hex_str = " ".join(f"{b:02X}" for b in line)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in line)
        print(f"{start+i:08X}  {hex_str:<48}  {ascii_str}")


def read_u16_array(data: bytes, offset: int, count: int):
    """little-endian uint16"""
    return list(struct.unpack_from(f"<{count}H", data, offset))


# =========================
# 2. EFFECTIVE_SCAN_RANGE
# =========================

def parse_effective_scan_range_hex(data: bytes):
    tag = b"@EFFECTIVE_SCAN_RANGE"
    offset = find_block(data, tag)

    print("\n==============================")
    print("EFFECTIVE_SCAN_RANGE HEX DUMP")
    print("==============================")
    dump_hex(data, offset, 128)

    values = read_u16_array(data, offset, 12)

    print("\n[UINT16解析]")
    for i, v in enumerate(values):
        print(f"{i:02d}: 0x{v:04X} ({v})")

    bbox = [
        (values[0], values[1]),
        (values[2], values[3]),
        (values[4], values[5]),
        (values[6], values[7]),
    ]

    return values, bbox


# =========================
# 3. REF_IMG_SCAN（核心曲线）
# =========================

def parse_ref_img_scan_hex(data: bytes):
    tag = b"@REF_IMG_SCAN"
    offset = find_block(data, tag)

    print("\n==============================")
    print("@REF_IMG_SCAN HEADER HEX")
    print("==============================")
    dump_hex(data, offset, 128)

    header = read_u16_array(data, offset, 11)

    point_count = header[1]
    width = header[2]
    height = header[4]

    print("\n[HEADER]")
    for i, v in enumerate(header):
        print(f"{i:02d}: 0x{v:04X} ({v})")

    print(f"\npoint_count = {point_count}")
    print(f"width = {width}, height = {height}")

    # curve data
    curve_offset = offset + 11 * 2

    print("\n==============================")
    print("CURVE HEX DUMP")
    print("==============================")
    dump_hex(data, curve_offset, min(128, point_count * 2))

    points = read_u16_array(data, curve_offset, point_count)

    return header, points


# =========================
# 4. 点转换（假设模型）
# =========================

def convert_to_xy(points, width):
    """
    假设：
    x 等间距
    y 为曲线值
    """
    if len(points) == 0:
        return []

    xy = [(i * width // len(points), v) for i, v in enumerate(points)]
    return xy


# =========================
# 5. 可视化
# =========================

def plot_points(xy):
    x = [p[0] for p in xy]
    y = [p[1] for p in xy]

    plt.figure()
    plt.scatter(x, y, s=10)
    plt.plot(x, y, alpha=0.5)

    plt.title("REF IMG SCAN Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().invert_yaxis()  # OCT常用

    plt.axis("equal")
    plt.show()


# =========================
# 6. 主函数
# =========================

def main():
    path = Path(r"E:\Data\OCT\拓普康OCT\41365.fda")
    data = path.read_bytes()

    # ---- EFFECTIVE_SCAN_RANGE ----
    eff, bbox = parse_effective_scan_range_hex(data)
    print("\nBBox guess:", bbox)

    # ---- REF_IMG_SCAN ----
    header, points = parse_ref_img_scan_hex(data)

    print("\nCurve preview:")
    print(points[:20])

    # ---- 转换坐标 ----
    xy = convert_to_xy(points, header[2])

    print("\nXY preview:")
    for p in xy[:10]:
        print(p)

    # ---- 画图 ----
    plot_points(xy)


if __name__ == "__main__":
    main()