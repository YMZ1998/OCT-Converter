import math

import pydicom
import numpy as np

def extract_oct_data(file_path):
    ds = pydicom.dcmread(file_path)

    seq = ds[(0x0009, 0x0001)].value  # Private Sequence
    print(seq)
    # 用来存放结果
    oct_data = {
        "OCTThumbData": None,
        "SLOData": None,
        "OCTFundus": None
    }

    for item in seq:
        name = item[(0x0009, 0x0002)].value  # Private Creator 名称
        data = item[(0x0009, 0x0003)].value  # 对应数组/bytes
        print(name)
        # print(data)
        if b'OCTThumbData' in name:
            oct_data["OCTThumbData"] = np.frombuffer(data, dtype=np.uint8)
        elif b'SLOData' in name:
            oct_data["SLOData"] = np.frombuffer(data, dtype=np.uint8)
        elif b'OCTFundus' in name:
            oct_data["OCTFundus"] = np.frombuffer(data, dtype=np.uint8)

    # 打印结果信息
    for k, v in oct_data.items():
        if v is not None:
            size = int(math.sqrt(v.size))
            size2 = size * size
            print(f"{k}: {size}x{size} elements,{size2-v.size}")
            print(f"{k}: {v.size} elements, dtype={v.dtype}")
        else:
            print(f"{k}: Not found")

    return oct_data

# 使用示例
file_path = r"E:\Data\OCT\视微OCT\2025-12-24_16-52-52\QH01(2025-12-24 16.52.52)_small.encg"
oct_data = extract_oct_data(file_path)
