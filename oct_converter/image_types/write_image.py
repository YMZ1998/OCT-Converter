import os

import cv2


def cv2_imwrite_safe(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    with open(path, "wb") as f:
        buf.tofile(f)
