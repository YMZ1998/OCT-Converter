import re
from pathlib import Path

e2e = Path("E:\Data\OCT2\海德堡\KH902-R10-007-007003DME-V1-FFA\KH902-R10-007-007003DME-V1-FFA-OD.E2E").read_bytes()
# e2e = Path("E:\Data\OCT2\海德堡\KH902-R10-007-007003DME-V1-FFA\KH902-R10-007-007003DME-V1-FFA-OS.e2e").read_bytes()
# e2e = Path(r"E:\Data\OCT\海德堡\海德堡FA.E2E").read_bytes()

# 匹配 UTF-16LE 形式的 "0:01.69 30°" / "10:43.50 30°"
pat = re.compile(
    rb'(?:\d\x00)+:\x00(?:\d\x00){2}\.\x00(?:\d\x00){2}\x20\x00(?:\d\x00){2}\xb0\x00'
)

labels = [m.group(0).decode("utf-16le") for m in pat.finditer(e2e)]

print(labels[:])
print(len(labels))
