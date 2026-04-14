import json
import struct
from datetime import datetime, timedelta

OLE_BASE = datetime(1899, 12, 30)


def parse_ole_datetime(raw_value):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not (-100000 <= value <= 1000000):
        return None
    try:
        return OLE_BASE + timedelta(days=value)
    except OverflowError:
        return None


def parse_study_datetime(buf: bytes):
    if len(buf) < 14:
        return None
    raw_value = struct.unpack_from("<d", buf, 6)[0]
    return parse_ole_datetime(raw_value)


def parse_study_datetime_candidates(buf: bytes):
    candidates = []
    seen = set()

    if len(buf) >= 14:
        raw_value = struct.unpack_from("<d", buf, 6)[0]
        parsed = parse_ole_datetime(raw_value)
        if parsed is not None:
            candidates.append(
                {
                    "offset": 6,
                    "encoding": "float64le",
                    "interpretation": "ole_datetime",
                    "confidence": "high",
                    "raw": raw_value,
                    "datetime": parsed.isoformat(sep=" "),
                    "reason": "Heidelberg chunk58 commonly stores study datetime as OLE datetime at offset 6.",
                }
            )
            seen.add((6, "ole_datetime"))

    for offset in range(0, len(buf) - 7):
        raw_double = struct.unpack_from("<d", buf, offset)[0]
        if not (30000 <= raw_double <= 80000):
            continue
        parsed_double = parse_ole_datetime(raw_double)
        key = (offset, "ole_datetime")
        if parsed_double is not None and key not in seen:
            confidence = "medium" if offset == 6 else "low"
            reason = "Plausible OLE datetime window match."
            if offset == 6:
                reason = "Heidelberg chunk58 commonly stores study datetime as OLE datetime at offset 6."
            candidates.append(
                {
                    "offset": offset,
                    "encoding": "float64le",
                    "interpretation": "ole_datetime",
                    "confidence": confidence,
                    "raw": raw_double,
                    "datetime": parsed_double.isoformat(sep=" "),
                    "reason": reason,
                }
            )
            seen.add(key)

    for offset in range(0, len(buf) - 3, 4):
        raw_u32 = struct.unpack_from("<I", buf, offset)[0]

        if 946684800 <= raw_u32 <= 1893456000:
            dt = datetime.utcfromtimestamp(raw_u32)
            key = (offset, "unix_seconds")
            if key not in seen:
                candidates.append(
                    {
                        "offset": offset,
                        "encoding": "uint32le",
                        "interpretation": "unix_seconds",
                        "confidence": "low",
                        "raw": raw_u32,
                        "datetime": dt.isoformat(sep=" ") + "Z",
                        "reason": "Numerically looks like a Unix timestamp, but may be overlap/noise.",
                    }
                )
                seen.add(key)

        if 30000 <= raw_u32 <= 80000:
            dt = parse_ole_datetime(raw_u32)
            key = (offset, "ole_days_int")
            if dt is not None and key not in seen:
                candidates.append(
                    {
                        "offset": offset,
                        "encoding": "uint32le",
                        "interpretation": "ole_days_int",
                        "confidence": "low",
                        "raw": raw_u32,
                        "datetime": dt.isoformat(sep=" "),
                        "reason": "Integer value falls in an OLE-date-like range, but lacks time-of-day precision.",
                    }
                )
                seen.add(key)

    candidates.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}[item["confidence"]], item["offset"]))
    return candidates


if __name__ == "__main__":
    buf = b'\x00\x00\x00\x00\x01\x00fff\xa6/{\xe6@\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xffn\x02\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\x88\x04\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x8a\x00\x00p\x99\x01\x00\xff\xff\xff\x03\x00\x00\x00\x00\x00\x00\x00'
    print(parse_study_datetime(buf))
    print(json.dumps(parse_study_datetime_candidates(buf), ensure_ascii=False, indent=2))
