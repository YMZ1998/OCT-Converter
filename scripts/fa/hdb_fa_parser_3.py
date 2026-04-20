import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TimeAngle:
    raw: str
    minutes: int
    seconds: int
    centiseconds: int
    angle: int
    start: int
    end: int

    @property
    def elapsed_seconds(self) -> float:
        return (self.minutes * 60) + self.seconds + (self.centiseconds / 100.0)


PATTERN = re.compile(
    rb"("
    rb"(?:\d\x00){1,2}"
    rb":\x00"
    rb"(?:\d\x00){2}"
    rb"\.\x00"
    rb"(?:\d\x00){2}"
    rb"\x20\x00"
    rb"(?:\d\x00){1,3}"
    rb"\xb0\x00"
    rb")"
)

TEXT_PATTERN = re.compile(r"(\d+):(\d{2})\.(\d{2}) (\d{1,3})°")


def parse_match(match: re.Match[bytes]):
    try:
        text = match.group(1).decode("utf-16le")
        semantic_match = TEXT_PATTERN.fullmatch(text)
        if semantic_match is None:
            return None

        minutes = int(semantic_match.group(1))
        seconds = int(semantic_match.group(2))
        centiseconds = int(semantic_match.group(3))
        angle = int(semantic_match.group(4))

        if seconds >= 60 or centiseconds >= 100:
            return None
        if angle > 180:
            return None

        return TimeAngle(
            raw=text,
            minutes=minutes,
            seconds=seconds,
            centiseconds=centiseconds,
            angle=angle,
            start=match.start(),
            end=match.end(),
        )
    except Exception:
        return None


def extract_time_angles(payload: bytes) -> list[TimeAngle]:
    results: list[TimeAngle] = []
    for match in PATTERN.finditer(payload):
        parsed = parse_match(match)
        if parsed is not None:
            results.append(parsed)
    return results


if __name__ == "__main__":
    e2e = Path(
        r"E:\Data\OCT2\海德堡\KH902-R10-007-007003DME-V1-FFA\KH902-R10-007-007003DME-V1-FFA-OS.e2e"
        #  r"E:\Data\OCT\海德堡\海德堡FA.E2E"
    ).read_bytes()
    results = extract_time_angles(e2e)

    for result in results[:20]:
        print(
            f"{result.raw:<15} | "
            f"time={result.minutes:02}:{result.seconds:02}.{result.centiseconds:02} | "
            f"angle={result.angle:3} | "
            f"offset=0x{result.start:X}-0x{result.end:X}"
        )

    print("total:", len(results))
