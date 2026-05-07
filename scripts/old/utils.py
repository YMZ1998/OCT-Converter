from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value).split("\x00", 1)[0]
    text = "".join(c for c in text if c.isprintable())
    return text.strip()
