import datetime
from typing import List, Union

class HeidelbergTimeParser:
    """
    Heidelberg OCT acquisitionTime解析器
    将 64位 acquisitionTime（FILETIME 风格，100纳秒，自1601-01-01 UTC）
    转换为 UTC 和 CST 可读时间
    """

    # FILETIME 到 Unix epoch 偏移（秒）
    EPOCH_AS_FILETIME = 116444736000000000
    HUNDREDS_OF_NANOSECONDS = 10_000_000

    @staticmethod
    def filetime_to_datetime(filetime: int) -> datetime.datetime:
        """
        单个 acquisitionTime -> UTC datetime
        """
        timestamp = (filetime - HeidelbergTimeParser.EPOCH_AS_FILETIME) / HeidelbergTimeParser.HUNDREDS_OF_NANOSECONDS
        return datetime.datetime.utcfromtimestamp(timestamp)

    @staticmethod
    def to_cst(dt_utc: datetime.datetime) -> datetime.datetime:
        """
        UTC datetime -> 中国标准时间 CST
        """
        return dt_utc + datetime.timedelta(hours=8)

    @classmethod
    def parse_single(cls, acquisitionTime: int) -> dict:
        """
        解析单个 acquisitionTime
        返回字典 {'utc': datetime, 'cst': datetime}
        """
        dt_utc = cls.filetime_to_datetime(acquisitionTime)
        dt_cst = cls.to_cst(dt_utc)
        return {'utc': dt_utc, 'cst': dt_cst}

    @classmethod
    def parse_list(cls, acquisitionTimes: List[int]) -> List[dict]:
        """
        批量解析 acquisitionTime 列表
        返回 [{utc: ..., cst: ...}, ...]
        """
        return [cls.parse_single(t) for t in acquisitionTimes]


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    # 单个解析
    acquisitionTime = 134025603671240000
    parsed = HeidelbergTimeParser.parse_single(acquisitionTime)
    print("UTC 时间:", parsed['utc'])
    print("CST 时间:", parsed['cst'])

    # 批量解析
    acquisitionTimes = [
        134025603671240000,
        134025604001010000,
        134025601524160000
    ]
    parsed_list = HeidelbergTimeParser.parse_list(acquisitionTimes)
    for i, p in enumerate(parsed_list):
        print(f"Index {i} -> UTC: {p['utc']}, CST: {p['cst']}")