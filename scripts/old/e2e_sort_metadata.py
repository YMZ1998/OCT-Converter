from scripts.old.parse_e2e_time import HeidelbergTimeParser


def sort_metadata(metadata):
    """
    根据 laterality + scan_pattern + numImages 分组
    """
    bscans = metadata.get("bscan_data", [])
    # 先解析时间并附加到 bscan 对象
    for bscan in bscans:
        acquisitionTime = bscan.get('acquisitionTime', 0)
        parsed = HeidelbergTimeParser.parse_single(acquisitionTime)
        bscan['parsed_time'] = parsed['cst']  # 或 parsed['utc'] 根据需要

    # 按时间排序
    bscans_sorted = sorted(bscans, key=lambda x: x['parsed_time'])
    metadata["bscan_data"] = bscans_sorted
    return metadata
