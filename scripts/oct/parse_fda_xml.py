import xml.etree.ElementTree as ET
import pandas as pd
import json


class OCTStudy:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.data = []

    # ===== 内部工具 =====
    def _get(self, node, tag):
        child = node.find(tag)
        return child.text if child is not None else None

    # ===== 核心解析 =====
    def parse(self):
        self.data = []

        for oct_node in self.root.findall("Oct"):
            patient = oct_node.find("Patient")

            row = {
                # 👤 Patient
                "ID": patient.findtext("ID"),
                "LastName": patient.findtext("LastName"),
                "FirstName": patient.findtext("FirstName"),
                "BirthDate": patient.findtext("BirthDate"),

                # 📅 时间
                "Date": self._get(oct_node, "Date"),
                "Time": self._get(oct_node, "Time"),

                # 📊 核心参数
                "DatasetID": self._get(oct_node, "DatasetID"),
                "ScanWidth": self._get(oct_node, "ScanWidth"),
                "ScanCount": self._get(oct_node, "ScanCount"),
                "ScanHeight": self._get(oct_node, "ScanHeight"),
                "RealScanX": self._get(oct_node, "RealScanX"),
                "RealScanY": self._get(oct_node, "RealScanY"),
            }

            self.data.append(row)

        return self

    # ===== 转 DataFrame =====
    def to_dataframe(self):
        df = pd.DataFrame(self.data)

        numeric_cols = [
            "ScanWidth", "ScanCount", "ScanHeight",
            "RealScanX", "RealScanY"
        ]
        df[numeric_cols] = df[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        return df

    # ===== 打印 =====
    def print(self):
        for i, row in enumerate(self.data):
            print(f"\n===== OCT {i} =====")
            for k, v in row.items():
                print(f"{k}: {v}")

    # ===== JSON输出 =====
    def to_json(self, indent=2):
        return json.dumps(self.data, ensure_ascii=False, indent=indent)

    # ===== 计算 spacing（可选扩展）=====
    def add_spacing(self):
        df = self.to_dataframe()
        df["x_spacing"] = df["RealScanX"] / df["ScanWidth"]
        df["y_spacing"] = df["RealScanY"] / df["ScanCount"]
        return df
if __name__ == "__main__":
    study = OCTStudy(r"E:\Data\OCT\拓普康OCT\studyDriveLink\20251231_102549.xml")
    study.parse()
    # study.print()

    df = study.to_dataframe()
    print(df.to_string(index=False))

    # df_spacing = study.add_spacing()
    # print(df_spacing.to_string(index=False))