import xml.etree.ElementTree as ET

import pandas as pd


def parse_oct_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    for oct_node in root.findall("Oct"):
        print("-"*100)
        print("DatasetID:", oct_node.findtext("DatasetID"))
        print("FundusCaptureMode:", oct_node.findtext("FundusCaptureMode"))
        print("ScanMode:", oct_node.findtext("ScanMode"))
        print("LREye:", oct_node.findtext("LREye"))
        print("ScanWidth:", oct_node.findtext("ScanWidth"))
        print("ScanCount:", oct_node.findtext("ScanCount"))
        print("ScanHeight:", oct_node.findtext("ScanHeight"))
        print("RealScanX:", oct_node.findtext("RealScanX"))
        print("RealScanY:", oct_node.findtext("RealScanY"))
        print("Date:", oct_node.findtext("Date"))
        print("Time:", oct_node.findtext("Time"))
        print("ResolutionZ:", oct_node.findtext("ResolutionZ"))
        print("ProductName:", oct_node.findtext("ProductName"))

        patient = oct_node.find("Patient")

        def get(tag):
            node = oct_node.find(tag)
            return node.text if node is not None else None

        row = {
            # 👤 patient info
            "FirstName": patient.findtext("FirstName"),
            "LastName": patient.findtext("LastName"),
            "BirthDate": patient.findtext("BirthDate"),
            "ID": patient.findtext("ID"),

            # 📊 scan params
            "ScanWidth": get("ScanWidth"),
            "ScanCount": get("ScanCount"),
            "ScanHeight": get("ScanHeight"),
            "RealScanX": get("RealScanX"),
            "RealScanY": get("RealScanY"),
        }
        print(row)
        rows.append(row)

    return pd.DataFrame(rows)


# ===== 使用 =====
df = parse_oct_xml(r"E:\Data\OCT\拓普康OCT\studyDriveLink\20251231_102549.xml")

# print(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False))
