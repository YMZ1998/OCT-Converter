import xml.etree.ElementTree as ET


def parse_oct_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    oct_list = []

    for oct_node in root.findall("Oct"):
        # ---------- Patient ----------
        patient = oct_node.find("Patient")

        patient_info = {
            "id": patient.findtext("ID"),
            "last_name": patient.findtext("LastName"),
            "first_name": patient.findtext("FirstName"),
            "sex": patient.findtext("Sex"),
            "birth_date": patient.findtext("BirthDate"),
        }

        # ---------- OCT Info ----------
        oct_info = {
            "oct_no": oct_node.attrib.get("No"),
            "patient": patient_info,
            "dataset_id": oct_node.findtext("DatasetID"),
            "session_id": oct_node.findtext("SessionID"),
            "date": oct_node.findtext("Date"),
            "time": oct_node.findtext("Time"),
            "scan_mode": oct_node.findtext("ScanMode"),
            "lr_eye": oct_node.findtext("LREye"),  # 1=左眼, 2=右眼
            "scan_width": int(oct_node.findtext("ScanWidth")),
            "scan_height": int(oct_node.findtext("ScanHeight")),
            "scan_count": int(oct_node.findtext("ScanCount")),
            "real_scan_x": float(oct_node.findtext("RealScanX")),
            "real_scan_y": float(oct_node.findtext("RealScanY")),
            "retinal_iq": int(oct_node.findtext("RetinalImageQuality")),
            "choroidal_iq": int(oct_node.findtext("ChoroidalImageQuality")),
            "product": oct_node.findtext("ProductName"),
        }

        oct_list.append(oct_info)

    return oct_list

def extract_ascii_strings(data, min_len=4):
    strings = []
    buf = b""

    for b in data:
        if 32 <= b <= 126:  # 可打印 ASCII
            buf += bytes([b])
        else:
            if len(buf) >= min_len:
                strings.append(buf.decode("ascii", errors="ignore"))
            buf = b""

    if len(buf) >= min_len:
        strings.append(buf.decode("ascii", errors="ignore"))

    return strings

def parse_foctarc_strings(strings):
    records = []
    i = 0

    while i < len(strings):
        if strings[i] == "FOCTARC":
            record = {
                "patient_id": strings[i+1],
                "birth_date": strings[i+2],
                "exam_patient_id": strings[i+3],
                "exam_date": strings[i+4],
                "start_time": strings[i+5],
                "end_time": strings[i+6],
                "exam_datetime": strings[i+7],
                "exam_id": strings[i+8],
                "scan_type": strings[i+9],
                "modality": strings[i+10],
                "acq_datetime": strings[i+11],
                "software_version": strings[i+12],
                "device_ip": strings[i+13],
                "dicom_version": strings[i+14],
                "device_model": strings[i+15],
            }
            records.append(record)
            i += 16
        else:
            i += 1

    return records

if __name__ == "__main__":
    # xml_path = r"E:\Data\OCT\拓普康OCT\studyDriveLink\19700101_080000.xml"
    xml_path = r"E:\Data\OCT\拓普康OCT\studyDriveLink\20251231_102549.xml"
    oct_infos = parse_oct_xml(xml_path)
    print(oct_infos)

    for info in oct_infos:
        print(
            info["oct_no"],
            info["patient"],
            info["scan_mode"],
            "左眼" if info["lr_eye"] == "1" else "右眼",
            "Quality:", info["retinal_iq"]
        )
    with open(r"E:\Data\OCT\拓普康OCT\FILELIST", "rb") as f:
        data = f.read()

    print(data)

    strings = extract_ascii_strings(data)
    for s in strings:
        print(s)

    records = parse_foctarc_strings(strings)
    for r in records:
        print(r)

