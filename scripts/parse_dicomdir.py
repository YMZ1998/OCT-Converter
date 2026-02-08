import pydicom

if __name__ == "__main__":

    ds = pydicom.dcmread(r"E:\Data\OCT\蔡司OCT\DataFiles\E195\DICOMDIR")

    records = ds.DirectoryRecordSequence

    for r in records:
        print("Type:", r.DirectoryRecordType)

        if r.DirectoryRecordType == "IMAGE":
            print("File:", r.ReferencedFileID)
