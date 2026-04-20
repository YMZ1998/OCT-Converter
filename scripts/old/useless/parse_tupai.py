import sqlite3

db_path = r"E:\Data\OCT\图湃OCT.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
SELECT name FROM sqlite_master 
WHERE type='table' AND name NOT LIKE 'sqlite_%';
""")

tables = [t[0] for t in cursor.fetchall()]

for table in tables:
    print("\n==== 表:", table, "====")

    cursor.execute(f"PRAGMA table_info(`{table}`);")
    columns = cursor.fetchall()

    for col in columns:
        if col[2].upper() == "BLOB":
            print("发现BLOB字段:", col[1])

cursor.execute("SELECT BinaryData FROM Table2025111290346 LIMIT 1;")
blob = cursor.fetchone()[0]

print("前32字节:")
print(blob[:32])
cursor.execute("SELECT rowid, length(BinaryData) FROM Table2025111290346 LIMIT 10;")
print(cursor.fetchall())

cursor.execute("SELECT BinaryData FROM Table2025111290346 WHERE rowid=5;")
blob = cursor.fetchone()[0]

print("前8字节:", blob[:8])
print("是否zlib头:", blob[:2] == b'\x78\x9c')

conn.close()