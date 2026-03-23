import sqlite3

db_path = r"E:\Data\OCT\图湃OCT.db"

def parse_all_tables(db_path, preview_rows=5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有用户表
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%';
    """)
    tables = [t[0] for t in cursor.fetchall()]

    if not tables:
        print("⚠ 没有找到用户表")
        return

    for table in tables:
        print("\n" + "="*60)
        print(f"📌 表名: {table}")
        print("="*60)

        # 获取表结构
        cursor.execute(f"PRAGMA table_info(`{table}`);")
        columns = cursor.fetchall()
        print("字段结构:")
        for col in columns:
            print(f"  {col[1]} | 类型: {col[2]} | 非空: {col[3]} | 主键: {col[5]}")

        # 获取索引
        cursor.execute(f"PRAGMA index_list(`{table}`);")
        indexes = cursor.fetchall()
        if indexes:
            print("索引:")
            for idx in indexes:
                print("  ", idx)
        else:
            print("无索引")

        # 打印前几行数据
        cursor.execute(f"SELECT * FROM `{table}` LIMIT {preview_rows};")
        rows = cursor.fetchall()
        if rows:
            col_names = [desc[0] for desc in cursor.description]
            print(f"前 {preview_rows} 行数据:")
            print("  列名:", col_names)
            for row in rows:
                # 如果字段是 BLOB，只显示长度
                row_display = []
                for cell in row:
                    if isinstance(cell, bytes):
                        row_display.append(f"<BLOB {len(cell)} bytes>")
                    else:
                        row_display.append(cell)
                print("  ", row_display)
        else:
            print("  (空表)")

    conn.close()
    print("\n🎉 全部表解析完成")

if __name__ == "__main__":
    parse_all_tables(db_path)