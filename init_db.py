"""
初始化示例 SQLite 数据库。

运行：
    python init_db.py
"""
import os
import sqlite3

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))
DB_PATH = os.getenv("DB_PATH", "students.db")
DB_FILE = DB_PATH if os.path.isabs(DB_PATH) else os.path.join(BASE_DIR, DB_PATH)


def init_database() -> None:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS scores;
        DROP TABLE IF EXISTS students;
        DROP TABLE IF EXISTS classes;

        CREATE TABLE classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            teacher TEXT NOT NULL
        );

        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            city TEXT NOT NULL,
            FOREIGN KEY (class_id) REFERENCES classes(id)
        );

        CREATE TABLE scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
        """
    )

    cursor.executemany(
        "INSERT INTO classes (name, teacher) VALUES (?, ?)",
        [
            ("软件一班", "张老师"),
            ("软件二班", "李老师"),
            ("人工智能班", "王老师"),
        ],
    )

    cursor.executemany(
        "INSERT INTO students (name, age, gender, class_id, city) VALUES (?, ?, ?, ?, ?)",
        [
            ("张三", 19, "男", 1, "广州"),
            ("李四", 21, "女", 1, "深圳"),
            ("王五", 22, "男", 2, "佛山"),
            ("赵六", 20, "女", 2, "东莞"),
            ("钱七", 23, "男", 3, "珠海"),
            ("孙八", 18, "女", 3, "广州"),
            ("周九", 24, "男", 1, "中山"),
            ("吴十", 21, "女", 2, "惠州"),
        ],
    )

    cursor.executemany(
        "INSERT INTO scores (student_id, subject, score) VALUES (?, ?, ?)",
        [
            (1, "Python", 88), (1, "数据库", 92),
            (2, "Python", 95), (2, "数据库", 86),
            (3, "Python", 76), (3, "数据库", 81),
            (4, "Python", 90), (4, "数据库", 89),
            (5, "Python", 84), (5, "数据库", 93),
            (6, "Python", 79), (6, "数据库", 85),
            (7, "Python", 91), (7, "数据库", 78),
            (8, "Python", 87), (8, "数据库", 90),
        ],
    )

    conn.commit()
    conn.close()
    print(f"示例数据库已生成：{DB_FILE}")


if __name__ == "__main__":
    init_database()
