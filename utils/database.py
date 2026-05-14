import sqlite3
import os


DB_PATH = "data/copilot.db"


def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_analysis(title, content):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO analysis_history (title, content) VALUES (?, ?)",
        (title, content)
    )

    conn.commit()
    conn.close()


def get_analysis_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, title, content, created_at FROM analysis_history ORDER BY created_at DESC")
    rows = cursor.fetchall()

    conn.close()

    return rows