# db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("rag_state.sqlite")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            openai_file_id TEXT NOT NULL
        )
        """)
        con.commit()

def get_doc(path: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT path, sha256, openai_file_id FROM docs WHERE path=?", (path,))
        row = cur.fetchone()
        return row  # (path, sha256, openai_file_id) o None

def upsert_doc(path: str, sha256: str, openai_file_id: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        INSERT INTO docs(path, sha256, openai_file_id)
        VALUES(?,?,?)
        ON CONFLICT(path) DO UPDATE SET
          sha256=excluded.sha256,
          openai_file_id=excluded.openai_file_id
        """, (path, sha256, openai_file_id))
        con.commit()