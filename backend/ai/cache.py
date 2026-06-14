import sqlite3
from pathlib import Path
import json
import time

CACHE_DIR = Path(".cache")
CACHE_DB = CACHE_DIR / "tool_cache.sqlite"

def init_cache():
    CACHE_DIR.mkdir(exist_ok=True)

    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_cache (
                key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
        """)

def make_cache_key(tool_name: str, params: dict) -> str:
    params_json = json.dumps(params, sort_keys=True)
    return f"{tool_name}:{params_json}"

def read_cache(tool_name: str, params: dict):
    init_cache()

    key = make_cache_key(tool_name, params)
    current_time = int(time.time()) #its time key created

    with sqlite3.connect(CACHE_DB) as conn:
        row = conn.execute(
            """
            SELECT response_json, expires_at
            FROM tool_cache
            WHERE key = ?
            """,
            (key,)
        ).fetchone()

    if row is None:
        return False, None

    response_json, expires_at = row

    if expires_at <= current_time:
        #delete the cache update it rightly
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute(
                """
                DELETE FROM tool_cache
                WHERE key = ?
                """,
                (key,)
            )
        return False, None

    return True, response_json

def write_cache(tool_name: str, params: dict, response_json: str, ttl_seconds: int):
    init_cache()

    key = make_cache_key(tool_name, params)
    created_at = int(time.time())
    expires_at = created_at + ttl_seconds
    params_json = json.dumps(params, sort_keys=True)

    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tool_cache
            (key, tool_name, params_json, response_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, tool_name, params_json, response_json, created_at, expires_at)
        )
