"""
db.py — PostgreSQL 连接与 CRUD 操作
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor, RealDictRow

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://superuser:123456@host.docker.internal:5432/postgres",
)


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def fetch_pending_videos(limit: int = 5) -> list[RealDictRow]:
    """获取状态为 pending 的视频，按创建时间升序。"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, platform, origin_id, title, description,
                       ptime, play_count, share_count, comment_count,
                       ai_result, ai_status, meta_data, created_at
                FROM video_data
                WHERE ai_status = '0'
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def update_video(video_id: int, ai_status: str, **fields) -> None:
    """更新视频的 ai_status 及任意额外字段。"""
    set_parts = ["ai_status = %s"]
    values: list = [ai_status]
    for key, val in fields.items():
        set_parts.append(f"{key} = %s")
        values.append(val)
    values.append(video_id)

    sql = f"UPDATE video_data SET {', '.join(set_parts)} WHERE id = %s"  # noqa: S608
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, values)
        conn.commit()
