"""
get_data_jobs.py — 定时抓取 Bilibili UP主 最近视频元数据并存入 DB

流程：
  1. 遍历 WATCH_UIDS 列表中的每个 UP主 UID
  2. 按 pubdate（最新发布）排序分页拉取，遇到超过 DAYS_KEEP 天的视频立即停止翻页
  3. 以 origin_id（bvid）为唯一键，INSERT … ON CONFLICT DO NOTHING 去重写入
  4. 接口报错时指数退避重试，最多 MAX_RETRIES 次

环境变量：
  TIKHUB_API_KEY   TikHub API 密钥（必填）
  WATCH_UIDS       逗号分隔的 UID 列表，如 "15324420,12345678"（默认 "15324420"）
  DAYS_KEEP        保留最近几天的视频（默认 7）
  FETCH_CRON_EXPR  抓取任务的 cron 表达式（默认 "0 * * * *"，每小时）
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

from db import get_conn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TIKHUB_API_KEY: str = os.environ.get("TIKHUB_API_KEY", "")
TIKHUB_BASE_URL: str = "https://api.tikhub.io"

FETCH_ORDER: str = "pubdate"  # 按最新发布排序，方便遇到旧数据时提前停止
MAX_RETRIES: int = 5
RETRY_BASE_DELAY: float = 2.0  # 指数退避基础秒数

_raw_uids: str = os.environ.get("WATCH_UIDS", "15324420")
WATCH_UIDS: list[str] = [u.strip() for u in _raw_uids.split(",") if u.strip()]

DAYS_KEEP: int = int(os.environ.get("DAYS_KEEP", "7"))


# ---------------------------------------------------------------------------
# API 调用（带重试）
# ---------------------------------------------------------------------------

def _fetch_page(uid: str, pn: int, *, client: httpx.Client) -> dict:
    """
    请求单页数据，失败时指数退避重试，最多 MAX_RETRIES 次。
    成功后返回 data.data 层级的 dict（含 list.vlist 和 page）。
    """
    url = f"{TIKHUB_BASE_URL}/api/v1/bilibili/web/fetch_user_post_videos"
    params = {"uid": uid, "pn": pn, "order": FETCH_ORDER}
    headers = {"Authorization": f"Bearer {TIKHUB_API_KEY}"}

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            # 外层 wrapper code
            if payload.get("code") != 200:
                raise ValueError(
                    f"API 外层 code={payload.get('code')} msg={payload.get('message')}"
                )

            # Bilibili 内层 code
            inner_code = payload["data"].get("code")
            if inner_code != 0:
                raise ValueError(
                    f"Bilibili 内层 code={inner_code} msg={payload['data'].get('message')}"
                )

            return payload["data"]["data"]

        except Exception as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "[fetch_page] uid=%s pn=%d 第 %d/%d 次失败: %s，%.1f 秒后重试",
                uid, pn, attempt, MAX_RETRIES, exc, delay,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    raise RuntimeError(
        f"uid={uid} pn={pn} 已重试 {MAX_RETRIES} 次仍然失败"
    ) from last_exc


def fetch_recent_videos(uid: str, cutoff_ts: int, *, client: httpx.Client) -> list[dict]:
    """
    分页拉取 uid 的视频列表，仅保留 created >= cutoff_ts 的视频。
    因为结果按发布时间倒序，一旦遇到 created < cutoff_ts 的条目即停止翻页。
    """
    collected: list[dict] = []
    pn = 1

    while True:
        data = _fetch_page(uid, pn, client=client)
        vlist: list[dict] = data.get("list", {}).get("vlist") or []

        if not vlist:
            logger.info("uid=%s pn=%d 返回空列表，停止翻页", uid, pn)
            break

        stop = False
        for video in vlist:
            if video.get("created", 0) < cutoff_ts:
                stop = True
                break
            collected.append(video)

        if stop:
            break

        # 判断是否还有下一页
        page_info = data.get("page") or {}
        total = page_info.get("count", 0)
        ps = page_info.get("ps", 25)
        if pn * ps >= total:
            break

        pn += 1

    return collected


# ---------------------------------------------------------------------------
# DB 写入
# ---------------------------------------------------------------------------

def _build_row(video: dict) -> dict:
    """从接口 vlist 条目提取入库字段。"""
    meta = video.get("meta") or {}
    stat = meta.get("stat") or {}
    return {
        "platform": "bilibili",
        "origin_id": video["bvid"],
        "title": video.get("title") or "",
        "description": video.get("description") or "",
        "ptime": video.get("created"),
        "favorite_count": stat.get("favorite"),
        "play_count": video.get("play"),
        "share_count": stat.get("share"),
        "comment_count": video.get("comment"),
        "meta_data": json.dumps(video, ensure_ascii=False),
    }


def save_videos(videos: list[dict]) -> int:
    """
    批量写入 video_data 表，origin_id 重复时自动跳过。
    返回实际插入行数。
    """
    if not videos:
        return 0

    sql = """
        INSERT INTO video_data
            (platform, origin_id, title, description, ptime,
             play_count, share_count, comment_count, favorite_count, meta_data)
        VALUES
            (%(platform)s, %(origin_id)s, %(title)s, %(description)s, %(ptime)s,
             %(play_count)s, %(share_count)s, %(comment_count)s, %(favorite_count)s, %(meta_data)s::jsonb)
        ON CONFLICT (origin_id) DO NOTHING
    """
    inserted = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for video in videos:
                row = _build_row(video)
                cur.execute(sql, row)
                inserted += cur.rowcount
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_fetch_job() -> None:
    """定时任务主函数：遍历 WATCH_UIDS，抓取并存储最近 DAYS_KEEP 天的视频。"""
    if not TIKHUB_API_KEY:
        logger.error("TIKHUB_API_KEY 未设置，跳过抓取任务")
        return

    cutoff_ts = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=DAYS_KEEP)).timestamp()
    )
    cutoff_str = datetime.fromtimestamp(cutoff_ts).strftime("%Y-%m-%d %H:%M:%S")
    logger.info("开始抓取任务 | UIDs=%s | 保留 %d 天 | cutoff=%s", WATCH_UIDS, DAYS_KEEP, cutoff_str)

    total_fetched = 0
    total_new = 0

    with httpx.Client() as client:
        for uid in WATCH_UIDS:
            try:
                videos = fetch_recent_videos(uid, cutoff_ts, client=client)
                new = save_videos(videos)
                total_fetched += len(videos)
                total_new += new
                logger.info("uid=%s | 抓取 %d 条 | 新增入库 %d 条", uid, len(videos), new)
            except Exception as exc:
                logger.error("uid=%s 抓取失败: %s", uid, exc, exc_info=True)

    logger.info("抓取任务完成 | 共抓取 %d 条 | 新增入库 %d 条", total_fetched, total_new)
