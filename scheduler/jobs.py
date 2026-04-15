"""
jobs.py — 定时任务核心逻辑
  1. 从 DB 取 pending 视频
  2. 调用字幕 API 获取字幕
  3. AI 压缩字幕（universal_extraction_prompt）
  4. AI 改编为公众号文章（article_generation_prompt）
  5. 将结果写回 DB
"""
from __future__ import annotations

import os
import logging
import time
from pathlib import Path

import httpx
from openai import OpenAI

from db import fetch_pending_videos, update_video
from prompts import universal_extraction_prompt, article_generation_prompt

logger = logging.getLogger(__name__)

SUBTITLE_API_URL: str = os.environ.get("SUBTITLE_API_URL", "http://universal-sub-api:8822")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gemini-3.1-pro-preview")
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "1"))
# 两次 AI 调用之间的间隔（秒），避免触发限流
AI_CALL_INTERVAL: float = float(os.environ.get("AI_CALL_INTERVAL", "1"))
# 生成结果本地保存目录（空字符串表示不保存）
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", str(Path(__file__).parent.parent / "output"))

logger.info(
    "[env] SUBTITLE_API_URL=%s | OPENAI_BASE_URL=%s | OPENAI_MODEL=%s | "
    "BATCH_SIZE=%d | AI_CALL_INTERVAL=%s | OPENAI_API_KEY=%s",
    SUBTITLE_API_URL,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    BATCH_SIZE,
    AI_CALL_INTERVAL,
    ("***" + OPENAI_API_KEY[-4:]) if len(OPENAI_API_KEY) >= 4 else ("(未设置)" if not OPENAI_API_KEY else "***"),
)


def _get_ai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 AI")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _chat(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """单次对话调用，返回模型回复文本。"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 主任务入口（被 APScheduler 调用）
# ---------------------------------------------------------------------------

def process_videos() -> None:
    logger.info("[job] 开始处理待处理视频")
    videos = fetch_pending_videos(limit=BATCH_SIZE)
    if not videos:
        logger.info("[job] 暂无待处理视频")
        return

    logger.info("[job] 本次处理 %d 条", len(videos))
    for video in videos:
        _process_one(video)


def _process_one(video: dict) -> None:
    vid_id: int = video["id"]
    url: str = f"https://www.bilibili.com/video/{video['origin_id']}"
    logger.info("[job] 处理 id=%d origin_id=%s", vid_id, url)
    try:
        update_video(vid_id, "processing")

        # 1. 获取字幕
        subtitle = _fetch_subtitles(url)
        if subtitle is None:
            update_video(vid_id, "failed", ai_result="错误：未找到字幕")
            logger.warning("[job] id=%d 未找到字幕", vid_id)
            return

        raw_text: str = subtitle["subtitles_text"]
        client = _get_ai_client()

        # 2. 字幕压缩：将原始字幕提炼为结构化信息单元
        logger.info("[job] id=%d 开始字幕压缩, raw_text=%s", vid_id, raw_text[:100].replace("\n", "\\n"))
        compressed = _compress_subtitles(client, raw_text)
        time.sleep(AI_CALL_INTERVAL)

        # 3. 文章生成：基于压缩后的信息单元改编公众号长文
        logger.info("[job] id=%d 开始文章生成", vid_id)
        article = _generate_article(client, compressed)

        # 4. 写回数据库
        update_video(
            vid_id,
            "done",
            ai_result=article,
        )

        # 5. 保存到本地文件
        _save_to_file(vid_id, video.get("title", ""), compressed, article)
        logger.info("[job] id=%d 处理完成", vid_id)

    except Exception as exc:
        logger.exception("[job] id=%d 处理失败: %s", vid_id, exc)
        update_video(vid_id, "failed", ai_result=str(exc)[:500])


# ---------------------------------------------------------------------------
# 字幕获取
# ---------------------------------------------------------------------------

def _fetch_subtitles(url: str) -> dict | None:
    """调用本服务的 /fetch 接口获取字幕，返回 None 表示无字幕。"""
    with httpx.Client(timeout=120) as client:
        resp = client.get(f"{SUBTITLE_API_URL}/fetch", params={"url": url})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# 字幕压缩（第一步 AI）
# ---------------------------------------------------------------------------

def _compress_subtitles(client: OpenAI, subtitle_text: str) -> str:
    """使用 universal_extraction_prompt 将字幕压缩为结构化信息单元。"""
    system_prompt = universal_extraction_prompt.format(transcript=subtitle_text)
    user_prompt = "请将字幕压缩为结构化信息单元。"
    return _chat(client, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# 文章生成（第二步 AI）
# ---------------------------------------------------------------------------

def _generate_article(client: OpenAI, compressed_content: str) -> str:
    """使用 article_generation_prompt 将压缩后的信息单元改编为公众号文章。"""
    system_prompt = article_generation_prompt.format(article_content=compressed_content)
    user_prompt = "请将压缩后的信息单元改编为公众号文章。"
    return _chat(client, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# 本地文件保存
# ---------------------------------------------------------------------------

def _save_to_file(vid_id: int, title: str, compressed: str, article: str) -> None:
    """将压缩字幕和生成文章保存到本地 OUTPUT_DIR 目录。"""
    if not OUTPUT_DIR:
        return
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()[:50]
    prefix = f"{vid_id}_{safe_title}" if safe_title else str(vid_id)
    (out / f"{prefix}_compressed.txt").write_text(compressed, encoding="utf-8")
    (out / f"{prefix}_article.txt").write_text(article, encoding="utf-8")
    logger.info("[job] id=%d 已保存到 %s", vid_id, out / prefix)