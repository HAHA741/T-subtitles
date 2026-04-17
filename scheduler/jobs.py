"""
jobs.py — 定时任务核心逻辑
  1. 从 DB 取 pending 视频
  2. 调用字幕 API 获取字幕
  3. AI 压缩字幕（universal_extraction_prompt）
  4. AI 改编为公众号文章（article_generation_prompt）
  5. AI 生成候选标题（title_generation_prompt，结构化）
  6. AI 生成配图方案（illustration_prompt，结构化）
  7. 将结果写回 DB
"""
from __future__ import annotations

import os
import logging
import time
from pathlib import Path

import base64
import json

import httpx
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from db import fetch_pending_videos, update_video
from prompts import (
    universal_extraction_prompt,
    article_generation_prompt,
    title_generation_prompt,
    illustration_prompt,
)

logger = logging.getLogger(__name__)

SUBTITLE_API_URL: str = os.environ.get("SUBTITLE_API_URL", "http://universal-sub-api:8822")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gemini-3.1-pro-preview")
IMAGE_OPENAI_MODEL: str = os.environ.get("IMAGE_OPENAI_MODEL", "doubao-seedream-5-0-260128")
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


# ---------------------------------------------------------------------------
# Pydantic 结构化输出模型
# ---------------------------------------------------------------------------

class TitleCandidate(BaseModel):
    title: str = Field(description="标题文本，不含引号")
    prototype: str = Field(description="使用的原型：悬念钩子/认知颠覆/一口气/数字冲击/亲身经历/组合")
    reason: str = Field(description="一句话说明选择该角度的理由")
    passed_t1: bool = Field(description="是否通过 T1 硬性规则检查")
    passed_t2: bool = Field(description="是否通过 T2 吸引力检查")
    passed_t3: bool = Field(description="是否通过 T3 诚实度检查")


class TitleOutput(BaseModel):
    candidates: list[TitleCandidate] = Field(description="8-10 个候选标题")
    top3: list[str] = Field(description="推荐的前3个标题，按推荐优先级排序")


class IllustrationItem(BaseModel):
    index: int = Field(description="图片序号，从1开始")
    position: str = Field(description="插入位置描述，如「第一段之后」或「封面」")
    visual_concept: str = Field(description="视觉意象，中文描述")
    prompt_en: str = Field(description="英文图片生成提示词")
    aspect_ratio: str = Field(description="画幅比例，如 2.35:1 或 16:9")
    url: str = Field(default="", description="生成后的图片 URL，由流水线回填")


class IllustrationOutput(BaseModel):
    article_type: str = Field(description="文章类型")
    style: str = Field(description="全局视觉风格名称")
    color_scheme: str = Field(description="色彩方案描述")
    illustrations: list[IllustrationItem] = Field(description="配图方案列表")


# ---------------------------------------------------------------------------
# AI 客户端
# ---------------------------------------------------------------------------

def _get_ai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 AI")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _get_instructor_client() -> instructor.Instructor:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 AI")
    raw = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    # Mode.MD_JSON: 通过 prompt 引导输出 Markdown 代码块内的 JSON，
    # 不依赖 tool_call / response_format，对中转端点兼容性最好。
    return instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)


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

def _generate_img(client: OpenAI, prompt: str, save_path: Path | None = None) -> str:
    """调用图像生成接口，返回可直接用于 Markdown 的引用字符串。

    - 响应含 url 字段：直接返回 URL
    - 响应含 b64_json 字段：解码后保存为 PNG，返回本地文件路径
    """
    response = client.images.generate(
        model=IMAGE_OPENAI_MODEL,
        prompt=prompt,
        size="1024x1024",
        # quality="high",
        # response_format="url",
        # output_format="jpeg",
        # output_compression=80,
        # n=1,
    )
    item = response.data[0]
    if item.url:
        logger.info("[img] 生成完成(url), url=%s", item.url)
        return item.url
    if item.b64_json:
        if save_path is None:
            raise RuntimeError("响应为 b64_json 但未提供 save_path，无法保存图片")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(base64.b64decode(item.b64_json))
        logger.info("[img] 生成完成(b64), 已保存到 %s", save_path)
        return str(save_path)
    raise RuntimeError("响应中既无 url 也无 b64_json")


def _generate_all_images(illus_output: IllustrationOutput, vid_id: int) -> IllustrationOutput:
    """遍历配图方案，为每张图调用图像生成接口，回填 url 字段。失败的图跳过不报错。

    b64_json 响应时将图片保存到 OUTPUT_DIR/images/{vid_id}_{index}.png，
    item.url 存储相对于 OUTPUT_DIR 的路径（供 _final.md 引用）。
    """
    img_client = _get_ai_client()
    img_dir = (Path(OUTPUT_DIR) / "images") if OUTPUT_DIR else None
    for item in illus_output.illustrations:
        try:
            time.sleep(AI_CALL_INTERVAL)
            save_path = (img_dir / f"{vid_id}_{item.index}.png") if img_dir else None
            result = _generate_img(img_client, item.prompt_en, save_path=save_path)
            # b64_json 路径转为相对路径，url 直接使用
            if save_path and result == str(save_path):
                item.url = f"images/{vid_id}_{item.index}.png"
            else:
                item.url = result
            logger.info("[img] 图片 %d 生成成功, ref=%s", item.index, item.url)
        except Exception as exc:
            logger.warning("[img] 图片 %d 生成失败（跳过）: %s", item.index, exc)
    return illus_output


def _build_markdown(article: str, illus_output: IllustrationOutput | None) -> str:
    """将文章正文与已生成 URL 的配图合并为 Markdown 文档。

    策略：
    - position 含「封面」的图片放在文章最顶部（作为 header image）
    - 其余有效图片（有 URL）按段落均匀分布插入正文
    - 均匀插入：将正文段落等分为 (n+1) 份，在每个分割点后插图
    """
    if not illus_output:
        return article

    illustrations = illus_output.illustrations

    # 分离封面和正文配图
    cover_items = [i for i in illustrations if "封面" in i.position and i.url]
    body_items = [i for i in illustrations if "封面" not in i.position and i.url]

    # 按 index 排序
    body_items.sort(key=lambda x: x.index)

    # 构建封面头部
    header_parts: list[str] = []
    if cover_items:
        c = cover_items[0]
        header_parts.append(f"![封面 — {c.visual_concept}]({c.url})\n")

    # 将正文按段落分割（空行分段）
    paragraphs = [p for p in article.split("\n\n") if p.strip()]

    # 均匀插入：在 len(paragraphs) 中均匀选 len(body_items) 个插入点
    if body_items:
        n = len(paragraphs)
        m = len(body_items)
        # 插入点为 n/(m+1), 2n/(m+1), ... 取整后的段落索引
        insert_after: dict[int, IllustrationItem] = {}
        for k, item in enumerate(body_items, start=1):
            idx = min(round(n * k / (m + 1)), n - 1)
            # 避免同一段落插多张图
            while idx in insert_after and idx < n - 1:
                idx += 1
            insert_after[idx] = item

        result_parts: list[str] = []
        for i, para in enumerate(paragraphs):
            result_parts.append(para)
            if i in insert_after:
                item = insert_after[i]
                result_parts.append(f"\n![{item.visual_concept}]({item.url})\n")
        body_md = "\n\n".join(result_parts)
    else:
        body_md = "\n\n".join(paragraphs)

    return "".join(header_parts) + body_md

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
        logger.info("[job] id=%d 文章全文 ↓↓↓\n%s\n[job] id=%d 文章全文 ↑↑↑", vid_id, article, vid_id)

        # 4. 拆分文章正文与尾部检查段
        ai_article, ai_check = _split_article(article)

        # 5. 标题生成（结构化，失败不阻断主流程）
        title_output: TitleOutput | None = None
        try:
            logger.info("[job] id=%d 开始标题生成", vid_id)
            time.sleep(AI_CALL_INTERVAL)
            title_output = _generate_titles_structured(_get_instructor_client(), ai_article)
            logger.info("[job] id=%d 标题生成完成, top3=%s", vid_id, title_output.top3)
        except Exception as exc:
            logger.warning("[job] id=%d 标题生成失败（不阻断主流程）: %s", vid_id, exc)

        # 6. 配图方案生成（结构化，失败不阻断主流程）
        illus_output: IllustrationOutput | None = None
        try:
            logger.info("[job] id=%d 开始配图方案生成", vid_id)
            time.sleep(AI_CALL_INTERVAL)
            illus_output = _generate_illustrations_structured(_get_instructor_client(), ai_article)
            logger.info("[job] id=%d 配图方案生成完成, count=%d", vid_id, len(illus_output.illustrations))
        except Exception as exc:
            logger.warning("[job] id=%d 配图方案生成失败（不阻断主流程）: %s", vid_id, exc)

        # 7. 逐张生成图片，回填 URL（失败的图自动跳过）
        if illus_output:
            try:
                logger.info("[job] id=%d 开始逐张生成图片, total=%d", vid_id, len(illus_output.illustrations))
                illus_output = _generate_all_images(illus_output, vid_id)
            except Exception as exc:
                logger.warning("[job] id=%d 图片生成阶段异常（不阻断主流程）: %s", vid_id, exc)

        # 8. 写回数据库
        update_video(
            vid_id,
            "done",
            ai_result=article,
            ai_article=ai_article,
            ai_check=ai_check,
            ai_compressed=compressed,
        )

        # 9. 合并图文，保存到本地文件
        final_md = _build_markdown(ai_article, illus_output)
        _save_to_file(vid_id, video.get("title", ""), compressed, ai_article, ai_check, title_output, illus_output, final_md)
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
# 标题生成（第三步 AI，结构化）
# ---------------------------------------------------------------------------

def _generate_titles_structured(client: instructor.Instructor, article_text: str) -> TitleOutput:
    """使用 title_generation_prompt 生成结构化候选标题列表。"""
    system_prompt = title_generation_prompt.format(article_content=article_text)
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        response_model=TitleOutput,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请为这篇文章生成候选标题。"},
        ],
        temperature=1,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# 配图方案生成（第四步 AI，结构化）
# ---------------------------------------------------------------------------

def _generate_illustrations_structured(client: instructor.Instructor, article_text: str) -> IllustrationOutput:
    """使用 illustration_prompt 生成结构化配图方案。"""
    system_prompt = illustration_prompt.format(article_content=article_text)
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        response_model=IllustrationOutput,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请为这篇文章生成配图方案。"},
        ],
        temperature=1,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# 文章拆分
# ---------------------------------------------------------------------------

ARTICLE_FOOTER_MARKER = "感谢阅读，下期见。"


def _split_article(article: str) -> tuple[str, str]:
    """按固定尾部分割符拆分文章，返回 (正文, 尾部检查段)。
    使用 rfind 取最后一次出现，避免正文中引用该句时提前截断。
    """
    idx = article.rfind(ARTICLE_FOOTER_MARKER)
    if idx == -1:
        return article.strip(), ""
    split_pos = idx + len(ARTICLE_FOOTER_MARKER)
    return article[:split_pos].strip(), article[split_pos:].strip()


# ---------------------------------------------------------------------------
# 本地文件保存
# ---------------------------------------------------------------------------

def _save_to_file(
    vid_id: int,
    title: str,
    compressed: str,
    ai_article: str,
    ai_check: str,
    title_output: TitleOutput | None = None,
    illus_output: IllustrationOutput | None = None,
    final_md: str | None = None,
) -> None:
    """将压缩字幕和生成文章保存到本地 OUTPUT_DIR 目录。"""
    if not OUTPUT_DIR:
        return
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()[:50]
    prefix = f"{vid_id}_{safe_title}" if safe_title else str(vid_id)
    (out / f"{prefix}_compressed.txt").write_text(compressed, encoding="utf-8")
    (out / f"{prefix}_article.txt").write_text(ai_article, encoding="utf-8")
    if ai_check:
        (out / f"{prefix}_check.txt").write_text(ai_check, encoding="utf-8")
    if title_output:
        (out / f"{prefix}_titles.json").write_text(
            json.dumps(title_output.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if illus_output:
        (out / f"{prefix}_illustrations.json").write_text(
            json.dumps(illus_output.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if final_md:
        (out / f"{prefix}_final.md").write_text(final_md, encoding="utf-8")
    logger.info("[job] id=%d 已保存到 %s", vid_id, out / prefix)