from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import os
import re
import shutil
import tempfile
import glob

COOKIE_FILE = "/app/cookies.txt"
# 从浏览器读取 cookies，本地调试用，例如 "firefox" 或 "chrome"
# 通过环境变量 COOKIES_FROM_BROWSER 配置，Docker 部署时留空
COOKIES_FROM_BROWSER = os.environ.get("COOKIES_FROM_BROWSER", "firefox")
print(f"[config] COOKIE_FILE={COOKIE_FILE} COOKIES_FROM_BROWSER={COOKIES_FROM_BROWSER}")
# ai-zh: B站AI字幕; zh-Hans/zh.*: 其他中文; en.*: 英文兜底
DEFAULT_LANGS = "ai-zh,zh-Hans,zh.*,en.*"
PORT = 8822

app = FastAPI(title="Universal Subtitle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 字幕清洗 -- 去除 SRT 序号、时间戳、空行，去重连续重复行，返回纯文本
# ---------------------------------------------------------------------------
SRT_INDEX_RE = re.compile(r'^\d+\s*$')
SRT_TIMESTAMP_RE = re.compile(
    r'^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}'
)
HTML_TAG_RE = re.compile(r'<[^>]+>')


def clean_srt(srt_text: str) -> str:
    """去除 SRT 格式噪音，返回适合传给智能体的纯文本。"""
    lines = srt_text.splitlines()
    cleaned = []
    prev = None
    for line in lines:
        line = line.strip()
        # 跳过序号行、时间戳行、空行
        if not line or SRT_INDEX_RE.match(line) or SRT_TIMESTAMP_RE.match(line):
            continue
        # 去除内联 HTML 标签（如 <i>, <font> 等）
        line = HTML_TAG_RE.sub('', line).strip()
        if not line:
            continue
        # 去除与上一行完全相同的重复行（自动字幕常见）
        if line == prev:
            continue
        cleaned.append(line)
        prev = line
    return ' '.join(cleaned)


DEBUG = os.environ.get("DEBUG", "0") == "1"


def _base_ydl_opts(temp_dir: str) -> dict:
    opts = {
        'outtmpl': f'{temp_dir}/%(id)s.%(ext)s',
        'quiet': not DEBUG,
        'no_warnings': not DEBUG,
        'verbose': DEBUG,
    }
    # 优先使用 cookies 文件（Docker 生产环境）
    if os.path.exists(COOKIE_FILE):
        opts['cookiefile'] = COOKIE_FILE
        print(f"[cookies] using file: {COOKIE_FILE}")
    # 其次使用浏览器 cookies（本地调试）
    elif COOKIES_FROM_BROWSER:
        opts['cookiesfrombrowser'] = (COOKIES_FROM_BROWSER,)
        print(f"[cookies] using browser: {COOKIES_FROM_BROWSER}")
    else:
        print("[cookies] no auth configured")
    return opts


# ---------------------------------------------------------------------------
# 查询可用字幕轨道（不下载视频）
# ---------------------------------------------------------------------------
def get_subtitle_info(url: str) -> dict:
    temp_dir = tempfile.mkdtemp()
    try:
        opts = _base_ydl_opts(temp_dir)
        opts.update({
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
        })
        print(f"[info] extracting: {url}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            import json
            # print(json.dumps(info, indent=2, default=str, ensure_ascii=False))
            subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})
            print(f"[info] subtitles={list(subs.keys())} auto={list(auto_subs.keys())}")
            print(json.dumps(auto_subs, indent=2, default=str, ensure_ascii=False))
            return {
                'id': info.get('id'),
                'title': info.get('title'),
                'duration': info.get('duration'),
                'subtitles': list(subs.keys()),
                'automatic_captions': list(auto_subs.keys()),
            }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 下载并返回字幕
# ---------------------------------------------------------------------------
def download_subtitles(url: str, lang_pref: str = DEFAULT_LANGS) -> dict | None:
    temp_dir = tempfile.mkdtemp()
    try:
        opts = _base_ydl_opts(temp_dir)
        langs = [l.strip() for l in lang_pref.split(',') if l.strip()]
        opts.update({
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'subtitleslangs': langs,
        })
        print(f"[fetch] url={url} langs={langs}")

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id')
            title = info.get('title')

        sub_files = sorted(glob.glob(os.path.join(temp_dir, f"{video_id}*.srt")))
        print(f"[fetch] found srt files: {sub_files}")
        if not sub_files:
            return None

        target_file = sub_files[0]
        # 从文件名提取实际语言代码，例如 vid.zh-Hans.srt -> zh-Hans
        basename = os.path.basename(target_file)
        lang_detected = basename.replace(f"{video_id}.", '').replace('.srt', '')

        with open(target_file, 'r', encoding='utf-8') as f:
            raw_srt = f.read()

        return {
            'title': title,
            'id': video_id,
            'source': url,
            'lang': lang_detected,
            'subtitles_raw': raw_srt,
            'subtitles_text': clean_srt(raw_srt),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.get("/info", summary="查询视频可用字幕轨道")
async def info(url: str):
    try:
        return get_subtitle_info(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch", summary="下载字幕（同时返回原始 SRT 与清洗后纯文本）")
async def fetch(url: str, lang: str = DEFAULT_LANGS):
    try:
        data = download_subtitles(url, lang)
        if data:
            return data
        raise HTTPException(status_code=404, detail="No subtitles found for this video.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
