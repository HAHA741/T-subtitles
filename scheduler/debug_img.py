"""
debug_img.py — 单独调试图像生成接口

用法（在 scheduler/ 目录，激活 venv 后）：
    python debug_img.py
    python debug_img.py "a cat sitting on a cloud, cinematic lighting"
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env（从项目根目录）
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
IMAGE_OPENAI_MODEL = os.environ.get("IMAGE_OPENAI_MODEL", "doubao-seedream-5-0-260128")

TEST_PROMPT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "A serene minimalist landscape, flat design, pastel colors, vector art style"
)


def debug_generate_img(prompt: str) -> None:
    print(f"\n{'='*60}")
    print(f"[debug] OPENAI_BASE_URL : {OPENAI_BASE_URL}")
    print(f"[debug] IMAGE_OPENAI_MODEL: {IMAGE_OPENAI_MODEL}")
    print(f"[debug] OPENAI_API_KEY  : {'***' + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) >= 4 else '(未设置)'}")
    print(f"[debug] prompt          : {prompt[:120]}")
    print(f"{'='*60}\n")

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY 未设置，退出。")
        return

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    print("[debug] 发送请求...")
    try:
        response = client.images.generate(
            model=IMAGE_OPENAI_MODEL,
            prompt=prompt,
            size="1024x1024",
            quality="high",
            # response_format="url",
            output_format="jpeg",
            output_compression=80,
            # n=1,
        )
        print("\n[debug] 原始响应:")
        # 尝试打印可序列化的响应结构
        try:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        except Exception:
            print(repr(response))

        data = response.data
        if not data:
            print("\n[ERROR] response.data 为空！")
            return

        item = data[0]
        print(f"\n[debug] data[0] 类型: {type(item)}")
        print(f"[debug] data[0].url      : {getattr(item, 'url', '(无 url 字段)')}")
        b64 = getattr(item, 'b64_json', None)
        print(f"[debug] data[0].b64_json : {'(有值, len=' + str(len(b64)) + ')' if b64 else '(空)'}")
        print(f"[debug] data[0].revised_prompt: {getattr(item, 'revised_prompt', '(无)')}")

        # b64_json 时解码保存为 PNG 供本地查看
        if b64:
            import base64 as _b64
            out_path = Path(__file__).parent.parent / "output" / "debug_img_output.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(_b64.b64decode(b64))
            print(f"[debug] b64_json 已保存到: {out_path}")

    except Exception as exc:
        print(f"\n[ERROR] 请求失败: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_generate_img(TEST_PROMPT)
