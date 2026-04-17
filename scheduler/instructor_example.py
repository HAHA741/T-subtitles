"""
instructor_example.py — Instructor 结构化输出示例

演示如何在现有 jobs.py 的基础上引入 Instructor，
让 AI 直接返回 Pydantic 对象而不是原始字符串。

用法：
    python instructor_example.py
"""
from __future__ import annotations

import os
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor

# ---------------------------------------------------------------------------
# 客户端初始化 — 唯一变化是用 instructor.from_openai() 包装
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gemini-3.1-pro-preview")


def get_instructor_client() -> instructor.Instructor:
    raw = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    # Mode.MD_JSON: 用 prompt 引导模型输出 Markdown 代码块内的 JSON，
    # 不依赖 tool_call / response_format，对中转端口兼容性最好。
    # 如果中转端点支持 tool_call，可改为默认 Mode（去掉 mode 参数）效果更稳。
    return instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)


# ---------------------------------------------------------------------------
# 示例1：文章拆分（替代现有 _split_article 的字符串 find 方案）
# ---------------------------------------------------------------------------
class ArticleOutput(BaseModel):
    article: str = Field(description="公众号正文全文，包含结尾的「感谢阅读，下期见。」")
    footer: str = Field(description="正文之后的固定引导语，如点赞/在看/星标等文字，可为空")


def generate_article_structured(client: instructor.Instructor, compressed: str) -> ArticleOutput:
    from prompts import article_generation_prompt
    system_prompt = article_generation_prompt.format(article_content=compressed)
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        response_model=ArticleOutput,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请将压缩后的信息单元改编为公众号文章。"},
        ],
        temperature=1,
        max_retries=2,  # 格式不对时自动重试
    )


# ---------------------------------------------------------------------------
# 示例2：标题生成（直接返回候选列表，不需要手动解析文本）
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


def generate_titles_structured(client: instructor.Instructor, article: str) -> TitleOutput:
    from prompts import title_generation_prompt
    system_prompt = title_generation_prompt.format(article_content=article)
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
# 示例3：配图方案（返回结构化配图列表）
# ---------------------------------------------------------------------------
class IllustrationItem(BaseModel):
    index: int = Field(description="图片序号，从1开始")
    position: str = Field(description="插入位置描述，如「第一段之后」或「封面」")
    visual_concept: str = Field(description="视觉意象，中文描述")
    prompt_en: str = Field(description="英文图片生成提示词")
    aspect_ratio: str = Field(description="画幅比例，如 2.35:1 或 16:9")


class IllustrationOutput(BaseModel):
    article_type: str = Field(description="文章类型")
    style: str = Field(description="全局视觉风格名称")
    color_scheme: str = Field(description="色彩方案描述")
    illustrations: list[IllustrationItem] = Field(description="配图方案列表")


# ---------------------------------------------------------------------------
# 快速验证兼容性（运行此文件时执行）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("[error] 请先设置环境变量 OPENAI_API_KEY")
        raise SystemExit(1)

    client = get_instructor_client()
    print(f"[test] 连接 {OPENAI_BASE_URL}，模型 {OPENAI_MODEL}")

    # 最小验证：让模型返回一个简单结构
    class Ping(BaseModel):
        message: str
        ok: bool

    result = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_model=Ping,
        messages=[{"role": "user", "content": "回复 message='pong', ok=true"}],
        max_retries=2,
    )
    print(f"[test] 结构化输出验证: message={result.message!r} ok={result.ok}")
    print("[test] Instructor 兼容性验证通过 ✅")
