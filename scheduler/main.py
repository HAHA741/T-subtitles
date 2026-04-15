"""
main.py — 定时调度入口

CRON_EXPR 格式（标准 5 段 cron）：
  分 时 日 月 周
  例: "0 * * * *"   每小时整点执行
      "*/30 * * * *" 每 30 分钟执行
      "0 9 * * 1-5"  工作日早 9 点执行

RUN_NOW=1 可让容器启动后立即执行一次，便于调试。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件（scheduler/ 的上一级）
load_dotenv(Path(__file__).parent.parent / ".env")

from apscheduler.schedulers.blocking import BlockingScheduler

from jobs import process_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CRON_EXPR: str = os.environ.get("CRON_EXPR", "0 * * * *")
RUN_NOW: bool = os.environ.get("RUN_NOW", "0") == "1"


def _parse_cron(expr: str) -> dict:
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"无效的 cron 表达式（需要 5 段）: {expr!r}")
    keys = ["minute", "hour", "day", "month", "day_of_week"]
    return dict(zip(keys, parts))


def main() -> None:
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    cron_kwargs = _parse_cron(CRON_EXPR)
    scheduler.add_job(process_videos, "cron", id="process_videos", **cron_kwargs)
    logger.info("调度器启动，cron: %s  解析: %s", CRON_EXPR, cron_kwargs)

    if RUN_NOW:
        logger.info("RUN_NOW=1，立即执行一次任务后退出")
        process_videos()
        return

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("调度器已停止")


if __name__ == "__main__":
    main()
