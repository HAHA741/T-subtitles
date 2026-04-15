-- 视频任务模板表，根据实际业务自行调整字段
CREATE TABLE IF NOT EXISTS videos (
    id            SERIAL PRIMARY KEY,
    title         TEXT,
    url           TEXT NOT NULL UNIQUE,
    platform      VARCHAR(50),           -- 平台，如 youtube / bilibili
    author        VARCHAR(200),          -- 作者/UP主
    views         BIGINT,                -- 播放量
    status        VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- status 取值: pending / processing / done / failed
    subtitle_raw        TEXT,             -- 原始 SRT 字幕
    subtitle_text       TEXT,             -- 清洗后纯文本字幕
    subtitle_compressed TEXT,             -- AI 压缩后的结构化信息单元
    article             TEXT,             -- AI 改编后的公众号文章
    error_msg     TEXT,                  -- 失败时的错误信息
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 可选：为常用查询字段建索引
CREATE INDEX IF NOT EXISTS idx_videos_status     ON videos (status);
CREATE INDEX IF NOT EXISTS idx_videos_platform   ON videos (platform);
CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos (created_at);
