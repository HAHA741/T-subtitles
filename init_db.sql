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

-- ---------------------------------------------------------------------------
-- 实际业务表
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS video_data (
    id            SERIAL PRIMARY KEY,
    platform      VARCHAR(50),
    origin_id     VARCHAR(200) NOT NULL UNIQUE,  -- 平台原始视频 ID
    title         TEXT,
    description   TEXT,
    ptime         BIGINT,                         -- 发布时间戳
    play_count    BIGINT,
    share_count   BIGINT,
    comment_count BIGINT,
    ai_status     VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- ai_status 取值: pending / processing / done / failed
    ai_result     TEXT,                           -- AI 完整输出（备份）
    ai_article    TEXT,                           -- 文章正文（尾部分割符之前）
    ai_check      TEXT,                           -- 文章检查段（尾部分割符及其后内容）
    ai_compressed TEXT,                           -- AI 压缩后的字幕素材
    meta_data     JSONB,
    createf_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_data_ai_status  ON video_data (ai_status);
CREATE INDEX IF NOT EXISTS idx_video_data_platform   ON video_data (platform);
CREATE INDEX IF NOT EXISTS idx_video_data_createf_at ON video_data (createf_at);

-- 如果表已存在，补充新列（幂等）
ALTER TABLE video_data ADD COLUMN IF NOT EXISTS ai_article    TEXT;
ALTER TABLE video_data ADD COLUMN IF NOT EXISTS ai_check      TEXT;
ALTER TABLE video_data ADD COLUMN IF NOT EXISTS ai_compressed TEXT;
