FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn yt-dlp
COPY app.py .
EXPOSE 8822
# 默认关闭 DEBUG，通过 docker-compose environment 或 -e 覆盖
ENV DEBUG=0
ENV COOKIES_FROM_BROWSER=
CMD ["python", "app.py"]