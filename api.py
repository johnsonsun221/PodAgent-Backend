import asyncio
import json as json_lib
import traceback
from typing import List, Optional
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core import (
    get_podcast_info, download_and_split_mp3, download_transcript,
    get_youtube_info, get_youtube_channel_info, get_youtube_subtitles, download_youtube_audio,
    simple_summarize, save_episode_to_pinecone, chat_with_episode, search_episodes,
    client, TMP_DIR, MAX_WHISPER_MINUTES
)

app = FastAPI(title="Podcast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Episode(BaseModel):
    title: str
    publish_time: Optional[str] = None
    mp3_url: Optional[str] = None
    transcript_url: Optional[str] = None
    transcript_type: Optional[str] = None


class Podcast(BaseModel):
    podcast_title: str
    episodes: List[Episode]


@app.get("/podcast", response_model=Podcast)
def api_get_podcast(rss_url: str):
    return get_podcast_info(rss_url)


@app.get("/youtube/info")
async def api_youtube_info(url: str):
    return await asyncio.to_thread(get_youtube_info, unquote(url))

@app.get("/youtube/channel")
async def api_youtube_channel(url: str):
    return await asyncio.to_thread(get_youtube_channel_info, unquote(url))


class ChatRequest(BaseModel):
    question: str
    episode_url: str

@app.post("/chat")
def api_chat(req: ChatRequest):
    return {"answer": chat_with_episode(req.question, req.episode_url)}

@app.get("/search")
def api_search(query: str):
    return search_episodes(query)

@app.get("/analyze/stream")
async def api_analyze_stream(
    mp3_url: Optional[str] = None,
    youtube_url: Optional[str] = None,
    transcript_url: Optional[str] = None,
    transcript_type: Optional[str] = None,
    episode_title: Optional[str] = None,
    podcast_title: Optional[str] = None,
):
    if mp3_url:        mp3_url        = unquote(mp3_url)
    if youtube_url:    youtube_url    = unquote(youtube_url)
    if transcript_url: transcript_url = unquote(transcript_url)

    async def generate():
        def evt(step: str, progress: int, **kw):
            return f"data: {json_lib.dumps({'step': step, 'progress': progress, **kw}, ensure_ascii=False)}\n\n"

        def whisper_transcribe(f):
            with open(f, "rb") as fp:
                return client.audio.transcriptions.create(model="whisper-1", file=fp).text

        async def run_whisper(start: int = 15) -> str:
            # 這個 helper 只做純計算，不 yield，由呼叫端自己 yield evt
            pass  # placeholder — 用不到，改用 inline

        try:
            if youtube_url:
                # ── YouTube：先嘗試字幕 ──
                yield evt("取得 YouTube 字幕", 5)
                sub_text = await asyncio.to_thread(get_youtube_subtitles, youtube_url)

                if sub_text:
                    yield evt("下載字幕", 70)
                    full_text = sub_text
                else:
                    yield evt("下載音訊中", 10)
                    await asyncio.to_thread(download_youtube_audio, youtube_url)
                    chunks = sorted((TMP_DIR / "episode").glob("chunk_*.mp3"))[:MAX_WHISPER_MINUTES]
                    total = len(chunks)
                    parts = []
                    for i, c in enumerate(chunks):
                        yield evt(f"語音轉文字 {i+1}/{total}", 15 + int(i/total*55))
                        parts.append(await asyncio.to_thread(whisper_transcribe, c))
                    full_text = "\n".join(parts)

            elif transcript_url:
                # ── RSS 有字幕 ──
                yield evt("下載字幕", 10)
                full_text = await asyncio.to_thread(
                    download_transcript, transcript_url, transcript_type or ""
                )
                yield evt("下載字幕", 70)

            else:
                # ── 一般 RSS ──
                yield evt("下載音訊中", 5)
                await asyncio.to_thread(download_and_split_mp3, mp3_url)
                chunks = sorted((TMP_DIR / "episode").glob("chunk_*.mp3"))[:MAX_WHISPER_MINUTES]
                total = len(chunks)
                parts = []
                for i, c in enumerate(chunks):
                    yield evt(f"語音轉文字 {i+1}/{total}", 15 + int(i/total*55))
                    parts.append(await asyncio.to_thread(whisper_transcribe, c))
                full_text = "\n".join(parts)

            source_url = youtube_url or mp3_url or ""
            await asyncio.to_thread(save_episode_to_pinecone, full_text, source_url, episode_title or "", podcast_title or "")
            yield evt("AI 生成摘要", 82)
            summary = await asyncio.to_thread(simple_summarize, full_text)
            yield evt("完成", 100, result=summary)

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json_lib.dumps({'step': 'error', 'progress': -1, 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
