import os
import re
import xml.etree.ElementTree as ET
import requests
import subprocess
import json
import yt_dlp

from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from pinecone.exceptions import NotFoundException

from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.messages import HumanMessage

load_dotenv()

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# 初始化 OpenAI
client = OpenAI()

# 初始化 Chat LLM
llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

# 初始化 Embeddings + VectorStore
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=100,
    show_progress_bar=False
)
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings
)


PODCAST_NS = "https://podcastindex.org/namespace/1.0"
MAX_WHISPER_MINUTES = 15   # Whisper 最多跑幾分鐘（$0.006/min）
MAX_CHARS = 24000          # 直接送 LLM 的最大字元數（約 6K tokens）


# --------------------------------
#  從 RSS 取得 Podcast
# --------------------------------
def get_podcast_info(rss_url: str):
    resp = requests.get(rss_url)
    root = ET.fromstring(resp.content)

    channel = root.find("channel")
    podcast_title = channel.find("title").text if channel is not None else "Unknown Podcast"

    episodes = []

    for item in root.findall(".//item"):
        title = item.find("title")
        pub_date_tag = item.find("pubDate")
        enclosure = item.find("enclosure")

        title = title.text if title is not None else "No Title"

        pub_date = None
        if pub_date_tag is not None:
            try:
                dt = datetime.strptime(pub_date_tag.text, "%a, %d %b %Y %H:%M:%S %Z")
                pub_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pub_date = pub_date_tag.text

        mp3_url = enclosure.get("url") if enclosure is not None else None

        # 偵測 Podcasting 2.0 字幕標籤
        transcript_url = None
        transcript_type = None
        for el in item.findall(f"{{{PODCAST_NS}}}transcript"):
            t = el.get("type", "")
            # 優先選 VTT / SRT，其次 plain text
            if "vtt" in t or "srt" in t or "plain" in t or "text" in t:
                transcript_url = el.get("url")
                transcript_type = t
                break

        episodes.append({
            "title": title,
            "publish_time": pub_date,
            "mp3_url": mp3_url,
            "transcript_url": transcript_url,
            "transcript_type": transcript_type,
        })

    return {
        "podcast_title": podcast_title,
        "episodes": episodes
    }


# --------------------------------
#  下載並解析字幕文字
# --------------------------------
def download_transcript(transcript_url: str, transcript_type: str = "") -> str:
    resp = requests.get(transcript_url)
    resp.raise_for_status()
    text = resp.text

    if "vtt" in transcript_type or transcript_url.endswith(".vtt"):
        return _parse_vtt(text)
    elif "srt" in transcript_type or transcript_url.endswith(".srt"):
        return _parse_srt(text)
    else:
        return text.strip()


def _parse_vtt(content: str) -> str:
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        lines.append(line)
    return " ".join(lines)


def _parse_srt(content: str) -> str:
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        lines.append(line)
    return " ".join(lines)


# --------------------------------
#  下載 MP3 + FFmpeg 切段
# --------------------------------
def download_and_split_mp3(mp3_url: str):
    episode_dir = TMP_DIR / f"episode"
    episode_dir.mkdir(exist_ok=True)

    audio_path = episode_dir / "podcast.mp3"

    print("Downloading:", mp3_url)

    audio = requests.get(mp3_url).content

    with open(audio_path, "wb") as f:
        f.write(audio)

    print("Splitting audio with ffmpeg...")

    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", "60",
        "-c", "copy",
        str(episode_dir / "chunk_%03d.mp3")
    ]

    subprocess.run(cmd)

    files = sorted(episode_dir.glob("chunk_*.mp3"))

    return files


# --------------------------------
#  Whisper STT
# --------------------------------
def whisper_stt_from_episode():
    episode_dir = TMP_DIR / f"episode"

    files = sorted(episode_dir.glob("chunk_*.mp3"))

    transcripts = []

    for file in files:
        print("Transcribing:", file)

        with open(file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        transcripts.append(transcript.text)

    full_text = "\n".join(transcripts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([full_text])

    return docs


# --------------------------------
#  存入 VectorStore Pinecone
# --------------------------------
def save_to_pinecone(docs):
    print("Saving to Pinecone...")
    vectorstore.add_documents(docs)
    print("Saved", len(docs), "documents")


# --------------------------------
#  清除 VectorStore Pinecone Index
# --------------------------------
def clear_pinecone():
    print("Clearing Pinecone VectorStore...")
    try:
        vectorstore.delete(delete_all=True)
    except NotFoundException:
        print("Namespace not found, skip delete.")
    print("VectorStore cleared!")


# ------------------------------------
#  從 VectorStore Pinecone 查詢相似內容
# ------------------------------------
def get_episode_chunks(k: int = 50) -> list[str]:
    docs = vectorstore.similarity_search(query=" ", k=k)
    return [doc.page_content for doc in docs]


# ---------------------------
#  Run Agent
# ---------------------------
def run_agent(k: int = 20):
    docs = get_episode_chunks(k)

    prompt = f"""
你是一個整理 Podcast Episode 的助手，請閱讀下列內容，產生摘要與重點，並回傳 JSON：
{{
    "summary": "...", 
    "key_points": ["...", "..."]
}}

Episode 內容：
""" + "\n".join(docs)

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"summary": text, "key_points": []}

    return {
        "answer": result.get("summary"),
        "key_points": result.get("key_points"),
    }


# --------------------------------
#  簡化摘要（不經 Pinecone，直接送 LLM）
# --------------------------------
def simple_summarize(text: str) -> dict:
    truncated = text[:MAX_CHARS]
    prompt = f"""你是一個整理內容的助手，請閱讀下列內容，用繁體中文產生摘要與重點，並回傳 JSON：
{{
    "summary": "完整摘要文字",
    "key_points": ["重點1", "重點2", "重點3"]
}}

內容：
{truncated}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content

    try:
        match = re.search(r'\{[\s\S]*\}', raw)
        result = json.loads(match.group() if match else raw)
    except (json.JSONDecodeError, AttributeError):
        result = {"summary": raw, "key_points": []}

    return {
        "answer": result.get("summary", ""),
        "key_points": result.get("key_points", []),
    }


# --------------------------------
#  YouTube：取得影片資訊
# --------------------------------
def get_youtube_info(url: str) -> dict:
    opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title", "YouTube 影片"),
        "uploader": info.get("uploader", ""),
        "duration": info.get("duration", 0),
        "thumbnail": info.get("thumbnail", ""),
    }


# --------------------------------
#  YouTube：嘗試取得字幕
# --------------------------------
def get_youtube_subtitles(url: str) -> str | None:
    sub_dir = TMP_DIR / "yt_sub"
    sub_dir.mkdir(exist_ok=True)
    for f in sub_dir.iterdir():
        f.unlink()

    opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "subtitleslangs": ["zh-TW", "zh-Hans", "zh", "en"],
        "skip_download": True,
        "outtmpl": str(sub_dir / "video"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)

    vtt_files = list(sub_dir.glob("*.vtt"))
    if not vtt_files:
        return None
    return _parse_youtube_vtt(vtt_files[0].read_text(encoding="utf-8"))


# --------------------------------
#  YouTube：下載音訊（供 Whisper 用）
# --------------------------------
def download_youtube_audio(url: str):
    episode_dir = TMP_DIR / "episode"
    episode_dir.mkdir(exist_ok=True)
    for f in episode_dir.iterdir():
        f.unlink()

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(episode_dir / "podcast.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)

    # FFmpeg 切段（複用 podcast 流程）
    audio_path = episode_dir / "podcast.mp3"
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "segment", "-segment_time", "60", "-c", "copy",
        str(episode_dir / "chunk_%03d.mp3"), "-y",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------
#  YouTube VTT 解析（處理自動字幕重複行）
# --------------------------------
def _parse_youtube_vtt(content: str) -> str:
    lines = []
    seen = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.startswith("WEBVTT"):
            continue
        if line.startswith(("Kind:", "Language:")):
            continue
        # 移除 inline timing 標記與 HTML tag
        cleaned = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
        cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            lines.append(cleaned)
    return " ".join(lines)