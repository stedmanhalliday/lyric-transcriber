import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import mlx_whisper
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

MODEL = "mlx-community/whisper-large-v3-turbo"
ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
HERE = Path(__file__).parent

app = FastAPI()
job_lock = threading.Lock()  # one job at a time


class StageError(Exception):
    def __init__(self, stage, detail):
        self.stage = stage
        self.detail = detail


class Job(BaseModel):
    url: str


def valid_youtube_url(url):
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and (parsed.hostname or "").lower() in ALLOWED_HOSTS


def last_line(text):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else "(no stderr)"


def run(stage, cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise StageError(stage, last_line(result.stderr))


def pipeline(url, workdir):
    timings = {}

    t = time.time()
    run("download", ["yt-dlp", "-x", "--audio-format", "mp3", "--no-playlist",
                     "-o", str(workdir / "audio.%(ext)s"), "--", url])
    mp3 = workdir / "audio.mp3"
    if not mp3.exists():
        raise StageError("download", "yt-dlp finished but no mp3 was written")
    timings["download"] = round(time.time() - t, 1)

    t = time.time()
    run("isolate", [sys.executable, "-m", "demucs", "--two-stems=vocals",
                    "-o", str(workdir / "sep"), str(mp3)])
    vocals = next((workdir / "sep").glob("*/audio/vocals.wav"), None)
    if vocals is None:
        raise StageError("isolate", "demucs finished but no vocals.wav was written")
    timings["isolate"] = round(time.time() - t, 1)

    t = time.time()
    try:
        result = mlx_whisper.transcribe(str(vocals), path_or_hf_repo=MODEL)
    except Exception as e:
        raise StageError("transcribe", last_line(str(e)) or type(e).__name__)
    lyrics = "\n".join(s["text"].strip() for s in result["segments"] if s["text"].strip())
    timings["transcribe"] = round(time.time() - t, 1)

    print(f"timings: {timings}", flush=True)
    return lyrics, timings


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.post("/transcribe")
def transcribe(job: Job):
    if not valid_youtube_url(job.url):
        return JSONResponse({"error": "Not a youtube.com or youtu.be URL."}, status_code=400)
    with job_lock:
        workdir = Path(tempfile.mkdtemp(prefix="lyrics-"))
        try:
            lyrics, timings = pipeline(job.url.strip(), workdir)
            return {"lyrics": lyrics, "timings": timings}
        except StageError as e:
            return JSONResponse({"error": f"{e.stage} failed: {e.detail}", "stage": e.stage}, status_code=500)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
