import os
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

import whisper
import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

# ─── Config ───────────────────────────────────────────────────────────────────

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

WhisperModel = Literal["tiny", "base", "small", "medium", "large"]

# Cache de modelos para no recargar en cada petición
_models: dict[str, whisper.Whisper] = {}

def get_model(name: WhisperModel) -> whisper.Whisper:
    if name not in _models:
        print(f"[whisper] Cargando modelo '{name}'...")
        _models[name] = whisper.load_model(name)
    return _models[name]


# ─── Lifespan (precarga modelo base al iniciar) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model("base")   # precarga en startup
    yield
    # cleanup: borrar archivos temporales al apagar
    for f in DOWNLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YouTube Transcriber API",
    description="Descarga audio de YouTube y lo transcribe con Whisper.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    url: HttpUrl
    model: WhisperModel = "base"
    language: str | None = None   # ej: "es", "en". None = autodetect

class TranscribeResponse(BaseModel):
    job_id: str
    title: str
    duration_seconds: float | None
    language: str
    text: str
    segments_count: int


class DownloadRequest(BaseModel):
    url: HttpUrl


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _download_audio(url: str, job_id: str) -> tuple[Path, dict]:
    """Descarga el audio de YouTube y retorna la ruta al MP3 + metadata."""
    out_template = str(DOWNLOAD_DIR / f"{job_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    mp3_path = DOWNLOAD_DIR / f"{job_id}.mp3"
    if not mp3_path.exists():
        raise FileNotFoundError("El archivo MP3 no fue creado correctamente.")

    return mp3_path, info


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "YouTube Transcriber API — visita /docs"}


@app.get("/models", tags=["Info"])
def list_models():
    """Retorna los modelos disponibles y cuáles ya están cargados en memoria."""
    available = ["tiny", "base", "small", "medium", "large"]
    return {
        "available": available,
        "loaded": list(_models.keys()),
    }


@app.post("/transcribe", response_model=TranscribeResponse, tags=["Transcribe"])
def transcribe_video(req: TranscribeRequest):
    """
    Descarga el audio de una URL de YouTube y lo transcribe con Whisper.

    - **url**: URL del video de YouTube
    - **model**: Modelo de Whisper a usar (default: `base`)
    - **language**: Código de idioma opcional (ej: `es`, `en`). Si no se pasa, se autodetecta.
    """
    job_id = uuid.uuid4().hex[:8]
    mp3_path = None

    try:
        # 1. Descargar audio
        mp3_path, info = _download_audio(str(req.url), job_id)

        # 2. Transcribir
        model = get_model(req.model)
        transcribe_opts = {}
        if req.language:
            transcribe_opts["language"] = req.language

        result = model.transcribe(str(mp3_path), **transcribe_opts)

        return TranscribeResponse(
            job_id=job_id,
            title=info.get("title", "Sin título"),
            duration_seconds=info.get("duration"),
            language=result.get("language", "unknown"),
            text=result["text"].strip(),
            segments_count=len(result.get("segments", [])),
        )

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=422, detail=f"Error al descargar: {e}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")
    finally:
        # Limpiar MP3 temporal
        if mp3_path and mp3_path.exists():
            mp3_path.unlink(missing_ok=True)


@app.post("/download-mp3", tags=["Download"])
def download_mp3(req: DownloadRequest):
    """
    Descarga el audio de YouTube como MP3 y lo retorna como archivo.
    """
    job_id = uuid.uuid4().hex[:8]

    try:
        mp3_path, info = _download_audio(str(req.url), job_id)
        filename = f"{info.get('title', job_id)}.mp3"

        return FileResponse(
            path=str(mp3_path),
            media_type="audio/mpeg",
            filename=filename,
            background=BackgroundTasks(),  # el archivo se borra después de enviarlo
        )

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=422, detail=f"Error al descargar: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")


@app.get("/info", tags=["Info"])
def video_info(url: str):
    """
    Retorna metadata de un video de YouTube sin descargarlo.

    - **url**: URL del video (query param)
    """
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "duration_seconds": info.get("duration"),
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "thumbnail": info.get("thumbnail"),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))










if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8089, reload=True)