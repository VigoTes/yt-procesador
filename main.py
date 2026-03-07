import os
import uuid
import asyncio
import httpx
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

import whisper
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, HttpUrl

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("API_KEY")

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

WhisperModel = Literal["tiny", "base", "small", "medium", "large"]

_models: dict[str, whisper.Whisper] = {}

def get_model(name: WhisperModel) -> whisper.Whisper:
    if name not in _models:
        print(f"[whisper] Cargando modelo '{name}'...")
        _models[name] = whisper.load_model(name)
    return _models[name]


# ─── API Key Auth ─────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Depends(api_key_header)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY no configurada en el servidor.")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida o ausente.")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model("base")
    yield
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
    language: str | None = None
    webhook_url: HttpUrl  # ← nuevo campo obligatorio


class TranscribeResponse(BaseModel):
    job_id: str
    title: str
    duration_seconds: float | None
    language: str
    text: str
    segments_count: int


class JobAccepted(BaseModel):
    job_id: str
    status: str = "accepted"
    message: str


class WebhookPayload(BaseModel):
    job_id: str
    status: Literal["success", "error"]
    # Campos presentes cuando status == "success"
    title: str | None = None
    duration_seconds: float | None = None
    language: str | None = None
    text: str | None = None
    segments_count: int | None = None
    # Campo presente cuando status == "error"
    error: str | None = None


class DownloadRequest(BaseModel):
    url: HttpUrl


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _download_audio(url: str, job_id: str) -> tuple[Path, dict]:
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


async def _notify_webhook(webhook_url: str, payload: WebhookPayload, retries: int = 3):
    """Envía el resultado al webhook con reintentos exponenciales."""
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, retries + 1):
            try:
                resp = await client.post(webhook_url, json=payload.model_dump())
                resp.raise_for_status()
                print(f"[webhook] Notificación enviada a {webhook_url} (job={payload.job_id})")
                return
            except Exception as exc:
                wait = 2 ** attempt
                print(f"[webhook] Intento {attempt}/{retries} fallido: {exc}. Reintentando en {wait}s...")
                if attempt < retries:
                    await asyncio.sleep(wait)

    print(f"[webhook] No se pudo notificar a {webhook_url} tras {retries} intentos.")


async def _transcribe_task(job_id: str, url: str, model_name: WhisperModel, language: str | None, webhook_url: str):
    """Tarea en background: descarga, transcribe y notifica al webhook."""
    mp3_path = None
    try:
        # 1. Descargar audio (bloqueante → corre en threadpool)
        loop = asyncio.get_event_loop()
        mp3_path, info = await loop.run_in_executor(None, _download_audio, url, job_id)

        # 2. Transcribir (bloqueante → corre en threadpool)
        model = get_model(model_name)
        transcribe_opts = {}
        if language:
            transcribe_opts["language"] = language

        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(str(mp3_path), **transcribe_opts)
        )

        payload = WebhookPayload(
            job_id=job_id,
            status="success",
            title=info.get("title", "Sin título"),
            duration_seconds=info.get("duration"),
            language=result.get("language", "unknown"),
            text=result["text"].strip(),
            segments_count=len(result.get("segments", [])),
        )

    except Exception as exc:
        print(f"[job {job_id}] Error durante el procesamiento: {exc}")
        payload = WebhookPayload(
            job_id=job_id,
            status="error",
            error=str(exc),
        )

    finally:
        if mp3_path and mp3_path.exists():
            mp3_path.unlink(missing_ok=True)

    # 3. Notificar resultado
    await _notify_webhook(webhook_url, payload)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "YouTube Transcriber API — visita /docs"}


@app.get("/models", tags=["Info"], dependencies=[Depends(verify_api_key)])
def list_models():
    available = ["tiny", "base", "small", "medium", "large"]
    return {"available": available, "loaded": list(_models.keys())}


@app.post(
    "/transcribe",
    response_model=JobAccepted,
    status_code=202,
    tags=["Transcribe"],
    dependencies=[Depends(verify_api_key)],
)
async def transcribe_video(req: TranscribeRequest, background_tasks: BackgroundTasks):
    """
    Encola la transcripción del video en background y retorna inmediatamente.

    Cuando el trabajo finaliza (o falla), se hace un **POST** a `webhook_url` con
    el resultado completo (schema `WebhookPayload`).

    - **url**: URL del video de YouTube
    - **model**: Modelo Whisper a usar (default: `base`)
    - **language**: Código de idioma opcional (`es`, `en`…). `null` = autodetect
    - **webhook_url**: URL que recibirá el resultado cuando la tarea termine
    """
    job_id = uuid.uuid4().hex[:8]

    background_tasks.add_task(
        _transcribe_task,
        job_id=job_id,
        url=str(req.url),
        model_name=req.model,
        language=req.language,
        webhook_url=str(req.webhook_url),
    )

    return JobAccepted(
        job_id=job_id,
        status="accepted",
        message=f"Job encolado. El resultado se enviará a {req.webhook_url}",
    )


@app.post("/download-mp3", tags=["Download"], dependencies=[Depends(verify_api_key)])
def download_mp3(req: DownloadRequest):
    """Descarga el audio de YouTube como MP3 y lo retorna como archivo."""
    job_id = uuid.uuid4().hex[:8]

    try:
        mp3_path, info = _download_audio(str(req.url), job_id)
        filename = f"{info.get('title', job_id)}.mp3"

        return FileResponse(
            path=str(mp3_path),
            media_type="audio/mpeg",
            filename=filename,
            background=BackgroundTasks(),
        )
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=422, detail=f"Error al descargar: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")


@app.get("/info", tags=["Info"], dependencies=[Depends(verify_api_key)])
def video_info(url: str):
    """Retorna metadata de un video de YouTube sin descargarlo."""
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


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8089, reload=True)