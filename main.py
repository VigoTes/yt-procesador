import os
import uuid
import asyncio
import subprocess
import httpx
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, HttpUrl

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("API_KEY")

import shutil
FFMPEG_BIN = shutil.which("ffmpeg") or os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe") or os.getenv("FFPROBE_PATH", "ffprobe")
if not shutil.which(FFMPEG_BIN):
    raise RuntimeError(
        f"ffmpeg no encontrado. Instálalo o setea la variable de entorno FFMPEG_PATH."
    )

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
    get_model("medium")
    yield
    for f in DOWNLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Video Transcriber API",
    description="Recibe un archivo de video, extrae el audio y lo transcribe con Whisper.",
    version="2.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class JobAccepted(BaseModel):
    job_id: str
    status: str = "accepted"
    message: str


class WebhookPayload(BaseModel):
    job_id: str
    codVideo: str
    status: Literal["success", "error"]
    filename: str
    model: WhisperModel
    duration_seconds: float | None = None
    language: str | None = None
    text: str | None = None
    segments_count: int | None = None
    error: str | None = None


class ScheduledGetRequest(BaseModel):
    url: str
    delay_seconds: int

class ScheduledGetAccepted(BaseModel):
    job_id: str
    status: str = "scheduled"
    url: str
    delay_seconds: int
    message: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_audio(video_path: Path, job_id: str) -> tuple[Path, float | None]:
    """Usa FFmpeg para extraer el audio del video y exportarlo como MP3."""
    mp3_path = DOWNLOAD_DIR / f"{job_id}.mp3"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "192k",
        str(mp3_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg falló: {result.stderr}")

    probe = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = None

    return mp3_path, duration


async def _save_upload(upload: UploadFile, job_id: str) -> Path:
    """Guarda el archivo subido en disco y devuelve su path."""
    suffix = Path(upload.filename).suffix or ".mp4"
    video_path = DOWNLOAD_DIR / f"{job_id}_input{suffix}"
    content = await upload.read()
    video_path.write_bytes(content)
    return video_path


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


async def _transcribe_task(
    job_id: str,
    codVideo: str,
    video_path: Path,
    original_filename: str,
    model_name: WhisperModel,
    language: str | None,
    webhook_url: str,
):
    """Tarea en background: extrae audio, transcribe y notifica al webhook."""
    mp3_path = None
    try:
        loop = asyncio.get_event_loop()

        mp3_path, duration = await loop.run_in_executor(
            None, _extract_audio, video_path, job_id
        )

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
            codVideo=codVideo,
            status="success",
            filename=original_filename,
            model=model_name,
            duration_seconds=duration,
            language=result.get("language", "unknown"),
            text=result["text"].strip(),
            segments_count=len(result.get("segments", [])),
        )

    except Exception as exc:
        print(f"[job {job_id}] Error: {exc}")
        payload = WebhookPayload(
            job_id=job_id,
            codVideo=codVideo,
            status="error",
            filename=original_filename,
            model=model_name,
            error=str(exc),
        )

    finally:
        if mp3_path and mp3_path.exists():
            mp3_path.unlink(missing_ok=True)
        if video_path.exists():
            video_path.unlink(missing_ok=True)

    await _notify_webhook(webhook_url, payload)


async def _delayed_get_task(job_id: str, url: str, delay_seconds: int):
    """Espera `delay_seconds` y luego ejecuta un GET a `url`."""
    print(f"[schedule-get] Job {job_id} — esperando {delay_seconds}s para llamar a: {url}")
    await asyncio.sleep(delay_seconds)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            print(f"[schedule-get] Job {job_id} — GET {url} → HTTP {resp.status_code}")
    except Exception as exc:
        print(f"[schedule-get] Job {job_id} — Error al llamar a {url}: {exc}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Video Transcriber API — visita /docs"}


@app.get("/models", tags=["Info"], dependencies=[Depends(verify_api_key)])
def list_models():
    available = ["tiny", "base", "small", "medium", "large"]
    return {"available": available, "loaded": list(_models.keys())}


@app.post(
    "/schedule-get",
    response_model=ScheduledGetAccepted,
    status_code=202,
    tags=["Scheduler"],
    dependencies=[Depends(verify_api_key)],
)
async def schedule_get(
    request: ScheduledGetRequest,
    background_tasks: BackgroundTasks,
):
    """
    Programa un GET diferido a una URL.

    Retorna **inmediatamente** con un `job_id` y encola la llamada en background.
    Pasado `delay_seconds`, se ejecuta un **GET** a `url`.

    Body JSON:
    - **url**: URL destino a llamar
    - **delay_seconds**: segundos a esperar antes de ejecutar el GET (mínimo 0)
    """
    if request.delay_seconds < 0:
        raise HTTPException(status_code=422, detail="delay_seconds no puede ser negativo.")

    job_id = uuid.uuid4().hex[:8]

    background_tasks.add_task(
        _delayed_get_task,
        job_id=job_id,
        url=request.url,
        delay_seconds=request.delay_seconds,
    )

    return ScheduledGetAccepted(
        job_id=job_id,
        status="scheduled",
        url=request.url,
        delay_seconds=request.delay_seconds,
        message=f"GET a '{request.url}' se ejecutará en {request.delay_seconds} segundos.",
    )


@app.post(
    "/transcribe",
    response_model=JobAccepted,
    status_code=202,
    tags=["Transcribe"],
    dependencies=[Depends(verify_api_key)],
)
async def transcribe_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Archivo de video (mp4, mkv, avi, mov, etc.)"),
    codVideo: str = Form(..., description="Código identificador del video"),
    model: WhisperModel = Form("base"),
    language: str | None = Form(None, description="Código de idioma opcional (es, en…). None = autodetect"),
    webhook_url: str = Form(..., description="URL que recibirá el resultado cuando la tarea termine"),
):
    """
    Encola la transcripción del video en background y retorna inmediatamente.

    Cuando el trabajo finaliza (o falla), se hace un **POST** a `webhook_url` con
    el resultado completo (schema `WebhookPayload`).
    """
    job_id = uuid.uuid4().hex[:8]
    video_path = await _save_upload(file, job_id)

    background_tasks.add_task(
        _transcribe_task,
        job_id=job_id,
        codVideo=codVideo,
        video_path=video_path,
        original_filename=file.filename or "video",
        model_name=model,
        language=language,
        webhook_url=webhook_url,
    )

    return JobAccepted(
        job_id=job_id,
        status="accepted",
        message=f"Job encolado. El resultado se enviará a {webhook_url}",
    )


@app.post("/extract-mp3", tags=["Extract"], dependencies=[Depends(verify_api_key)])
async def extract_mp3(
    file: UploadFile = File(..., description="Archivo de video del que extraer el audio"),
):
    """Extrae el audio de un video subido y lo retorna como MP3."""
    job_id = uuid.uuid4().hex[:8]
    video_path = await _save_upload(file, job_id)

    try:
        loop = asyncio.get_event_loop()
        mp3_path, _ = await loop.run_in_executor(None, _extract_audio, video_path, job_id)
        stem = Path(file.filename).stem if file.filename else job_id

        video_path.unlink(missing_ok=True)

        bg = BackgroundTasks()
        bg.add_task(lambda: mp3_path.unlink(missing_ok=True))

        return FileResponse(
            path=str(mp3_path),
            media_type="audio/mpeg",
            filename=f"{stem}.mp3",
            background=bg,
        )
    except RuntimeError as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8089, reload=True)