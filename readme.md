# 🎙️ YouTube Transcriber API

API REST construida con **FastAPI** y **Whisper** que permite descargar el audio de cualquier video de YouTube y transcribirlo automáticamente. También expone un endpoint para descargar el audio en formato MP3.

---

## 🚀 Características

- 📥 Descarga audio de YouTube en alta calidad vía `yt-dlp`
- 🧠 Transcripción automática con los modelos de OpenAI Whisper
- 🌐 Detección automática de idioma (o especificación manual)
- 📄 Obtención de metadata del video sin descargarlo
- 🎵 Descarga directa del audio como archivo MP3
- ⚡ Caché de modelos Whisper para evitar recargas innecesarias
- 🧹 Limpieza automática de archivos temporales

---

## 🛠️ Requisitos

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) instalado en el sistema

### Instalación de ffmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### Instalación de dependencias Python

```bash
pip install fastapi uvicorn openai-whisper yt-dlp pydantic python-dotenv
```

---

## ▶️ Ejecución

```bash
python main.py
```

El servidor arrancará en `http://0.0.0.0:8080` por defecto.

También puedes ejecutarlo con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

---

## 🔐 Autenticación

Todos los endpoints están protegidos con una API Key. Debes enviarla en el header de cada request:

```
X-API-Key: tu_clave_secreta_aqui
```

La clave se define en un archivo `.env` en la misma carpeta que `main.py`:

```env
API_KEY=tu_clave_secreta_aqui
```

> ⚠️ Nunca subas el archivo `.env` a tu repositorio. Agrégalo al `.gitignore`.

---

## 📁 Estructura del proyecto

```
tu_proyecto/
├── main.py          # Aplicación principal
├── .env             # Variables de entorno (API Key)
├── .gitignore       # Debe incluir .env
└── downloads/       # Carpeta temporal (se crea automáticamente)
```

---

## 📖 Documentación interactiva

Una vez corriendo el servidor, accede a:

- **Swagger UI:** `http://localhost:8080/docs`
- **ReDoc:** `http://localhost:8080/redoc`

---

## 🔌 Endpoints

### `GET /`
Health check. Retorna el estado de la API.

---

### `GET /models`
Lista los modelos Whisper disponibles e indica cuáles están cargados en memoria.

**Respuesta de ejemplo:**
```json
{
  "available": ["tiny", "base", "small", "medium", "large"],
  "loaded": ["base"]
}
```

---

### `GET /info`
Retorna la metadata de un video de YouTube sin descargarlo.

**Query params:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `url` | `string` | URL del video de YouTube |

**Respuesta de ejemplo:**
```json
{
  "title": "Nombre del video",
  "uploader": "Nombre del canal",
  "duration_seconds": 312,
  "view_count": 150000,
  "upload_date": "20240101",
  "thumbnail": "https://..."
}
```

---

### `POST /transcribe`
Descarga el audio del video y lo transcribe con Whisper.

**Body (JSON):**
| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `url` | `string` | ✅ | URL del video de YouTube |
| `model` | `string` | ❌ | Modelo Whisper a usar (default: `base`) |
| `language` | `string` | ❌ | Código de idioma, ej: `es`, `en`. Si no se indica, se autodetecta |

**Ejemplo de request:**
```bash
curl -X POST "http://localhost:8080/transcribe" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: tu_clave_secreta_aqui" \
  -d '{
    "url": "https://www.youtube.com/watch?v=XXXX",
    "model": "small",
    "language": "es"
  }'
```

**Respuesta de ejemplo:**
```json
{
  "job_id": "a3f9c12b",
  "title": "Nombre del video",
  "duration_seconds": 312.0,
  "language": "es",
  "text": "Transcripción completa del video aquí...",
  "segments_count": 42
}
```

---

### `POST /download-mp3`
Descarga el audio del video y lo retorna como archivo MP3.

**Body (JSON):**
| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `url` | `string` | ✅ | URL del video de YouTube |

**Ejemplo de request:**
```bash
curl -X POST "http://localhost:8080/download-mp3" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: tu_clave_secreta_aqui" \
  -d '{"url": "https://www.youtube.com/watch?v=XXXX"}' \
  --output audio.mp3
```

---

## 🧠 Modelos Whisper disponibles

| Modelo | RAM aprox. | Velocidad | Precisión |
|--------|-----------|-----------|-----------|
| `tiny` | ~1 GB | ⚡⚡⚡⚡ | ⭐ |
| `base` | ~1 GB | ⚡⚡⚡ | ⭐⭐ |
| `small` | ~2 GB | ⚡⚡ | ⭐⭐⭐ |
| `medium` | ~5 GB | ⚡ | ⭐⭐⭐⭐ |
| `large` | ~10 GB | 🐢 | ⭐⭐⭐⭐⭐ |

> El modelo `base` se precarga automáticamente al iniciar el servidor.

---

## ⚠️ Consideraciones

- Los archivos MP3 temporales se eliminan automáticamente tras cada transcripción o descarga.
- Al apagar el servidor, se limpian todos los archivos restantes en la carpeta `downloads/`.
- El tiempo de respuesta del endpoint `/transcribe` depende de la duración del video y del modelo seleccionado.
- Se recomienda usar `small` o `medium` para un buen balance entre velocidad y precisión.

---

## 📄 Licencia

MIT