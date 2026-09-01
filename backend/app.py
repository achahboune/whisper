from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import yt_dlp
import tempfile
import os
import subprocess
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper...")
model = whisper.load_model("base")
print("Whisper loaded.")


class TranscriptionRequest(BaseModel):
    url: str


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "transcription-api",
        "model": "base",
        "platforms": [
            "TikTok",
            "Instagram",
            "Facebook",
            "X"
        ]
    }


@app.post("/transcribe-url")
@app.post("/tiktok")
async def transcribe_url(request: TranscriptionRequest):

    url = request.url.strip()

    if not url:
        raise HTTPException(
            status_code=400,
            detail="URL is required"
        )

    temp_dir = tempfile.mkdtemp()

    try:
        audio_template = os.path.join(
            temp_dir,
            "audio.%(ext)s"
        )

        ydl_opts = {
            "outtmpl": audio_template,
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav"
                }
            ]
        }

        print(f"Downloading media: {url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                url,
                download=True
            )

        files = os.listdir(temp_dir)

        print(f"Downloaded files: {files}")

        audio_files = [
            f for f in files
            if f.lower().endswith(
                (
                    ".wav",
                    ".mp3",
                    ".m4a",
                    ".aac",
                    ".opus",
                    ".webm"
                )
            )
        ]

        if not audio_files:
            raise Exception(
                "No audio file was downloaded"
            )

        audio_path = os.path.join(
            temp_dir,
            audio_files[0]
        )

        audio_size = os.path.getsize(audio_path)

        print(f"Audio file: {audio_path}")
        print(f"Audio size: {audio_size} bytes")

        if audio_size < 10000:
            raise Exception(
                "Downloaded audio file is suspiciously small."
            )

        try:
            duration_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ],
                capture_output=True,
                text=True
            )

            if duration_result.returncode != 0:
                raise Exception(
                    duration_result.stderr.strip()
                )

            duration = float(
                duration_result.stdout.strip()
            )

            print(
                f"Audio duration: {duration:.2f} seconds"
            )

        except Exception as e:
            duration = None
            print(
                f"Could not determine audio duration: {repr(e)}"
            )

        print("Starting transcription...")

        result = model.transcribe(
            audio_path,
            fp16=False,
            language=None,
            verbose=False
        )

        text = result.get(
            "text",
            ""
        ).strip()

        detected_language = result.get(
            "language"
        )

        print(
            f"Detected language: {detected_language}"
        )

        if not text:
            raise Exception(
                "No speech detected in the audio. "
                "The video may contain no spoken words."
            )

        print("Transcription completed.")

        return {
            "success": True,
            "text": text,
            "language": detected_language,
            "platform": info.get("extractor_key"),
            "title": info.get("title"),
            "duration": duration
        }

    except HTTPException:
        raise

    except Exception as e:
        print("========== ERROR ==========")
        print("Exception type:", type(e).__name__)
        print("Exception repr:", repr(e))
        print("Exception string:", str(e))
        traceback.print_exc()
        print("===========================")

        error_message = str(e).strip()

        if not error_message:
            error_message = (
                f"{type(e).__name__}: An unknown error occurred"
            )

        raise HTTPException(
            status_code=500,
            detail=error_message
        )

    finally:
        try:
            for filename in os.listdir(temp_dir):
                filepath = os.path.join(
                    temp_dir,
                    filename
                )

                if os.path.isfile(filepath):
                    os.remove(filepath)

            os.rmdir(temp_dir)

        except Exception as cleanup_error:
            print(
                f"Cleanup error: {repr(cleanup_error)}"
            )