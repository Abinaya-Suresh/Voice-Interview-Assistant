from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import tempfile
import io
import os

app = FastAPI(
    title="STT API",
    description="Speech-to-Text API using sounddevice (no PyAudio)",
    version="1.0"
)

recognizer = sr.Recognizer()


# -------------------- ROOT --------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Speech-to-Text API"}


# -------------------- LIVE RECORD --------------------
@app.get("/listen_long")
def listen_long(
    timeout: int = Query(15, description="Max wait time for speech (seconds)"),
    phrase_time_limit: int = Query(60, description="Max duration for one phrase (seconds)")
):
    """
    Capture speech from microphone using sounddevice and return recognized text.
    Works on host system (Docker mic passthrough only on Linux).
    """
    try:
        #print("🎙️ Listening with sounddevice...")

        # Record audio from mic
        fs = 16000
        audio_data = sd.rec(int(phrase_time_limit * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        #print("🎧 Processing speech...")

        # Convert to AudioData
        audio_bytes = audio_data.tobytes()
        audio = sr.AudioData(audio_bytes, fs, 2)

        # Recognize speech
        text = recognizer.recognize_google(audio)
        print("[INFO] Recognized:", text)
        return {"transcription": text}

    except sr.UnknownValueError:
        return JSONResponse(status_code=200, content={"error": "Speech not recognized. Please try again."})
    except sr.RequestError as e:
        return JSONResponse(status_code=500, content={"error": f"API request failed: {e}"})
    except Exception as e:
        print(f" Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------- FILE UPLOAD --------------------
@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav, .flac, etc.) and get transcription.
    """
    try:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)

        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        os.remove(tmp_path)
        return {"filename": file.filename, "transcription": text}

    except sr.UnknownValueError:
        return {"error": "Speech not recognized"}
    except sr.RequestError as e:
        return {"error": f"API request failed: {e}"}
    except Exception as e:
        print(f" Error in /upload_audio: {e}")
        return {"error": str(e)}
