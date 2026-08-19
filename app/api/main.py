import time
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.services.conversation_service import ask_question
from app.speech.speech_to_text import transcribe_audio

import base64
import os
import uuid
import pyttsx3

def text_to_speech(text):

    audio_file = f"response_{uuid.uuid4().hex}.wav"

    engine = pyttsx3.init()

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    engine.save_to_file(
        text,
        audio_file
    )

    engine.runAndWait()

    return audio_file


app = FastAPI(
    title="AI-Powered Speech-to-Text & Conversational RAG Assistant"
)
class QuestionRequest(BaseModel):

    question: str

@app.get("/")
def home():

    return {
        "message": "Conversational RAG API is running"
    }
@app.post("/ask")

def ask(request: QuestionRequest):

    answer = ask_question(request.question)

    return {
        "question": request.question,
        "answer": answer
    }
@app.post("/voice")
async def voice(file: UploadFile = File(...)):

    print("\n========== VOICE REQUEST ==========")

    total_start = time.time()

    # -----------------------------------------
    # Save audio
    # -----------------------------------------

    start = time.time()

    audio_path = f"temp_{file.filename}"

    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())

    print(
        f"1. Audio saved: {time.time() - start:.2f} seconds"
    )

    # -----------------------------------------
    # Whisper
    # -----------------------------------------

    start = time.time()

    print("2. Starting Whisper...")

    text = transcribe_audio(audio_path)

    print(
        f"3. Whisper finished: {time.time() - start:.2f} seconds"
    )

    print("Transcribed text:", text)

    # -----------------------------------------
    # RAG
    # -----------------------------------------

    start = time.time()

    print("4. Starting RAG...")

    answer = ask_question(text)

    print(
        f"5. RAG finished: {time.time() - start:.2f} seconds"
    )

    print("Answer:", answer)

    # -----------------------------------------
    # TTS
    # -----------------------------------------

    start = time.time()

    print("6. Starting Text-to-Speech...")

    audio_file = text_to_speech(answer)

    print(
        f"7. TTS finished: {time.time() - start:.2f} seconds"
    )

    # -----------------------------------------
    # Read generated audio
    # -----------------------------------------

    start = time.time()

    with open(audio_file, "rb") as audio:
        audio_bytes = audio.read()

    audio_base64 = base64.b64encode(
        audio_bytes
    ).decode("utf-8")

    print(
        f"8. Audio encoding finished: "
        f"{time.time() - start:.2f} seconds"
    )

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------

    try:
        os.remove(audio_path)
        os.remove(audio_file)
    except Exception:
        pass

    print(
        f"========== TOTAL TIME: "
        f"{time.time() - total_start:.2f} seconds ==========\n"
    )

    return {
        "transcribed_text": text,
        "answer": answer,
        "audio": audio_base64
    }