import whisper

from app.services.conversation_service import ask_question


# --------------------------------
# 1. Load Whisper model
# --------------------------------

print("Loading Whisper model...")

model = whisper.load_model("medium")

print("Whisper model loaded")


# --------------------------------
# 2. Audio file path
# --------------------------------

audio_path = r"C:\Users\Priya\OneDrive\Documents\Audio Records\AudioRec_260817_gr3b4thz.wav"


# --------------------------------
# 3. Speech-to-text function
# --------------------------------

def transcribe_audio(audio_path):

    result = model.transcribe(
        audio_path,
        language="en",
        temperature=0
    )

    text = result["text"].strip()

    return text
   


# --------------------------------
# 4. Test speech-to-text
# --------------------------------

if __name__ == "__main__":

    # --------------------------------
    # 1. Speech-to-text
    # --------------------------------

    text = transcribe_audio(audio_path)

    print("\n==============================")
    print("TRANSCRIBED TEXT")
    print("==============================")

    print(text)


    # --------------------------------
    # 2. Send text to conversational RAG
    # --------------------------------

    answer = ask_question(text)


    # --------------------------------
    # 3. Display answer
    # --------------------------------

    print("\n==============================")
    print("RAG ANSWER")
    print("==============================")

    print(answer)