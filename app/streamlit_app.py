import streamlit as st
import requests
import base64

# =========================================================
# PAGE CONFIGURATION & SETUP
# =========================================================
st.set_page_config(
    page_title="AI Conversational RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000"

st.title("🤖 AI Conversational RAG Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# RENDER CONVERSATION HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("text"):
            st.write(msg["text"])
        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/wav")

# =========================================================
# INPUT METHODS (TEXT & VOICE)
# =========================================================
st.subheader("Ask a Question")

tab1, tab2 = st.tabs(["💬 Text Input", "🎤 Voice Input"])

user_query = None
audio_file_payload = None

# --- TAB 1: TEXT INPUT ---
with tab1:
    text_input = st.chat_input("Type your question here...")
    if text_input:
        user_query = text_input

# --- TAB 2: VOICE INPUT ---
with tab2:
    audio_value = st.audio_input("Record your question")
    if audio_value and st.button("Send Voice Question"):
        audio_file_payload = {
            "file": ("voice_question.wav", audio_value.getvalue(), "audio/wav")
        }

# =========================================================
# PROCESS INPUT & BACKEND REQUESTS
# =========================================================

# Handle Text Request
if user_query:
    # Append user message to state and render
    st.session_state.messages.append({"role": "user", "text": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Process response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API_URL}/ask", json={"question": user_query}, timeout=120)
                if res.status_code == 200:
                    answer = res.json().get("answer", "")
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "text": answer})
                else:
                    st.error(f"Backend Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# Handle Voice Request
elif audio_file_payload:
    with st.chat_message("assistant"):
        with st.spinner("Processing audio..."):
            try:
                res = requests.post(f"{API_URL}/voice", files=audio_file_payload, timeout=600)
                if res.status_code == 200:
                    result = res.json()
                    
                    transcribed = result.get("transcribed_text", "")
                    answer = result.get("answer", "")
                    
                    # Display user's transcribed question first
                    st.session_state.messages.append({"role": "user", "text": f"🎤 *Transcribed:* {transcribed}"})
                    
                    # Decode audio if available
                    audio_bytes = None
                    if "audio" in result:
                        audio_bytes = base64.b64decode(result["audio"])
                    
                    # Record assistant response to state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "text": answer,
                        "audio_bytes": audio_bytes
                    })
                    st.rerun()
                else:
                    st.error(f"Backend Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")