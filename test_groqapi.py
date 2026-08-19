import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("groq_api_key"))

# Use a valid Groq model name
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence.",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)