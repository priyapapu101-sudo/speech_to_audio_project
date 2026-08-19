import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_chroma import Chroma


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()

cohere_api_key = os.getenv("cohere_api_key")

if not cohere_api_key:
    raise ValueError(
        "cohere_api_key is not found in the .env file"
    )


# --------------------------------
# 2. Create embedding model
# --------------------------------

embedding = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)

print("Embedding model created")


# --------------------------------
# 3. Load existing ChromaDB
# --------------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

print("ChromaDB loaded")


# --------------------------------
# 4. Create retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)

print("Retriever created")


# --------------------------------
# 5. Create Cohere LLM
# --------------------------------

llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
    cohere_api_key=cohere_api_key
)

print("Cohere LLM created")


# --------------------------------
# 6. Conversation memory
# --------------------------------

chat_history = []


# --------------------------------
# 7. Conversational RAG function
# --------------------------------

def ask_question(question):

    # --------------------------------
    # Step 1: Prepare conversation history
    # --------------------------------

    history_text = ""

    for user_message, assistant_message in chat_history:

        history_text += f"""
User: {user_message}

Assistant: {assistant_message}

"""


    # --------------------------------
    # Step 2: Rewrite follow-up question
    # --------------------------------

    if chat_history:

        rewrite_prompt = f"""
You are helping a RAG system understand
a user's follow-up question.

Look at the conversation history and rewrite
the user's latest question into a standalone
question.

Do not answer the question.

Conversation history:
{history_text}

Latest user question:
{question}

Standalone question:
"""

        rewritten_response = llm.invoke(rewrite_prompt)

        search_question = rewritten_response.content.strip()

    else:

        search_question = question


    # --------------------------------
    # Step 3: Retrieve documents
    # --------------------------------

    results = retriever.invoke(search_question)


    # --------------------------------
    # Step 4: Create context
    # --------------------------------

    context = "\n\n".join(
        document.page_content
        for document in results
    )


    # --------------------------------
    # Step 5: Create final prompt
    # --------------------------------

    prompt = f"""
You are a helpful conversational AI assistant.

Answer the user's question using only the
information provided in the context.

You can use the conversation history to
understand what the user is referring to.

If the answer cannot be found in the context,
say that you don't know based on the provided
documents.

Conversation history:
{history_text}

Context:
{context}

Current question:
{question}

Answer:
"""


    # --------------------------------
    # Step 6: Generate answer
    # --------------------------------

    response = llm.invoke(prompt)

    answer = response.content


    # --------------------------------
    # Step 7: Save conversation
    # --------------------------------

    chat_history.append(
        (question, answer)
    )


    # --------------------------------
    # Step 8: Return answer
    # --------------------------------

    return answer


# --------------------------------
# 8. Test conversational memory
# --------------------------------

if __name__ == "__main__":

    # First question
    question1 = "What is RAG?"

    answer1 = ask_question(question1)

    print("\n================================")
    print("QUESTION 1")
    print("================================")

    print(question1)

    print("\n================================")
    print("ANSWER 1")
    print("================================")

    print(answer1)


    # Second question
    question2 = "What are its benefits?"

    answer2 = ask_question(question2)

    print("\n================================")
    print("QUESTION 2")
    print("================================")

    print(question2)

    print("\n================================")
    print("ANSWER 2")
    print("================================")

    print(answer2)