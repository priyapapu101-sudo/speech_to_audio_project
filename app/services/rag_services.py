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
    raise ValueError("cohere_api_key is not found in the .env file")


# --------------------------------
# 2. Create embedding model
# --------------------------------

embedding = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)


# --------------------------------
# 3. Load existing ChromaDB
# --------------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)


# --------------------------------
# 4. Create retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------
# 5. Create Cohere LLM
# --------------------------------

llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
    cohere_api_key=cohere_api_key
)


# --------------------------------
# 6. RAG function
# --------------------------------

def ask_question(question):

    # Retrieve relevant documents
    results = retriever.invoke(question)

    # Combine document contents
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the information
provided in the context below.

If the answer cannot be found in the context,
say that you don't know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content


# --------------------------------
# 7. Test the function
# --------------------------------

if __name__ == "__main__":

    question = "What is RAG?"

    answer = ask_question(question)

    print("\n==============================")
    print("QUESTION")
    print("==============================")

    print(question)

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(answer)