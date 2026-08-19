# from dotenv import load_dotenv
# import os

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_cohere import CohereEmbeddings
# from langchain_chroma import Chroma


# # --------------------------------
# # 1. Load environment variables
# # --------------------------------

# load_dotenv()

# cohere_api_key = os.getenv("cohere_api_key")

# print("Cohere API key loaded:", bool(cohere_api_key))


# # --------------------------------
# # 2. Load PDF
# # --------------------------------

# pdf_path = r"C:\Users\Priya\OneDrive\Desktop\speech_to_audio_project\data\documents\RAG_basics.pdf"

# loader = PyPDFLoader(pdf_path)

# documents = loader.load()

# print("Number of pages:", len(documents))


# # --------------------------------
# # 3. Split document into chunks
# # --------------------------------

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100
# )

# chunks = splitter.split_documents(documents)

# print("Number of chunks:", len(chunks))


# # --------------------------------
# # 4. Create embeddings
# # --------------------------------

# embedding = CohereEmbeddings(
#     model="embed-english-v3.0"
# )

# print("Embedding model created")


# # --------------------------------
# # 5. Store chunks in ChromaDB
# # --------------------------------

# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding,
#     persist_directory="chroma_db"
# )

# print("Documents stored in ChromaDB")


# # --------------------------------
# # 6. Create retriever
# # --------------------------------

# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 3}
# )

# print("Retriever created")


# # --------------------------------
# # 7. Test retrieval
# # --------------------------------

# question = "What is RAG?"

# results = retriever.invoke(question)

# print("\nRetrieved results:")

# for i, document in enumerate(results):

#     print(f"\n--- Result {i + 1} ---")

#     print(document.page_content)

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_chroma import Chroma


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()

cohere_api_key = os.getenv("cohere_api_key")

print("Cohere API key loaded:", bool(cohere_api_key))


# --------------------------------
# 2. Load PDF
# --------------------------------

pdf_path = r"C:\Users\Priya\OneDrive\Desktop\speech_to_audio_project\data\documents\RAG_basics.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))


# --------------------------------
# 3. Split document into chunks
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------
# 4. Create Cohere embeddings
# --------------------------------

embedding = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)

print("Embedding model created")


# --------------------------------
# 5. Store chunks in ChromaDB
# --------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

print("Documents stored in ChromaDB")


# --------------------------------
# 6. Create retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created")


# --------------------------------
# 7. Create Cohere LLM
# --------------------------------

llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
    cohere_api_key=cohere_api_key
)

print("Cohere LLM created")


# --------------------------------
# 8. Ask a question
# --------------------------------

question = "What is RAG?"

print("\nQuestion:")
print(question)


# --------------------------------
# 9. Retrieve relevant documents
# --------------------------------

results = retriever.invoke(question)

print("\nRetrieved documents:")

for i, document in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)


# --------------------------------
# 10. Combine retrieved content
# --------------------------------

context = "\n\n".join(
    document.page_content
    for document in results
)


# --------------------------------
# 11. Create prompt
# --------------------------------

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the information
provided in the context below.

If the answer is not available in the context,
say that you don't know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""


# --------------------------------
# 12. Send prompt to Cohere LLM
# --------------------------------

response = llm.invoke(prompt)


# --------------------------------
# 13. Print final answer
# --------------------------------

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(response.content)