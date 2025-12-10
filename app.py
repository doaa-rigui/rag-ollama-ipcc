from fastapi import FastAPI
from pydantic import BaseModel
import requests

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI()

print("Loading vector database...")

embedding_fn = OllamaEmbeddings(model="nomic-embed-text:latest")
vectordb = Chroma(
    persist_directory="vectordb",
    embedding_function=embedding_fn
)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

print("✓ Vector DB Loaded")

def call_ollama(prompt: str, model: str = "gemma2:2b"):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False  
            },
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except Exception as e:
        return f"[Ollama Error] {str(e)}"


class QueryIn(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/ask")
def ask(q: QueryIn):
    print(f"Question: {q.question}")

    docs = retriever.invoke(q.question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {q.question}

Answer:
"""

    answer = call_ollama(prompt)

    sources = [
        {
            "content": doc.page_content[:200] + "...",
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources}
