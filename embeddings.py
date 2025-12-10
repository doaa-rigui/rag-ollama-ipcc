from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI()

print("Loading vector database...")
embedding_fn = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

vectordb = Chroma(
    persist_directory="vectordb",
    embedding_function=embedding_fn
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

print("Initializing LLM...")
llm = ChatOllama(
    model="llama3.1",  
    temperature=0.0,
    base_url="http://localhost:11434"
)

# Create prompt template
prompt = PromptTemplate.from_template(
    "Use the following context to answer the question. If the answer is not in the context, say 'I don't know.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

print("✓ FastAPI server ready!")


class QueryIn(BaseModel):
    question: str


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "RAG IPCC API - Send POST requests to /ask"
    }


@app.post("/ask")
def ask(q: QueryIn):
    """
    Answer questions using RAG over IPCC documents
    """
    try:
        print(f"\nReceived question: {q.question}")
        result = qa({"query": q.question})
        
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "...",  
                "metadata": doc.metadata
            })
        
        return {
            "answer": result["result"],
            "sources": sources,
            "num_sources": len(sources)
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "num_sources": 0
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)