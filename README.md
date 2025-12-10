# RAG System with Ollama and IPCC Climate Reports

A Retrieval-Augmented Generation (RAG) system that allows you to query IPCC AR6 climate change reports using local LLMs via Ollama. This project includes a complete pipeline from document ingestion to a web-based query interface.


## 🚀 Quick Start

### Prerequisites

```bash
# Install Python 3.8+ and pip
# Install Ollama (https://ollama.ai/)
```

### Pull required models
```
ollama pull nomic-embed-text:latest
ollama pull gemma2:2b
```
### Setup Python Environment

1. Install dependencies
`pip install -r requirements.txt`
2. Create data directory and add IPCC PDFs
`mkdir -p data`
3. Add your PDF files to the `data/` directory

### Run the Pipeline

1. Ingest Documents `python ingest.py`. This processes PDFs in data/ and saves chunks to chunks/.
2. Create Vector Database `python embeddings.py`. This loads documents and creates the vector database in vectordb/.
3. Start the Backend `uvicorn app:app --reload --port 8000`
4. Launch Web Interface `streamlit run ui_streamlit.py`

## Configuration

### Models
- **Embedding**: `nomic-embed-text:latest`
- **LLM**: `gemma2:2b` or `llama3.1` (configurable)

### Chunking Parameters
- **Chunk size**: 1000 characters
- **Chunk overlap**: 200 characters

### Retrieval Parameters
- **Search type**: similarity
- **Number of chunks**: 3-4 (configurable)

## 📊 Features

- **Document Processing**: Automatic PDF loading and chunking
- **Vector Search**: Semantic search using ChromaDB
- **Local LLM**: Privacy-preserving local inference with Ollama
- **Web Interface**: User-friendly Streamlit UI
- **REST API**: FastAPI backend for integration
- **Source Citation**: Returns sources with metadata
