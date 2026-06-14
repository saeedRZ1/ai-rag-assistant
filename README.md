# AI RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with your own PDF and text documents.

Built with **LangChain · ChromaDB · FastAPI · Streamlit · GPT-4o**.

---

## Features

- 📄 Ingests PDF and TXT documents automatically
- 🔍 Semantic search via ChromaDB vector store (HuggingFace embeddings)
- 💬 Conversational multi-turn chat with memory
- 🌐 REST API (FastAPI) + Chat UI (Streamlit)
- 📌 Source citations with every answer

---

## Project Structure

```
rag-assistant/
├── src/
│   ├── rag_pipeline.py   # Core RAG logic (ingest → embed → retrieve → generate)
│   ├── api.py            # FastAPI REST endpoints
│   └── app.py            # Streamlit chat interface
├── data/
│   └── sample_docs/      # Put your PDFs and TXT files here
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add your documents

Drop PDF or TXT files into `data/sample_docs/`.

---

## Usage

### Option A — Streamlit Chat UI

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser and start chatting.

### Option B — FastAPI REST API

```bash
uvicorn src.api:app --reload
```

API docs at `http://localhost:8000/docs`

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

**Response:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is a technique that...",
  "sources": ["data/sample_docs/sample.txt"],
  "num_chunks_retrieved": 4
}
```

### Option C — Python directly

```python
from src.rag_pipeline import RAGAssistant

assistant = RAGAssistant(docs_path="./data/sample_docs")

result = assistant.ask("What is machine learning?")
print(result["answer"])
print("Sources:", result["sources"])
```

---

## How It Works

```
Documents (PDF/TXT)
      ↓
  Text Chunking (RecursiveCharacterTextSplitter)
      ↓
  Embedding (HuggingFace all-MiniLM-L6-v2)
      ↓
  Vector Store (ChromaDB — local & persistent)
      ↓
User Query → Semantic Search → Top-K Chunks
      ↓
  GPT-4o → Answer + Source Citations
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Orchestration | LangChain |
| Embeddings | HuggingFace sentence-transformers |
| Vector Store | ChromaDB |
| API | FastAPI |
| UI | Streamlit |

---

## License

MIT
