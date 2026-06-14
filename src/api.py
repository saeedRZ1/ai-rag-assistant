"""
FastAPI REST API for the RAG Assistant.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGAssistant


# ------------------------------------------------------------------
# Global assistant instance
# ------------------------------------------------------------------

assistant: Optional[RAGAssistant] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant
    docs_path = os.getenv("DOCS_PATH", "./data/sample_docs")
    persist_dir = os.getenv("PERSIST_DIR", "./chroma_db")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    assistant = RAGAssistant(docs_path=docs_path, persist_dir=persist_dir, model=model)
    yield
    # cleanup (nothing needed for ChromaDB)


app = FastAPI(
    title="RAG Assistant API",
    description="Retrieval-Augmented Generation over your own documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_chunks_retrieved: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("OPENAI_MODEL", "gpt-4o")}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = assistant.ask(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
def clear_memory():
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized.")
    assistant.clear_memory()
    return {"status": "memory cleared"}


@app.get("/")
def root():
    return {
        "message": "RAG Assistant API is running.",
        "docs": "/docs",
        "endpoints": ["/ask", "/clear-memory", "/health"],
    }
