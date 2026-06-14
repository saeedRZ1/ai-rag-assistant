"""
RAG Pipeline - Core retrieval-augmented generation logic.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


SYSTEM_PROMPT = """You are a helpful AI assistant. Use the following context to answer 
the user's question accurately. If you don't know the answer based on the context, 
say so clearly — do not make up information.

Context:
{context}

Question: {question}
Answer:"""


class RAGAssistant:
    """
    End-to-end RAG pipeline:
      1. Ingests PDF/TXT documents from a directory
      2. Chunks and embeds them into a ChromaDB vector store
      3. Answers queries via semantic retrieval + GPT-4o generation
    """

    def __init__(
        self,
        docs_path: str,
        persist_dir: str = "./chroma_db",
        model: str = "gpt-4o",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        memory_window: int = 5,
    ):
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        self.top_k = top_k

        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            print(f"Loading existing vector store from '{persist_dir}'...")
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
        else:
            print(f"Building vector store from docs in '{docs_path}'...")
            self.vectorstore = self._build_vectorstore(chunk_size, chunk_overlap)

        self.chain = self._build_chain(model, memory_window)
        print("RAG Assistant ready!")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_documents(self):
        docs = []

        pdf_loader = DirectoryLoader(
            self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        docs.extend(pdf_loader.load())

        txt_loader = DirectoryLoader(
            self.docs_path, glob="**/*.txt", loader_cls=TextLoader
        )
        docs.extend(txt_loader.load())

        if not docs:
            raise ValueError(f"No PDF or TXT documents found in '{self.docs_path}'")

        print(f"Loaded {len(docs)} document pages/sections.")
        return docs

    def _build_vectorstore(self, chunk_size: int, chunk_overlap: int) -> Chroma:
        docs = self._load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        vectorstore.persist()
        print(f"Vector store saved to '{self.persist_dir}'.")
        return vectorstore

    def _build_chain(self, model: str, memory_window: int) -> ConversationalRetrievalChain:
        llm = ChatOpenAI(model=model, temperature=0, streaming=False)

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=memory_window,
            output_key="answer",
        )

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=SYSTEM_PROMPT,
        )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer with source citations.

        Returns:
            {
                "answer": str,
                "sources": list[str],
                "num_chunks_retrieved": int
            }
        """
        result = self.chain({"question": question})

        sources = list(
            {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
        )

        return {
            "answer": result["answer"],
            "sources": sources,
            "num_chunks_retrieved": len(result["source_documents"]),
        }

    def clear_memory(self):
        """Reset conversation history."""
        self.chain.memory.clear()
        print("Conversation memory cleared.")

    def add_documents(self, new_docs_path: str):
        """Ingest additional documents into the existing vector store."""
        loader = DirectoryLoader(new_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        print(f"Added {len(chunks)} new chunks to vector store.")
