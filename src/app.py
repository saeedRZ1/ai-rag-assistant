"""
Streamlit Chat UI for the RAG Assistant.

Run with:
    streamlit run src/app.py
"""

import os
import streamlit as st
from rag_pipeline import RAGAssistant


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 RAG Assistant")
st.caption("Ask questions about your documents.")

# ------------------------------------------------------------------
# Sidebar — configuration
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    docs_path = st.text_input("Documents path", value="./data/sample_docs")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

    if st.button("🔄 Clear conversation"):
        st.session_state.messages = []
        if "assistant" in st.session_state:
            st.session_state.assistant.clear_memory()
        st.rerun()

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Documents → chunks → embeddings")
    st.markdown("2. Query → vector search → top-k chunks")
    st.markdown("3. Chunks + query → LLM → answer")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state:
    with st.spinner("Loading RAG pipeline..."):
        try:
            st.session_state.assistant = RAGAssistant(
                docs_path=docs_path,
                model=model,
                top_k=top_k,
            )
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()

# ------------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.text(src)

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = st.session_state.assistant.ask(prompt)
                answer = result["answer"]
                sources = result["sources"]
                st.markdown(answer)
                if sources:
                    with st.expander(f"📄 Sources ({result['num_chunks_retrieved']} chunks retrieved)"):
                        for src in sources:
                            st.text(src)
            except Exception as e:
                answer = f"Error: {e}"
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
