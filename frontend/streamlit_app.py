"""Minimal Streamlit UI: ingest papers, chat with the collection.

Run with:
    streamlit run frontend/streamlit_app.py

Expects the FastAPI backend running at API_BASE (default localhost:8000).
"""

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄")
st.title("📄 Research Paper Assistant")
st.caption("Hybrid RAG (dense + BM25 + LLM rerank + multi-query) over your papers")

tab_ingest, tab_chat = st.tabs(["Ingest", "Chat"])

with tab_ingest:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    title = st.text_input("Title (optional)")
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Chunking, embedding, and indexing..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            data = {"title": title} if title else {}
            resp = requests.post(f"{API_BASE}/papers/ingest/upload", files=files, data=data)
        if resp.ok:
            st.success(f"Ingested: {resp.json()}")
        else:
            st.error(resp.text)

    st.divider()
    if st.button("Refresh paper list"):
        st.rerun()
    papers_resp = requests.get(f"{API_BASE}/papers")
    if papers_resp.ok:
        st.write(papers_resp.json())

with tab_chat:
    question = st.text_input("Ask a question about your ingested papers")
    if question and st.button("Ask"):
        with st.spinner("Retrieving and reasoning..."):
            resp = requests.post(f"{API_BASE}/chat", json={"question": question, "top_k": 5})
        if resp.ok:
            data = resp.json()
            st.markdown(f"### Answer\n{data['answer']}")
            st.markdown(f"**Confidence:** {data['confidence']:.0%}")
            if data["key_points"]:
                st.markdown("**Key points:**")
                for kp in data["key_points"]:
                    st.markdown(f"- {kp}")
            if data["citations"]:
                st.markdown("**Citations:**")
                for c in data["citations"]:
                    st.markdown(f"> *{c['title']}* — {c['snippet']}")
            with st.expander("Debug: queries used / chunks retrieved"):
                st.write(data["queries_used"])
                st.write(data["retrieved_chunk_ids"])
        else:
            st.error(resp.text)
