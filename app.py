import streamlit as st
from ingest import ingest_pdf
from rag import retrieve, rerank, answer

st.set_page_config(page_title="Mini RAG AI Application")

st.title("📄 Mini RAG AI Application")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Ingest Document"):
        ingest_pdf("temp.pdf")
        st.success("Document ingested successfully")

question = st.text_input("Ask a question")

if st.button("Get Answer"):
    matches = retrieve(question)
    ranked = rerank(question, matches)
    final_answer, context = answer(question, ranked)

    st.subheader("Answer")
    st.write(final_answer)

    st.subheader("Sources")
    st.text(context)
