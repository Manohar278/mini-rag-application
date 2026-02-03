import streamlit as st
from ingest import ingest_pdf
from rag import retrieve, rerank, answer

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(page_title="Mini RAG AI Application")

st.title("📄 Mini RAG AI Application")
st.write("Upload a document and ask questions based on its content.")

# ----------------------------------
# 1️⃣ Upload & Ingest PDF
# ----------------------------------
st.header("1️⃣ Upload Document")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Ingest Document"):
        ingest_pdf("temp.pdf")
        st.success("✅ Document ingested and stored in Pinecone")

# ----------------------------------
# 2️⃣ Ask Question
# ----------------------------------
st.header("2️⃣ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Retrieve chunks
        matches = retrieve(question)

        # Rerank chunks
        ranked_chunks = rerank(question, matches)

        # Generate answer
        final_answer, context = answer(question, ranked_chunks)

        # ----------------------------------
        # Answer output
        # ----------------------------------
        st.subheader("✅ Answer")
        st.write(final_answer)

        # ----------------------------------
        # Sources / Citations
        # ----------------------------------
        st.subheader("📚 Sources")
        st.text(context)
