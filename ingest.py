import os
import uuid
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone connection (index MUST already exist)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("mini-rag")

def ingest_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Chunking
    chunks = []
    chunk_size = 500
    overlap = 50
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    # Embed + upsert
    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": model.encode(chunk).tolist(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors)
