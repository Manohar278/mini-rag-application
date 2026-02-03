from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import uuid

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone connection (simple)
pc = Pinecone(api_key=os.environ["pcsk_7N8TDx_T7CDM66rqR9VvwPRVHfCZd1iJejAdNkA1VStAXXRCcnZeRBcWMibb5pHyXCBfSk"])
index = pc.Index("mini-rag")   # ⚠️ index MUST already exist

def ingest_pdf(pdf_path):
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

    # Embed + upload
    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": model.encode(chunk).tolist(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors)
