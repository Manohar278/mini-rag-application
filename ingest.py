from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone_utils import get_index
import uuid

model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    chunks = []
    chunk_size = 500
    overlap = 50
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    index = get_index()

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": model.encode(chunk).tolist(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors)
