import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_pdf(pdf_path):
    # Pinecone initialized INSIDE the function
    pc = Pinecone(api_key=os.environ["pcsk_7N8TDx_T7CDM66rqR9VvwPRVHfCZd1iJejAdNkA1VStAXXRCcnZeRBcWMibb5pHyXCBfSk"])
    index = pc.Index("mini-rag")

    # your existing ingestion logic continues here
