from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from pinecone_utils import get_index

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = pipeline("text-generation", model="distilgpt2")

def retrieve(query, top_k=5):
    index = get_index()  # ✅ called only when needed

    query_vector = embedding_model.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]
