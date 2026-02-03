import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = pipeline("text-generation", model="distilgpt2")

# Pinecone connection (index MUST already exist)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("mini-rag")

def retrieve(query, top_k=5):
    query_vector = embedding_model.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]

def rerank(query, matches):
    pairs = [(query, m["metadata"]["text"]) for m in matches]
    scores = reranker.predict(pairs)

    ranked = list(zip(scores, matches))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked

def answer(query, ranked_chunks):
    if not ranked_chunks:
        return "I don't know.", ""

    context = ""
    for i, (_, match) in enumerate(ranked_chunks[:3]):
        context += f"[{i+1}] {match['metadata']['text']}\n\n"

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm(prompt, max_length=300, do_sample=False)
    return response[0]["generated_text"], context
