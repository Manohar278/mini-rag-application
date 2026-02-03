# rag.py
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# -------------------------
# Embedding model
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Pinecone connection
# -------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("mini-rag")

# -------------------------
# LLM (FLAN-T5)
# -------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

# -------------------------
# Retrieve from Pinecone
# -------------------------
def retrieve(question, top_k=5):
    query_embedding = embed_model.encode(question).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]

# -------------------------
# Rerank (simple pass-through)
# -------------------------
def rerank(question, matches):
    return matches

# -------------------------
# Generate answer
# -------------------------
def answer(question, matches):
    context = ""
    for i, match in enumerate(matches):
        if match.metadata and "text" in match.metadata:
            context += f"[{i+1}] {match.metadata['text']}\n\n"

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""".strip()

    response = llm(prompt)[0]["generated_text"]
    return response, context
