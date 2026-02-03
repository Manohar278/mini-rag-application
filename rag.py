import os
from pinecone import Pinecone

def retrieve(query):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("mini-rag")
    # retrieval logic
