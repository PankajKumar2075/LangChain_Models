from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Delhi is the capital of India.",
    "LangChain is a framework for developing applications powered by language models.",
    "Hugging Face provides pretrained machine learning models.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store and retrieve embeddings for semantic search."
]

query = 'tell me about viral kohli'

doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1][-1])

print(query)

print(documents[index])

print("similarity score is:", score)