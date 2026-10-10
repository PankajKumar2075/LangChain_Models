from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documnets = [
    "Delhi is the capital of India",
    "Kolkata is the capital of west bengal",
    "Paris is the capital of france"
]

result = embedding.embed_documnets(documnets)

print(str(result))