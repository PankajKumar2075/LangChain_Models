import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found in .env file"
    )

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
    max_new_tokens=128,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)