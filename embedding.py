from embed import index
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ.get("API_KEY"),  # API key for authentication
)


vector_store = PineconeVectorStore(index=index, embedding=embeddings)

with open("policies.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create the Document object
doc = Document(page_content=text, metadata={"source": "policies.txt"})

vector_store.add_documents([doc])




