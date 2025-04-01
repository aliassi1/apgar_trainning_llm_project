# Import necessary libraries for vector store, embeddings, and LLM operations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import tool, StructuredTool
from typing import Optional
import os
from embed import index
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Google's embedding model for converting text to vectors
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.environ["API_KEY"],  
    )

# Set up Pinecone vector store with the embeddings model
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
# Initialize Google's Gemini model for question answering
llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=os.environ["API_KEY"], 
            temperature=0.5  # Controls randomness in responses
        )

# Create a retriever that will fetch the 4 most similar documents
retriever = vector_store.as_retriever(search_type="similarity", k=4)

# Set up a question-answering chain that combines the LLM with the retriever
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,  # Include source documents in the response
    )

# Define the document retrieval tool using the @tool decorator
@tool
def document_ret(query: str) -> str:
    """
    This tool is to retrieve data from documentation vector store used when user ask about company policies.
    Args:
        query (str): question about documentation or company policies.
    Returns:
        str: The data retrieved from the documentation vector store with the source document.
    """
   
    # Get response from the QA chain
    response = qa_chain.invoke({"query": query})
    
    # Combine the answer with the source document information
    result = f"{response['result']}\n\nSource: {response['source_documents'][0].metadata['source']}"
    return str(result)




