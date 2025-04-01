import os
from dotenv import load_dotenv
import time
from pinecone import Pinecone,ServerlessSpec

load_dotenv()


pc = Pinecone(api_key=os.environ["API_KEY_2"])


index_name = "embeddings-index"  
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

