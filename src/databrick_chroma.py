__import__("pysqlite3")
import sys
import os

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join("./", "db.sqlite3"),
    }
}







import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./content")

# Create or access a collection
collection = client.create_collection(name="chunked_text_files_collections1")

# How to get your Databricks token

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Initialize the OpenAI client
openai_client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-fdab60e7-1257.cloud.databricks.com/serving-endpoints"
    
)

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to create embeddings using the Databricks model
def create_embedding(input_string):
    embeddings = openai_client.embeddings.create(
        input=input_string,
        model="embed_end_point" # Use your specific model name here
    )
    return embeddings.data[0].embedding

# File paths of your text files
file_paths = [r"data/input/policy_doc/Buyers_guide.txt", r"data/input/policy_doc/Consumer-Bill-Of-Rights.txt", r"data/input/policy_doc/Declaration.txt"]

# Process each file and store in chunks
for file_path in file_paths:
    with open(file_path, "r") as file:
        text_content = file.read()
        
        # Split the text into chunks (e.g., 500 characters per chunk)
        text_chunks = chunk_text(text_content, chunk_size=500)
        
        # Extract metadata (e.g., filename)
        file_name = os.path.basename(file_path)
        
        # Add each chunk as a document with metadata
        for idx, chunk in enumerate(text_chunks):
            # Create an embedding for the chunk
            embedding = create_embedding(chunk)
            
            # Add each chunk and its embedding as a document
            collection.add(
                documents=[chunk],  # Chunk content
                metadatas=[{"filename": file_name, "chunk_id": idx}],  # Metadata, including chunk ID
                ids=[f"{file_name}_chunk_{idx}"],  # Unique ID for each chunk
                embeddings=[embedding]  # Store the embedding
            )

print("Text files stored as chunks with embeddings in ChromaDB!")
