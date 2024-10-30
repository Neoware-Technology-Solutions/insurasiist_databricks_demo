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






import chromadb

# Initialize ChromaDB client

client = chromadb.PersistentClient(path="./content")

# Create or access a collection
collection = client.create_collection(name="chunked_text_files_collections")

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# File paths of your text files
file_paths = [r"data/input/Buyers_guide.txt", r"data/input/Consumer-Bill-Of-Rights.txt", r"data/input/Declaration.txt"]

# Process each file and store in chunks
for file_path in file_paths:
    with open(file_path, "r") as file:
        text_content = file.read()
        
        # Split the text into chunks (e.g., 500 characters per chunk)
        text_chunks = chunk_text(text_content, chunk_size=500)
        
        # Extract metadata (e.g., filename)
        file_name = file_path.split("/")[-1]
        
        # Add each chunk as a document with metadata
        for idx, chunk in enumerate(text_chunks):
            collection.add(
                documents=[chunk],  # Chunk content
                metadatas=[{"filename": file_name, "chunk_id": idx}],  # Metadata, including chunk ID
                ids=[f"{file_name}_chunk_{idx}"]  # Unique ID for each chunk
            )

print("Text files stored as chunks in ChromaDB!")
