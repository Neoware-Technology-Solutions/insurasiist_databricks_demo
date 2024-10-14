
# Insurance Assistant Chatbot

## Overview
The Insurance Assistant Chatbot is designed to assist insurance agents in efficiently resolving customer queries related to insurance policies. This AI-driven tool leverages advanced embedding and language models available on Databricks to provide accurate, context-aware answers based on both structured customer data and unstructured policy information.

## Features
- **Customer Query Handling**: Agents can ask questions about customer insurance policies, and the assistant will provide answers based on the provided context.
- **Image Analysis**: Users can upload images of accident-damaged vehicles along with descriptions. The assistant will analyze the claims against the images to verify their accuracy.
- **Structured and Unstructured Data Utilization**: The assistant can effectively use structured data from customer databases and unstructured data from policy documents.

## Technologies Used
- **Embedding Model**: `system.ai.bge_large_en_v1_5` for creating embeddings from unstructured policy information.
- **Language Model**: `mistralsystem.ai.mistral_7b_instruct_v0_2` for generating responses to customer queries based on the context provided.
- **Databricks**: The application runs on Databricks, utilizing its powerful data processing capabilities.

## How It Works
1. **Data Ingestion**: 
   - Policy documents (unstructured data) and customer data (structured data) are ingested into the system.
   - The unstructured policy details are transformed into embeddings using the embedding model and stored in a Delta table for efficient retrieval.
   - The structured customer data is stored in designated tables for easy access and management.

2. **Query Processing**: 
   - When an agent poses a question, the assistant retrieves relevant information from both structured and unstructured sources. 
   - It leverages the embeddings stored in the Delta table to enhance the accuracy of responses.

3. **Image Upload and Analysis**:
   - Users can upload images of vehicles involved in accidents.
   - They provide a description of the damage.
   - The assistant checks the user's claims against the uploaded images to validate their accuracy.

## Setup Instructions
To set up the Insurance Assistant Chatbot, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Install Required Dependencies**:
   pip install -r requirements.txt
3. **Install Required Dependencies**:
   pip install -r requirements.txt
4. **Running the Application: Start the application using Streamlit:**:
   streamlit run main.py

   


5. **Install Required Dependencies: Install the required Python packages using pip: **  
