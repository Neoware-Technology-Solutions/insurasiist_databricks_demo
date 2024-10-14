import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from databricks.vector_search.client import VectorSearchClient
import streamlit as st
import pandas as pd
from databricks import sql
from dotenv import load_dotenv
import google.generativeai as genai
import os
import PIL.Image
import openai
from demo import final_answer,response,retrive_result_from_vector_db,search_policy_in_tables,describe_image,create_validation_prompt,matching,extract_policyid,load_table
from IPython.display import display
from IPython.display import Markdown
import os

# Load environment variables from .env file for secure access to sensitive data
load_dotenv()

# Set API key for Google Generative AI
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Initialize the Generative AI model
model = genai.GenerativeModel('gemini-1.5-flash')

# Retrieve Databricks configuration from environment variables
DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Streamlit app title
st.title("Insurance Agent Assistant")

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

# File uploader for optional image input
uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

# User input handling
if prompt := st.chat_input("What is up?"):
    # Add the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if an image was uploaded
    if not uploaded_file:
        # Get assistant response without image
        response = final_answer(prompt)  # Call the function to get assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})    
    else:
        # Display the uploaded image
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Generate AI description for the uploaded image
        st.write("Generating description for the image...")
        description = describe_image(uploaded_file)
        st.write("AI Description:", description)

        # Create a validation prompt for comparison with user input
        input_prompt = create_validation_prompt(description)

        # Compare user input (prompt) with AI-generated description
        result = matching(prompt, input_prompt)
        with st.chat_message("assistant"):
            st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})    