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
st.title("Insurassist")
# Configure other settings and functions...

# Function to render the card UI
if 'policy_chat_messages' not in st.session_state:
    st.session_state.policy_chat_messages = []

if 'claim_chat_messages' not in st.session_state:
    st.session_state.claim_chat_messages = []

# Display chat history
st.sidebar.title("Options")
option = st.sidebar.selectbox(
    "What would you like to do?",
    ("Learn about Policy", "File a Claim")
)

# Conditional rendering based on selected option
if option == "Learn about Policy":
    st.title("Learn about Policy")
    st.write("Here you can find information about various insurance policies.")
    
    # Chat functionality for learning about policies
    for message in st.session_state.policy_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the policy..."):
        st.session_state.policy_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Replace this with your logic to generate a response about the policy
        response = final_answer(prompt)  # Example function call
        st.session_state.policy_chat_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

elif option == "File a Claim":
    # st.title("File a Claim")
    st.write("Follow the steps to file a claim.")
    st.write("1. **Upload an image related to your claim.**")
    st.write("2. **Provide a brief description of the claim.**")
    st.write("3. **Click 'Submit' to file your claim.**")
    for i in range(1):
     uploaded_file=st.file_uploader(f"Upload your asset's image here:")

    css = '''
<style>
    /* Style for the specific file uploader */
    [data-testid='stFileUploader'] {
        width: max-content; /* Set the width to fit the content */
    }
    [data-testid='stFileUploader'] section {
        padding: 0; /* Remove padding */
        float: left; /* Align uploader to the left */
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none; /* Hide the default drag-and-drop area */
    }
    [data-testid='stFileUploader'] section + div {
        float: right; /* Align the button to the right */
        padding-top: 0; /* Remove top padding */
    }
    /* Reduce line spacing between elements */
    .stWrite {
        margin-bottom: 5px; /* Adjust bottom margin as needed */
    }
    .stFileUploader {
        margin-bottom: 5px; /* Adjust bottom margin as needed */
    }
</style>
'''

    st.markdown(css, unsafe_allow_html=True)

    
    # Image upload functionality for filing a claim
    # uploaded_file = st.file_uploader("Upload an image for your claim", type=["png", "jpg", "jpeg", "jfif"])

    # Chat functionality for filing a claim
    for message in st.session_state.claim_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about filing a claim..."):
        st.session_state.claim_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if uploaded_file:
            # Display the uploaded image
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', width=300) 

            # Generate the AI description and display it
            st.write("Generating description for the image...")
            description = describe_image(uploaded_file)
            # st.write("AI Description:", description)

            # Create a validation prompt for comparison
            input_prompt = create_validation_prompt(description,prompt)

            # Compare user input (prompt) with AI description
            result = matching(prompt, input_prompt)
            response = f"The description for your claim is: {result}"  # Example response
            st.session_state.claim_chat_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            