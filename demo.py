import streamlit as st
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from databricks.vector_search.client import VectorSearchClient
from paddleocr import PaddleOCR  # Make sure to import PaddleOCR
from paddleocr import draw_ocr  # Import draw_ocr if not already imported

import streamlit as st
import cv2
import base64
import random
import string
import pandas as pd
from databricks import sql
from dotenv import load_dotenv
import google.generativeai as genai
import os
import PIL.Image
import openai



from IPython.display import display
from IPython.display import Markdown
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from environment variables
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Retrieve Databricks configuration from environment variables
DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


##FILE A CLAIM #

def describe_image(img):
    # Pass the uploaded file directly to PIL.Image.open()
    img = PIL.Image.open(img)
    
    # Call the model to generate a description (mock for now, API does not support images yet)
    response = model.generate_content([ """You are an expert in automobile repair and accident analysis. 
        Analyze the uploaded image of the vehicle and provide a detailed description of the visible damages, addressing the following:
        - List all damaged or broken parts of the vehicle.
        - Note any signs of structural damage.
        - Specify which components need repair or replacement (e.g., windshield, bumper, headlights, tires).
        - Mention any additional issues such as scratches, dents, or cracks.
        - Include an assessment of the overall impact on the vehicle's safety and functionality.
        
        This description is intended for an insurance claim report. Please provide a comprehensive analysis.
        """,img],stream=False)  # Replace with actual functionality when supported
    description = response.text if response else "Description could not be generated."
    
    return description

def create_validation_prompt(description,user_input):
    # Insert the description into the input prompt using an f-string
    prompt = f"""
**Image Description:** {description}
**User Damage Description:** {user_input}

** you are a virtual assistant who verifies if user description matches with the Image description**

- **If both Match:** If the user's description aligns with the image or somewhat match, confirm the matching damages and guide the user to proceed with the claim at [http://127.0.0.1:5500/claim.html].
- **Mismatch:** If discrepancies exist, identify the unmatched damages and request clearer images of those areas for further assessment.

**Strictly follow the Response Template:in Bullet points**

- **Matching:**
    - **Confirmed Damages:** 
        - [List of matching damages]
    - **Next Steps:** Proceed to claim at [http://127.0.0.1:5500/claim.html]

- **Mismatch:**
    - **Unidentified Damages:** 
        - [List of unmatched damages]
    - **Action:** Please upload clearer images of these areas for review.
"""
    return prompt

def matching(user_input,instruction):

    model = genai.GenerativeModel(
    "models/gemini-1.5-flash", system_instruction=instruction)
    response = model.generate_content(user_input)
    print(response.text)
    return response.text

#KNOW YOUR POLICY##

def extract_policyid(question):
    chat_model = ChatDatabricks(endpoint="llm_end_point", max_tokens=50)

    EXTRACT_POLICY_ID_PROMPT = """You are an assistant for an insurance company. Your task is to extract the policy ID from the user's question. Never give any extra sentence.
    If the question does not contain a policy ID, respond with "No policy ID found".

    Question: {query}
    Policy ID:
    """

    prompt = PromptTemplate(template=EXTRACT_POLICY_ID_PROMPT, input_variables=["query"])
    chain = LLMChain(llm=chat_model, prompt=prompt)
    response = chain.run({"query": question})
    return response.strip()


def load_table(table_name):
    connection = sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

def search_policy_in_tables(policy_id):
    table_names = [
        "workspace.rag.claim_data",
        "workspace.rag.policy_data",
        "workspace.rag.contact_info"
    ]
    result = {}
    
    for table_name in table_names:
        df = load_table(table_name)
        filtered_df = df[df["PolicyID"] == policy_id]
        if not filtered_df.empty:
            result[table_name] = filtered_df.to_dict(orient="records")
    
    return result

def retrive_result_from_vector_db(question):
    client = VectorSearchClient(disable_notice=True)
    results = client.get_index("data","workspace.rag.document_id").similarity_search(
        query_text=question,
        columns=["text"],
        num_results=2
    )
    docs = results.get('result', {}).get('data_array', [])
    return docs 

def response(context, customer_data, query):
    chat_model = ChatDatabricks(endpoint="llm_end_point", max_tokens=200)

    CUSTOM_PROMPT = """

    You are an expert assistant for an insurance company, helping insurance agents resolve customer queries efficiently. Your primary responsibility is to provide accurate information and answer questions strictly based on the details provided in the insurance policy documents and the customer data.

    Utilize the information from the Policy Document Information: {context} and Customer Data: {customer_data} to formulate your responses.
    Only respond to queries that are directly related to the insurance policies and the provided context.
    Advise customers to call the toll-free number for further assistance instead of contacting an agent directly.
    Ensure that all responses are aligned with the information in the policy documents and customer data, and do not speculate or provide advice outside the scope of these resources.

    Use the following information to assist the insurance agent in answering the customer's question:
    Policy Document Information:
    {context}

    Customer Data:
    {customer_data}

    Question: {query}
    Answer:
    """

    prompt = PromptTemplate(template=CUSTOM_PROMPT, input_variables=["context", "customer_data", "query"])
    chain = LLMChain(llm=chat_model, prompt=prompt)
    response = chain.run(context=context, customer_data=customer_data, query=query)
    return response
#final function to make the know your policy response
def final_answer(question):
    policy_id,csv = extract_policyid(question)
    documents = retrive_result_from_vector_db(question)
    data = search_policy_in_tables(policy_id)
    final_answer = response(documents, data, question)
    return final_answer

## KYC UPLOAD ##

ocr = PaddleOCR(lang='en')



def do_pdocr(img, to_show=False, showTexts=True, showScores=True):

    img_path = img  # Ensure this path is correct
    img = cv2.imread(img_path)  # Load the image
    # Check if the input image is valid
    if img is None or not isinstance(img, (np.ndarray,)):
        raise ValueError("Input must be a valid image array.")
    
    # Convert the image to RGB
    im_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform OCR
    result = ocr.ocr(im_show, cls=False)  
    
    if to_show:  
        # Extract bounding boxes, texts, and scores from the result
        boxes = [] if not result else [line[0] for line in result[0]]
        txts = None if not result else [line[1][0] for line in result[0]]  
        scores = None if not result else [line[1][1] for line in result[0]] 
        
        # Draw the OCR results on the image
        im_show = draw_ocr(im_show, boxes, 
                           txts=None if not showTexts else txts,
                           scores=None if not showScores else scores,
                           font_path='C:/Windows/Fonts/Arial.ttf')
    
    return im_show, txts


def ensure_directory_exists(directory_path):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_recognized_text_to_txt(user_id, recognized_texts):
    """Save the entire recognized text to a text file in the 'recognized_texts' folder using user ID as filename."""
    # Define the folder for recognized texts and ensure it exists
    recognized_text_folder = "output/recognized_texts"
    ensure_directory_exists(recognized_text_folder)
    
    filename = os.path.join(recognized_text_folder, f"{user_id}_recognized_text.txt")  # File path with folder
    with open(filename, "w") as file:
        for text in recognized_texts:
            file.write(f"{text}\n")  # Write each recognized text on a new line

def save_captured_image(image, user_id):
    """Save the captured image in the 'captured_images' folder using user ID as part of the filename."""
    # Define the folder for captured images and ensure it exists
    captured_image_folder = "output/captured_images"
    ensure_directory_exists(captured_image_folder)

    filename = os.path.join(captured_image_folder, f"{user_id}_captured_id.png")  # File path with folder
    cv2.imwrite(filename, image)  # Save the image
    return filename  # Return the file path 

def structure_recognized_text(user_id):
    """Read and structure the recognized text for display from the 'recognized_texts' folder."""
    filename = f"output/recognized_texts/{user_id}_recognized_text.txt"  # Use the folder path and user ID
    structured_data = {}

    # Ensure the file exists
    if os.path.exists(filename):
        with open(filename, "r") as file:
            lines = file.readlines()

        # Parse and structure the text
        for line in lines:
            line = line.strip()  # Clean the line from spaces and newlines
            
            # Only extract the required fields
            if "LICENSE" in line or "DL" in line:
                structured_data["License Number"] = line
            elif "FN" in line:
                structured_data["First Name"] = line.split("FN")[1].strip()
            elif "LN" in line:
                structured_data["Last Name"] = line.split("LN")[1].strip()
            elif "DOB" in line:
                structured_data["Date of Birth"] = line.split("DOB")[1].strip()

    return structured_data
# Function to display structured text in Streamlit
def display_structured_text(structured_data):
    """Display the structured recognized text in Streamlit."""
    # st.title("Recognized Driver's License Information")

    if structured_data:
        with st.form("confirm_details_form"):
            for key, value in structured_data.items():
                st.write(f"**{key}:** {value}")
                # Checkbox for the user to confirm the data
                st.checkbox(f"Is the above {key} correct?", value=True)

            # Submit button
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.success(f"Hi {structured_data.get('First Name', 'User')}, you can now upload any relevant photos and explain what happened in your claim.")
            # Additional functionality for uploading files and further instructions can be added here.
    else:
        st.write("No structured data available")
#string to make a file name
def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))    

