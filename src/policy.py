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



from IPython.display import display
from IPython.display import Markdown
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from environment variables
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

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

def create_validation_prompt(description):
    # Insert the description into the input prompt using an f-string
    prompt = f"""
      
        Please check if the damages described by the user align with the condition noted in the AI-generated description.
        
        **AI Description of Vehicle Damages:**
        {description}

        **User Description of Damages:**
        {{user_input}}

        Based on the descriptions provided, do the user's observations about the damages match the AI's assessment? 
        If they match or partially match, confirm that the user has accurately described the damages. 
        If there are discrepancies, please specify which damages were correctly identified and which were missed.
    """
    return prompt

def matching(user_input,instruction):

    model = genai.GenerativeModel(
    "models/gemini-1.5-flash", system_instruction=instruction)
    response = model.generate_content(user_input)
    print(response.text)
    return response.text


# Retrieve Databricks configuration from environment variables
DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

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

    CUSTOM_PROMPT = """You are an expert assistant for an insurance company, helping insurance agents resolve customer queries efficiently. 
    Your role is to provide accurate information and answer questions based on the provided context. 
    If the question is not related to insurance policies or the provided context, kindly decline to answer. 
    If you don't know the answer, simply state that you don't know; do not attempt to fabricate a response. 
    Keep your answers concise and to the point.

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

def final_answer(question):
    policy_id,csv = extract_policyid(question)
    documents = retrive_result_from_vector_db(question)
    data = search_policy_in_tables(policy_id)
    final_answer = response(documents, data, question)
    return final_answer










   
 

