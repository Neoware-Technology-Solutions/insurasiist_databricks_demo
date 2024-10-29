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
