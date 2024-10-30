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
import openai
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os


openai.api_key = "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"

client = chromadb.PersistentClient(path="./content")

# Create or access a collection
collection = client.get_collection(name="chunked_text_files_collections")



# Load environment variables from .env file
load_dotenv()
API_KEY=os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY





def fetch_policyid(user_question):
    prompt = """

    User Question: {user_question}

    Your task is to extract the policy id from the user's question, if mentioned, and return it. 
    Follow these instructions strictly:
    - Only return the policy id with no additional text.
    - Do not include "PolicyID:" or any other extra wording—only the number.
    - You have three CSVs to check for data retrieval:
        1. Claim.csv - Contains columns: ClaimID, PolicyID, Name, ClaimAmount, ClaimType, ClaimStatus, Claimed Date, Claim Settlement Date, Date of Loss.
        2. contact_info.csv - Contains columns: PolicyID, Name, Phone number, E-mail ID.

    Based on the user query, determine which CSV file is most suitable for retrieving information related to the **PolicyID**.

    Response template:
    Policy Id: [PolicyID]
    Csv: [CSV file name]

    
    """
 

   
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question}
        ]
    )
 

 
    response_text = response.choices[0].message.content.strip()
    print("xxxxx",response_text)

# Assuming the response follows the template: "Policy Id: [PolicyID]\nCsv: [CSV file name]"
# Split the response to get Policy Id and Csv
    lines = response_text.splitlines()
    policy_id = lines[0].split(":")[1].strip()  # Extract the Policy ID
    csv_name = lines[1].split(":")[1].strip()  # Extract the CSV name

    # Return both Policy ID and CSV name
    return policy_id, csv_name


def get_policy_data_and_filter(policy_id, csv_name):
    # Load the appropriate CSV file based on the OpenAI response
    if csv_name.lower() == 'claim.csv':
        df = pd.read_csv('data/input/customer_database/claim_data.csv')
    elif csv_name.lower() == 'contact_info.csv':
        df = pd.read_csv('data/input/customer_database/contact_info.csv')
    else:
        return "Invalid CSV file name."

    # Filter the selected CSV by PolicyID
    filtered_df = df[df['PolicyID'] == policy_id]
    print("filter",filtered_df)
    
    # If no matching records, return a message
    if filtered_df.empty:
        return f"No data found for Policy ID: {policy_id} in {csv_name}"

    # Load 'policy_data.csv' and filter it by the same PolicyID
    try:
        policy_data_df = pd.read_csv('data/input/customer_database/policy_data.csv')
        filtered_policy_data_df = policy_data_df[policy_data_df['PolicyID'] == policy_id]
    except FileNotFoundError:
        return "'policy_data.csv' not found."

    # If no matching records in policy_data.csv, return a message
    if filtered_policy_data_df.empty:
        policy_data_msg = f"No data found for Policy ID: {policy_id} in 'policy_data.csv'"
    else:
        # Include all matching records
        policy_data_msg = "\n".join([f"{', '.join([f'{col}: {val}' for col, val in row.items()])}" for _, row in filtered_policy_data_df.iterrows()])

    # Prepare the result from the selected CSV
    selected_csv_msg = "\n".join([f"{', '.join([f'{col}: {val}' for col, val in row.items()])}" for _, row in filtered_df.iterrows()])

    # Return the results from both the selected CSV and the 'policy_data.csv'
    return f"Selected CSV ({csv_name}) Data:\n{selected_csv_msg}\n\n'policy_data.csv' Data:\n{policy_data_msg}"
def retrive_result_from_vector_db(query):
    results = collection.query(query_texts=query, n_results=5, include=['documents','embeddings'])
    print("aaaaaa",results)
    retrieved_documents= results['documents'][0]

    return retrieved_documents
 
def response(context, customer_data, query):
    # Define the prompt, combining context and customer data into the system message
    prompt = f"""
You are an expert assistant for an insurance company, helping insurance agents resolve customer queries efficiently. Your primary responsibility is to provide accurate information and answer questions strictly based on the details provided in the insurance policy documents and the customer data.

Utilize the information from both Policy Document Information: {context} and Customer Data: {customer_data} to formulate your responses.
Only respond to queries that are directly related to the insurance policies and the provided context.
Advise customers to call the toll-free number for further assistance instead of contacting an agent directly.
Ensure that all responses are aligned with the information in the policy documents and customer data, and do not speculate or provide advice outside the scope of these resources.

**Policy Document Information:**
{context}

**Customer Data:**
{customer_data}

**Question:** {query}

**Answer:**
"""

    # Structure the messages properly for the API call
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}  # Only include the user's question
    ]

    # Call OpenAI API
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    # Extract the response text
    response_text = response.choices[0].message.content.strip()
    print("RRRRRR", response_text)
    return response_text  # Return the response text for further use

def final_answer(question):
    policy_id,csv =fetch_policyid (question)
    print("xxx",policy_id)
    data=get_policy_data_and_filter(policy_id,csv)
    print("abcdef",data)
    documents = retrive_result_from_vector_db(question)
    print("yyyy",documents)
    final_answer = response(documents, data, question)
    return final_answer
