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

def create_validation_prompt(description,user_input):
    # Insert the description into the input prompt using an f-string
    prompt = f"""
    **Image Description:** {description}
    **User Damage Description:** {user_input}

    ** you are a virtual assistant who verifies if user description matches with the Image description**

    - **If both Match:** If the user's description aligns with the image or somewhat match, confirm the matching damages and guide the user to proceed with the claim at [http://127.0.0.1:5500/claim.html].
    - **Mismatch:** If discrepancies exist, identify the unmatched damages and request clearer images of those areas for further assessment.

    **Strictly follow the Response Template:**

    - **Matching:**
        -**Confirmed Damages:**
        -** [List of matching damages]**
        - **Next Steps:** Proceed to claim at [http://127.0.0.1:5500/claim.html]

    - **Mismatch:**
        - **Unidentified Damages:** [List of unmatched damages]
        - **Action:** Please upload clearer images of these areas for review.
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

def final_answer(question):
    policy_id,csv = extract_policyid(question)
    documents = retrive_result_from_vector_db(question)
    data = search_policy_in_tables(policy_id)
    final_answer = response(documents, data, question)
    return final_answer

st.title("Insurance Agent Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])


if prompt := st.chat_input("What is up?"):
        # Add the user's question to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if not uploaded_file:
            response = final_answer(prompt)  # Get assistant response using final_answer
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})    
        if uploaded_file:
        # Display the uploaded image
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Generate the AI description and display it
            st.write("Generating description for the image...")
            description = describe_image(uploaded_file)
            st.write("AI Description:", description)

            # Create a validation prompt for comparison
            input_prompt = create_validation_prompt(description)

            # Compare user input (prompt) with AI description
            result = matching(prompt, input_prompt)
            with st.chat_message("assistant"):
                st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})    
            