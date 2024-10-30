import streamlit as st
import chromadb
import openai
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
import cv2
from paddleocr import PaddleOCR  # Make sure to import PaddleOCR
from paddleocr import draw_ocr  # Import draw_ocr if not already imported
import numpy as np
import base64
import random
import string
import streamlit.components.v1 as components
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from PIL import Image
import PIL.Image
import time
from PIL import Image




from src.policy import retrive_result_from_vector_db,response,fetch_policyid,final_answer,get_policy_data_and_filter
from src.claim import describe_image,create_validation_prompt,matching
from src.kyc import save_recognized_text_to_txt,save_captured_image,display_structured_text,generate_random_string,ensure_directory_exists,structure_recognized_text,do_pdocr,create_file_path


client = chromadb.PersistentClient(path="./content")

# Create or access a collection
collection = client.get_collection(name="chunked_text_files_collections")



# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')










scroll_to_top_js = """
    <script>
        window.onload = function() {
            window.scrollTo(0, 0);
        }
    </script>
"""

if 'policy_chat_messages' not in st.session_state:
    st.session_state.policy_chat_messages = []
if "claim_submitted" not in st.session_state:
    st.session_state.claim_submitted = False    

if 'claim_chat_messages' not in st.session_state:
    st.session_state.claim_chat_messages = []

if "structured_data" not in st.session_state:
    st.session_state.structured_data = {}

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'learn'  # Default to learning about policy



# Placeholder for chat messages
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #2c3e50;  /* Dark blue */
        color: white;  /* Text color for other content */
        border-radius: 10px;  /* Rounded corners */
        padding: 20px;  /* Padding inside sidebar */
    }
    h2 {
        font-size: 24px;  /* Larger font size for the header */
        text-align: left;  /* Center align the header */
        color: white;  /* White color for the header text */
    }
    p {
        font-size: 16px;  /* Font size for the description */
        text-align: left;  /* Center align the description */
        color: black;  /* White color for the description text */
    }
    .button {
        background-color: black;  /* Black button background */
        color: black;  /* Black text for the button */
        border: none;  /* Remove border */
        border-radius: 5px;  /* Rounded corners for button */
        padding: 10px 20px;  /* Padding for button */
        margin: 10px 0;  /* Margin for spacing between buttons */
        text-align: left;  /* Center text in button */
        display: block;  /* Make button a block element */
        font-size: 16px;  /* Font size for the button text */
        cursor: pointer;  /* Change cursor to pointer */
</style>
""", unsafe_allow_html=True)


image_path = "data/ui_img/download.png"  # Replace with your image file path
image = Image.open(image_path)
st.sidebar.image(image, caption="Your Caption Here",use_column_width=True)

# Embed the video in the sidebar


st.sidebar.markdown("<h2>Hey, it's Neo!</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>Let me help you with your insurance.</h2>", unsafe_allow_html=True)
# check1 = st.sidebar.button("Know your Policy 📄")
learn_button = st.sidebar.button("Know your Policy 📄")
claim_button = st.sidebar.button("File a Claim 📝 📝")
kyc_button = st.sidebar.button("KYC Process 🔍 📸")

if learn_button:
    st.session_state.current_view = 'learn'
elif claim_button:
    st.session_state.current_view = 'claim'
elif kyc_button:
    st.session_state.current_view = 'KYC'    


if st.session_state.current_view == 'learn':
    
# if check1:
    st.title("Know Your Policy")
    st.write("💬 Have questions about your insurance? I'm here to help! 🤝 Ask away, and let’s simplify your policy together!.")
    
    # Chat functionality for learning about policies
    for message in st.session_state.policy_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("How can I help you..."):
        st.session_state.policy_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Processing your question..."):
        # Replace this with your logic to generate a response about the policy
            response = final_answer(prompt)  # Example function call
        st.session_state.policy_chat_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


elif st.session_state.current_view == 'KYC':
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
    st.title("**Let's complete your KYC verification.**")
   

    random_suffix = generate_random_string()
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    
    if picture:
         with st.spinner("Please wait, it will take a few seconds..."):
                # Simulate processing time
            time.sleep(5)
            picture_filename = f"kyc_image_{random_suffix}.png"  # Filename with random suffix

            output_directory = "data/output/taken_pic"  # Path to the desired folder

# Create the directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Complete file path
            file_path = os.path.join(output_directory, picture_filename)

            # Save the image to the specified directory
            with open(file_path, "wb") as f:
                f.write(picture.getbuffer())  # Assuming picture is a BytesIO object
        
            if st.button("Confirm KYC Video"):
                st.write("KYC video uploaded successfully. Please review and confirm your details and proceed to the next step.")
                processed_image, recognized_texts = do_pdocr(file_path, to_show=True, showTexts=True, showScores=True)
                processed_image = save_captured_image(processed_image, user_id=f"aa11{random_suffix}")
                print("xxxxxx", recognized_texts)
                save_recognized_text_to_txt(random_suffix, recognized_texts)
                st.session_state.structured_data = structure_recognized_text(random_suffix)
            if st.session_state.structured_data:
                st.session_state.name = st.session_state.structured_data.get("First Name", "") + " " + st.session_state.structured_data.get("Last Name", "")
                st.session_state.license_number = st.session_state.structured_data.get("License Number", "")
                # Check if structured data exists
                first_name = st.session_state.structured_data.get("First Name", "")
                last_name = st.session_state.structured_data.get("Last Name", "")
                dob = st.session_state.structured_data.get("Date of Birth", "")
                license_number = st.session_state.structured_data.get("License Number", "")

                st.text_input("First Name", value=first_name)
                st.text_input("Last Name", value=last_name)
                st.text_input("License Number", value=license_number)

                # Use session state to handle form submission
                if st.button("Confirm and Submit"):
                    st.session_state.claim_submitted = True

            # Show success message if the claim is submitted
            if st.session_state.claim_submitted:
                st.success(f"Hi {first_name}, your license number is {license_number}. Your KYC information has been submitted successfully! .")


elif st.session_state.current_view == 'claim':

    st.markdown("""
    ### Instructions:
    - Upload an asset photo related to your claim.
    - Describe the damages observed in the asset in the text box.
    - Click the 'Submit' button.
""")
    # st.write(f"Name: {st.session_state.name}")
    # st.write(f"License Number: {st.session_state.license_number}")
    uploaded_file = st.file_uploader("Upload an Asset Photo for Your Claim", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Please describe the damages observed in the asset:")

 
        # Open the uploaded image using PIL
        

        # Button to submit the response
    if st.button("Submit"):
            with st.spinner("Please wait, it will take a few seconds..."):
                # Simulate processing time
                time.sleep(5)  # Replace this with your actual processing code
                image = PIL.Image.open(uploaded_file)

        # Display the uploaded image
                st.image(image, caption="Uploaded Image", width=300)
                
                # Generate the AI description
                description = describe_image(uploaded_file)
                
                # Create a validation prompt for comparison
                input_prompt = create_validation_prompt(description, prompt)

                # Compare user input with AI description
                result = matching(prompt, input_prompt)
                response = f"The description for your claim is: {result}"

                # Display the result
                st.success(response)
    else:
        st.warning("Please upload an image and describe the damages before submitting.")

components.html(scroll_to_top_js)                

