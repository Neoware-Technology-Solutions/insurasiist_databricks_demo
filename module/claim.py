import os
import time
from PIL import Image
import openai
import pandas as pd
import csv
import chromadb
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import PIL.Image
import openai
import streamlit as st
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
import os


# Load environment variables from .env file
load_dotenv()

# Access the API key from environment variables
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')



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
