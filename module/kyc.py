import streamlit as st
import numpy as np
from paddleocr import PaddleOCR  # Make sure to import PaddleOCR
from paddleocr import draw_ocr  # Import draw_ocr if not already imported
import streamlit as st
import cv2
import base64
import random
import string
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import os
import PIL.Image
import openai
from IPython.display import display
from IPython.display import Markdown
import os



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
    recognized_text_folder = "data/output/recognized_texts"
    ensure_directory_exists(recognized_text_folder)
    
    filename = os.path.join(recognized_text_folder, f"{user_id}_recognized_text.txt")  # File path with folder
    with open(filename, "w") as file:
        for text in recognized_texts:
            file.write(f"{text}\n")  # Write each recognized text on a new line

def save_captured_image(image, user_id):
    """Save the captured image in the 'captured_images' folder using user ID as part of the filename."""
    # Define the folder for captured images and ensure it exists
    captured_image_folder = "data/output/captured_images"
    ensure_directory_exists(captured_image_folder)

    filename = os.path.join(captured_image_folder, f"{user_id}_captured_id.png")  # File path with folder
    cv2.imwrite(filename, image)  # Save the image
    return filename  # Return the file path 

def structure_recognized_text(user_id):
    """Read and structure the recognized text for display from the 'recognized_texts' folder."""
    filename = f"data/output/recognized_texts/{user_id}_recognized_text.txt"  # Use the folder path and user ID
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

#string to make a file name
def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))    

