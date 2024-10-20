import streamlit as st
import os
import cv2
import easyocr
import numpy as np
from PIL import Image
from groq import Groq
import pandas as pd
from paddleocr import PaddleOCR

os.environ['GROQ_API_KEY'] = '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'

client = Groq()
ocr = PaddleOCR(use_angle_cls=True, lang='en') 

def read_text_from_image(image_path):
    try:
        result = ocr.ocr(image_path)
        text = ' '.join([line[1][0] for line in result[0]])  
        return text.strip()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contrast = cv2.equalizeHist(img)
    return contrast

def extract_product_info(text):
    try:
        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Extract the following details from the product description below:
                    - Product Name
                    - Brand Name
                    - Size
                    - MRP (Maximum Retail Price)
                    - Expiry Date (Best before)
                    - Manufacturing Date (Mft dt)

                    Product Description:
                    {text}

                    Output the information in the following format:
                    Product Name: <product_name>
                    Brand Name: <brand_name>
                    Size: <size>
                    MRP: <mrp>
                    Expiry Date: <expiry_date>
                    Manufacturing Date: <manufacturing_date>
                    """
                }
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.65,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while extracting product info: {e}")
        return None

def parse_product_info(info_text):
    lines = info_text.strip().split('\n')
    info_dict = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            info_dict[key.strip()] = value.strip()
    return info_dict

st.title("Product Label OCR and Information Extractor")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
       st.subheader("Product Label:")
       st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    processed_image = process_image(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


    with col2:
        with st.spinner("Performing OCR..."):
            extracted_text = read_text_from_image(processed_image)

        if extracted_text:

            with st.spinner("Extracting product information..."):
                product_info = extract_product_info(extracted_text)

            if product_info:
                st.subheader("Product Information:")
                info_dict = parse_product_info(product_info)
            
                df = pd.DataFrame(list(info_dict.items()), columns=['Attribute', 'Value'])
            
                st.table(df)
            else:
                st.warning("Failed to extract product information. Please try again.")
        else:
            st.error("Failed to extract text from the image. Please try another image.")
   