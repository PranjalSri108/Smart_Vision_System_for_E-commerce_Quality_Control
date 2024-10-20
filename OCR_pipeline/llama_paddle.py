import os
import cv2
from PIL import Image
from groq import Groq
from paddleocr import PaddleOCR

os.environ['GROQ_API_KEY'] = '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'

client = Groq()
final_output = ""

image_path = "images_test/image.png"

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contrast = cv2.equalizeHist(gray)

preprocessed_image = Image.fromarray(gray)
preprocessed_image.save('processed.jpg') 

image_path = "processed.jpg"

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

def read_text_from_image(image_path):
    try:
        result = ocr.ocr(image_path)
    
        text = ' '.join([line[1][0] for line in result[0]])  
        return text.strip()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_image(image_path):
    print(f"Extracting text from {os.path.basename(image_path)}...")
    
    extracted_text = read_text_from_image(image_path)
    return extracted_text

def process_images(path):
    global final_output
    if os.path.isfile(path):
        final_output = process_image(path)

    elif os.path.isdir(path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

        for filename in os.listdir(path):

            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(path, filename)
                extracted_text = process_image(image_path)
                
                final_output += extracted_text
       
    else:
        print(f"The path {path} is neither a valid file nor a directory.")

process_images(image_path)

completion = client.chat.completions.create(
    model="llama3-groq-70b-8192-tool-use-preview",
    messages=[
        {
            "role": "user",
            "content":  f"""
            I am providing you the extracted text from the back side of product label using OCR 
            Extract the following details from the product description below:
            - Product Name
            - Brand Name
            - Size
            - MRP (Maximum Retail Price)
            - Expiry Date (Best before)
            - Manufacturing Date (Mft dt)

            Product Description:
            {final_output}

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
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")