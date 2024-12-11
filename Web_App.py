import os
import cv2
import json
import sqlite3
import webbrowser
import numpy as np
import pandas as pd
from PIL import Image
from groq import Groq
import streamlit as st
from collections import Counter
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from inference_sdk import InferenceHTTPClient

# -------------------------- DATABASE INITIALIZATION --------------------------------

def init_product_database():
    conn = sqlite3.connect('products.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            brand_name TEXT,
            size TEXT,
            mrp TEXT,
            expiry_date TEXT,
            manufacturing_date TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def init_nos_database():
    conn = sqlite3.connect('nos_count.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            count INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def init_fruit_database():
    conn = sqlite3.connect('fruit_detections.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS fruit_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fruit_name TEXT,
            condition TEXT,
            count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_product_database()
init_nos_database()
init_fruit_database()

YT_VIDEO = ''
GITHUB_LINK = 'https://github.com/PranjalSri108/Smart_Vision_System_for_E-commerce_Quality_Control.git'

st.title("Flipkart GRID 6.0 Submission")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Product Label OCR", "NOS counting", "Freshness Detection", "Quality Inspector", "Conveyor Simulation", "Databases"])

with st.sidebar:
    st.info(f"""QUALITY CONTROL ASSISTANT 📝
            Team CIFAR version v4.2""")
    st.sidebar.image("image.jpg", use_container_width=True)

    if st.button('Working Video', use_container_width=True):
        webbrowser.open_new_tab(YT_VIDEO)

    if st.button('Github Repository', use_container_width=True):
        webbrowser.open_new_tab(GITHUB_LINK)

#----------------------------------TASK 1 + TASK 2------------------------------------------

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

with tab1:
    st.header("Product Label Information Extractor")
    st.success("""
    This tool uses Optical Character Recognition (OCR) to extract text from product labels. 
    It then employs AI to identify key product information such as name, brand, size, price, 
    and dates. This automated process significantly speeds up inventory management and 
    quality control tasks.
    """)

    input_method_ocr = st.radio("Choose input method", ("Upload Image", "Webcam"))

    if input_method_ocr == "Webcam":
        uploaded_imaeg_ocr = st.camera_input("Take a picture of the product label")
    else:
        uploaded_imaeg_ocr = st.file_uploader("Upload an image of the product label", type=["jpg", "jpeg", "png"])

    if uploaded_imaeg_ocr is not None:
        image = Image.open(uploaded_imaeg_ocr)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Product Label:")
            st.image(image, use_container_width=True)

        img_array = np.array(image)
        processed_image = process_image(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        with col2:
            with st.spinner("Performing OCR..."):
                extracted_text = read_text_from_image(processed_image)

            if extracted_text:
                with st.spinner("Extracting product information..."):
                    product_info = extract_product_info(extracted_text)

                if product_info:
                    st.subheader("Product information:")
                    info_dict = parse_product_info(product_info)
                    df = pd.DataFrame(list(info_dict.items()), columns=['Attribute', 'Value'])
                    st.table(df)

                    # Save to database
                    conn = sqlite3.connect('products.db')
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO products (product_name, brand_name, size, mrp, expiry_date, manufacturing_date)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        info_dict.get("Product Name"),
                        info_dict.get("Brand Name"),
                        info_dict.get("Size"),
                        info_dict.get("MRP"),
                        info_dict.get("Expiry Date"),
                        info_dict.get("Manufacturing Date")
                    ))
                    conn.commit()
                    conn.close()

                else:
                    st.info("Failed to extract product information. Please try again.")
            else:
                st.error("Failed to extract text from the image. Please try another image.")

#--------------------------------------TASK 3-----------------------------------------------

NOS_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dEZoySc4HAJUazGd6JIj"
)
with tab2:
    st.header("NOS Counting using IR")
    st.success("""
    We used a YOLOv11 custom-trained model to detect 100+ categories of food, beauty, drink, and medical items.
    Accurate NOS (Number of Samples) counting is crucial for inventory management. The model ensures precise
    object detection on a conveyor belt, enabling reliable counting across various product types, regardless
    of their color or transparency.
    """)

    input_method_nos = st.radio("Choose input method", ("Upload Image", "Webcam"), key="nos")

    if input_method_nos == "Webcam":
        uploaded_image = st.camera_input("Take a picture to count items")
    else:
        uploaded_image = st.file_uploader("Upload an image to count items", type=["jpg", "jpeg", "png"], key="nos_uploader")
    
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        image_height, image_width = image.shape[:2]
        
        max_dimension = 800
        scaling_factor = min(max_dimension/image_width, max_dimension/image_height)
        if scaling_factor < 1:
            new_width = int(image_width * scaling_factor)
            new_height = int(image_height * scaling_factor)
            image = cv2.resize(image, (new_width, new_height))
        
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, image)
        
        with st.spinner("Counting objects..."):
            try:
                result = NOS_CLIENT.infer(temp_image_path, model_id="object-clvwu/2")
                
                if "predictions" in result:
                    predictions = result["predictions"]
                    image_with_boxes = image.copy()
                    
                    for pred in predictions:
                        x = float(pred["x"]) * image_width
                        y = float(pred["y"]) * image_height
                        width = float(pred["width"]) * image_width
                        height = float(pred["height"]) * image_height
                        
                        x1 = max(0, int(x - width/2))
                        y1 = max(0, int(y - height/2))
                        x2 = min(image_width - 1, int(x + width/2))
                        y2 = min(image_height - 1, int(y + height/2))
                        
                        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{pred['class']} ({pred.get('confidence', 0):.2f})"
                        cv2.putText(image_with_boxes, label,
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 2)
                    
                    class_counts = Counter(pred["class"] for pred in predictions)
                    total_objects = len(predictions)
                    
                    conn = sqlite3.connect('nos_count.db')
                    c = conn.cursor()
                    for class_name, count in class_counts.items():
                        c.execute('''
                            INSERT INTO predictions (class_name, count, timestamp)
                            VALUES (?, ?, ?)
                        ''', (class_name, count, datetime.now()))
                    conn.commit()
                    conn.close()
                    
                    col1, col2 = st.columns(2)
                     
                    with col1:
                        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB),
                                caption="Detected Objects",
                                use_container_width=True)
                    
                    with col2:
                        class_counts_table = [{"Class": class_name, "Count": count} for class_name, count in sorted(class_counts.items())]
                        st.table(class_counts_table)

                else:
                    st.error("No predictions found in the model output")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# ---------------------------------------------- TASK 4 --------------------------------------------------

ROBOFLOW_TRAINED_MODEL = 'https://demo.roboflow.com/grid_6.0_dataset-qrd2r/2?publishable_key=rf_CzzWnlDIYYfT9AlYaYLXCH0D22l2'

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="poRUdDnwwYOb6TNeyxil"
)

CLASS_COLORS = {
    "apple": (0, 255, 0),
    "banana": (255, 255, 0),
    "guava": (255, 0, 255),
    "grapes": (0, 0, 255),
    "orange": (0, 165, 255),
    "pomegranate": (255, 0, 0),
    "strawberry": (255, 192, 203),
    "bitter_gourd": (128, 128, 0),
    "capsicum": (0, 128, 0),
    "tomato": (255, 99, 71),
    "rotten_apple": (128, 0, 128),
    "rotten_banana": (255, 150, 150),
    "rotten_guava": (139, 69, 19),
    "rotten_grapes": (75, 0, 130),
    "rotten_orange": (255, 140, 0),
    "rotten_pomegranate": (165, 42, 42),
    "rotten_strawberry": (255, 105, 180),
    "rotten_bittergourd": (128, 128, 128),
    "rotten_capsicum": (255, 228, 196),
    "rotten_tomato": (220, 20, 60)
}

import sqlite3
from datetime import datetime, timedelta

def insert_fruit_detection(fruit_name, condition, count):
    conn = sqlite3.connect('fruit_detections.db')
    cursor = conn.cursor()
    
    try:
        current_time = datetime.now()

        time_threshold = current_time - timedelta(seconds=30)
        
        cursor.execute('''
            SELECT * FROM fruit_detections
            WHERE fruit_name = ? AND condition = ? AND timestamp >= ?
        ''', (fruit_name, condition, time_threshold))
        
        existing_entry = cursor.fetchone()
        
        if existing_entry:
            print(f"Skipping duplicate entry for {fruit_name} ({condition}) within the last 30 seconds.")
        else:
            cursor.execute('''
                INSERT INTO fruit_detections (fruit_name, condition, count, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (fruit_name, condition, count, current_time))
            conn.commit()
            print(f"Inserted new entry for {fruit_name} ({condition}).")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        conn.close()

with tab3:
   st.header("Fruit and Vegetable Freshness Detection")
   st.success("""
   Our AI-powered freshness detection system can identify and classify various fruits and 
   vegetables, distinguishing between fresh and rotten produce. This technology helps 
   maintain quality standards, reduce waste, and ensure only the freshest products reach 
   customers. The system can detect subtle signs of spoilage that might be missed by 
   human inspectors.
   """)

   if st.button('USE LIVE VIDEO FEED'):
        webbrowser.open_new_tab(ROBOFLOW_TRAINED_MODEL)

   input_method_fresh = st.radio("Choose input method", ("Upload Image", "Webcam"), key="fresh")

   if input_method_fresh == "Webcam":
        uploaded_image = st.camera_input("Take a picture to count items")
   else:
        uploaded_image = st.file_uploader("Upload an image to count items", type=["jpg", "jpeg", "png"], key="fresh_uploader")

   if uploaded_image is not None:
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))

    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image)
    
    with st.spinner("Detection freshness of produce:"):
        result = CLIENT.infer(temp_image_path, model_id="grid_6.0_dataset-qrd2r/2")

    if "predictions" in result:
        predictions = result["predictions"]
        fruit_counts = Counter(pred['class'] for pred in predictions)

        for fruit, count in fruit_counts.items():
            condition = "rotten" if "rotten" in fruit else "fresh"
            fruit_name = fruit.replace("rotten_", "")
            insert_fruit_detection(fruit_name, condition, count)
        
        for prediction in predictions:
            class_name = prediction['class']
            confidence = prediction['confidence']
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']

            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            color = CLASS_COLORS.get(class_name, (255, 0, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_bg_start = (x1, y1 - text_size[1] - 10)
            label_bg_end = (x1 + text_size[0], y1)
    
            cv2.rectangle(image, label_bg_start, label_bg_end, color, cv2.FILLED)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        st.image(image, channels="BGR", use_container_width=True)
    else:
        st.write("No predictions found.")

#-----------------------------------------BONUS TASK---------------------------------------------------

INSPECTION_TRAINED_MODEL = 'https://demo.roboflow.com/bottle_can_quality/1?publishable_key=rf_UvXf0JevdVfcHVV4hMA2XtK8BLq1'

BOTTLE_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ze2R08zoIqYfFDzJ66lH"
)

with tab4:
  st.header("Bottle and Can Quality Inspector")
  st.success("""
  This bonus task demonstrates our approach to structural quality checks for products. 
  The system analyzes bottle images to detect damages, assess liquid levels, and verify 
  label integrity. In cans it detects deformation, fissure or open-can defects This automated inspection process ensures consistency in quality control, 
  reduces human error, and can significantly speed up the production line while maintaining 
  high quality standards.
  """)

  if st.button('USE YOUR WEB-CAM'):
      webbrowser.open_new_tab(INSPECTION_TRAINED_MODEL)

  uploaded_file = st.file_uploader("Choose an image of a Bottle...", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    height, width = image.shape[:2]

    image = cv2.resize(image, (int(width * 0.1), int(height * 0.1)))

    cv2.imwrite("uploaded_image.jpg", image)

    with st.spinner("Inspecting Quality:"):
        result = BOTTLE_CLIENT.infer("uploaded_image.jpg", model_id="bottle_quality/4")

    predictions = result.get('predictions', [])

    point_liq = (0,0)
    point_bottle = (0,0)
    liq_height = 0

    colors = {
        'Bottle': (255, 0, 0),         
        'Damaged_Bottle': (255, 150, 0),
        'Liquid_Level': (0, 255, 0),   
        'Proper_Label': (255, 255, 0),   
        'Damaged_Label': (0, 0, 255) 
    }

    for prediction in predictions:
        class_name = prediction['class']  
        color = colors.get(class_name, (0, 255, 255)) 

        x_min = int(prediction['x'] - prediction['width'] / 2)
        y_min = int(prediction['y'] - prediction['height'] / 2)
        x_max = int(prediction['x'] + prediction['width'] / 2)
        y_max = int(prediction['y'] + prediction['height'] / 2)

        if class_name == "Bottle" or class_name == "Damaged_Bottle":
            point_bottle = (int((x_min + x_max) / 2), y_max)
            cv2.circle(image, point_bottle, radius=1, color=(255, 0, 255),thickness=2)

        if class_name == "Liquid_Level":
            point_liq = (int((x_min + x_max) / 2), (y_max + y_min)//2)
            cv2.circle(image, point_liq, radius=1, color=(255, 0, 255),thickness=2)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    cv2.line(image, point_liq, point_bottle, color=(255, 0, 255), thickness=1)
    liq_height = point_bottle[1] - point_liq[1] 

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quality Check:")
    with col2:
        st.subheader("Inference:")

    col1, col2, col3 = st.columns([3, 1, 2])

    with col1:
        st.image(image_rgb, use_container_width=True)

    with col2:
        for prediction in predictions:
            class_name = prediction['class']

            if class_name == "Bottle": st.info("BOTTLE:")
            if class_name == "Damaged_Bottle": st.info("BOTTLE:")
            if class_name == "Liquid_Level": st.info("LIQ LEVEL:")
            if class_name == "Proper_Label": st.info("LABEL:")
            if class_name == "Damaged_Label": st.info("LABEL:")

    with col3:
        for prediction in predictions:
            class_name = prediction['class']

            if class_name == "Bottle": st.success("Correct Shape")
            if class_name == "Damaged_Bottle": st.error("Damaged Shape")
            if ((class_name == "Liquid_Level") and liq_height > 30): st.success("Correct")
            if ((class_name == "Liquid_Level") and liq_height < 30): st.error("Wrong")
            if class_name == "Proper_Label": st.success("Proper Label")
            if class_name == "Damaged_Label": st.error("Damaged")
  

  uploaded_file_can = st.file_uploader("Choose an image of a Can....", type=["jpg", "jpeg", "png"])

  if uploaded_file_can is not None:
    file_bytes = np.asarray(bytearray(uploaded_file_can.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    height, width = image.shape[:2]

    image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))

    cv2.imwrite("uploaded_image.jpg", image)

    with st.spinner("Inspecting Quality:"):
        result = BOTTLE_CLIENT.infer("uploaded_image.jpg", model_id="bottle_can_quality/1")

    predictions = result.get('predictions', [])

    point_liq = (0,0)
    point_bottle = (0,0)
    liq_height = 0

    colors = {
        'Perfect': (0, 255, 0),         
        'Open-can': (255, 255, 0),
        'Fissure': (0, 0, 255),   
        'Deformation': (255, 0, 255),   
    }

    for prediction in predictions:
        class_name = prediction['class']  
        color = colors.get(class_name, (0, 255, 255)) 

        x_min = int(prediction['x'] - prediction['width'] / 2)
        y_min = int(prediction['y'] - prediction['height'] / 2)
        x_max = int(prediction['x'] + prediction['width'] / 2)
        y_max = int(prediction['y'] + prediction['height'] / 2)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 6)

        (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=4)
    
        text_offset_x, text_offset_y = x_min, y_min - text_height - 30
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y + text_height))
        cv2.rectangle(image, box_coords[0], box_coords[1], color=colors.get(class_name), thickness=cv2.FILLED) 
        
        cv2.putText(image, class_name, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0), thickness=4)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quality Check:")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Inference:")
        for prediction in predictions:
            class_name = prediction['class']

            if class_name == "Perfect": st.success("Perfect: Ready for supply")
            if class_name == "Open-can": st.error("Not Approved: Open Can")
            if class_name == "Fissure": st.success("Not Approved: Fissure")
            if class_name == "Deformation": st.error("Not Approved: Deformation")

#------------------------------------------BONUS TASK-----------------------------------------------

IMAGE_PATH = 'Lighting-2.jpg'
HEAT_PATH = 'heatma.jpg'

with tab5:
    st.header("Simulation of Conveyor Belt")

    st.subheader("Smart Lighting System:")
    st.write(f'''To enhance the accuracy of defect detection and quality assessment in our 
             smart vision system, we propose the implementation of a unique, controlled 
             lighting setup tailored specifically for hyperspectral imaging. Proper illumination 
             is critical to minimize shadows, reflections, and noise, ensuring high-quality image 
             acquisition across various lighting conditions. Our proposed lighting system includes:''')
    
    col1, col2 = st.columns([1,1])

    with col1:
        st.image(IMAGE_PATH, channels="BGR", use_container_width=True)
        st.image(HEAT_PATH, channels="BGR", use_container_width=True)

    with col2: 
        st.info(f'''Here's a more concise points of the idea:

1. *Uniform Illumination*: High-intensity LED lights on both sides of the conveyor belt provide even lighting to minimize shadows, enhancing hyperspectral camera accuracy.

2. *Wavelength-Specific Lights*: A mix of lights emitting at different wavelengths ensures the hyperspectral camera captures detailed spectral data for precise defect and quality analysis.

3. *Automated Lighting Adjustment*: Sensors will monitor lighting conditions and automatically adjust light intensity and wavelength in real-time for optimal imaging.''')
        
    st.subheader("Robotic Arm Integration with Mounted Camera:")

    st.write(f"""We have also designed an AUTOMATED SEGREGATOR using ROBOTIC ARM equipped with a CAMERA, 
               the RED blocks here represent the faulty/damaged/rotten items which gets detected by the camera and 
               then gets separated from the fresh batch.\n The video shown here is a GAZEBO stimulation using UR5 
               Robotic arm which we prepared using ROS Noetic""")

    video_file = open('gazebo_arm.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

# ---------------------------------- VIEW DATABASES --------------------------------------

with tab6:
    st.header("Database Viewer")
    
    # Options to choose which database to view
    db_choice = st.radio("Choose a database to view:", ("Products", "NOS Counter", "Fruit Detections"))
    
    def view_products_db():
        conn = sqlite3.connect('products.db')
        df = pd.read_sql_query("SELECT * FROM products", conn)
        conn.close()
        return df

    def view_nos_counter_db():
        conn = sqlite3.connect('nos_count.db')
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        return df

    def view_fruit_detections_db():
        conn = sqlite3.connect('fruit_detections.db')
        df = pd.read_sql_query("SELECT * FROM fruit_detections", conn)
        conn.close()
        return df

    if db_choice == "Products":
        st.subheader("Products Database")
        try:
            df_products = view_products_db()
            if not df_products.empty:
                st.table(df_products)
            else:
                st.info("No records found in the Products database.")
        except Exception as e:
            st.error(f"Error loading Products database: {e}")

    elif db_choice == "NOS Counter":
        st.subheader("NOS Counter Database")
        try:
            df_nos = view_nos_counter_db()
            if not df_nos.empty:
                st.table(df_nos)
            else:
                st.info("No records found in the Predictions database.")
        except Exception as e:
            st.error(f"Error loading Predictions database: {e}")

    elif db_choice == "Fruit Detections":
        st.subheader("Fruit Detections Database")
        try:
            df_fruits = view_fruit_detections_db()
            if not df_fruits.empty:
                st.table(df_fruits)
            else:
                st.info("No records found in the Fruit Detections database.")
        except Exception as e:
            st.error(f"Error loading Fruit Detections database: {e}")
