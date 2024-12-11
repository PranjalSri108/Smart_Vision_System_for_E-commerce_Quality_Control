# Quality Control Assistant 📝
### Flipkart GRID 6.0 Submission | Team CIFAR v4.2

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7+-yellow.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![SQLite3](https://img.shields.io/badge/SQLite3-Built--in-blue.svg)](https://www.sqlite.org)

A comprehensive quality control system that combines OCR, computer vision, and AI to automate product inspection and quality assurance processes and maintains updated databases of each operation. 

### DEPLOYED APP: [streamlit/applink](https://team-cifar-flipartgrid-submission.streamlit.app/)
### WORKING VIDEO: [youtube/videolink](https://youtu.be/yBr6TauzZuc?si=brlCcT_e9grymsqJ)

### Main file of code: Web_App.py

## 🚀 Features

### 1. Product Label Information Extractor
- Utilizes PaddleOCR for text extraction from product labels
- AI-powered information parsing using Groq LLM
- Extracts key details:
  - Product Name
  - Brand Name
  - Size
  - MRP
  - Expiry Date
  - Manufacturing Date
- **Data is automatically saved into a SQLite database (`products.db`).**
- Supports both image upload and webcam input

### 2. NOS (Number of Samples) Counter
- Image Recognition-based object counting
- Yolov11 custom trained model on 100+ product categories
- Real-time counting on conveyor belts
- **Counts are saved into a SQLite database (`nos_count.db`).**
- Color and transparency independent detection

### 3. Freshness Detection System
- AI-powered detection of fresh and rotten produce
- Supports multiple categories:
  - Fruits: Apple, Banana, Guava, Grapes, Orange, Pomegranate, Strawberry
  - Vegetables: Bitter Gourd, Capsicum, Tomato
- Real-time classification with bounding boxes
- Confidence score display
- **Detections are saved into a SQLite database (`fruit_detections.db`).**

### 4. Quality Inspector
- Automated inspection system for bottles and cans
- Detects multiple defect types:
  - Bottles:
    - Structural damage
    - Liquid level verification
    - Label integrity check
  - Cans:
    - Deformation
    - Fissures
    - Open-can defects

### 5. Database Viewer
- Real-time updated database view option in specific format
- Product Label OCR Database:
  | ID no.         | Product Name    | Brand Name     | Size       | MRP      | Expiry Date    | Manufacturing Date  | TIME STAMP     |
  |----------------|-----------------|----------------|------------|----------|----------------|---------------------|----------------|
  | ....           | ....            | ....           | ....       | ....     | ....           | ....                | ....           |

- Brand Detection and NOS Counter Database:
  | ID no.         | Class   Name    | NOS count      | TIME STAMP     |
  |----------------|-----------------|----------------|----------------|
  | ....           | ....            | ....           | ....           |

- Freshness Detection Database:
  | ID no.         | Fruit/Vegetable   Name    | Condition      | NOS count      |TIME STAMP     |
  |----------------|---------------------------|----------------|----------------|---------------|
  | ....           | ....                      | ....           | ....           | ...           |

### 6. Conveyor Belt Simulation
- Smart lighting system simulation
- Robotic arm integration with mounted camera
- Automated segregation system
- Gazebo simulation using UR5 Robotic arm

## 🛠️ Technologies Used
- Streamlit - Web application framework
- OpenCV - Image processing
- PaddleOCR - Optical Character Recognition
- Groq - Large Language Model API
- SQLite3 - Database for storing extracted data and detections
- Roboflow - Object detection models
- ROS Noetic - Robotic arm simulation
- Gazebo - 3D robotics simulator

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/PranjalSri108/Smart_Vision_System_for_E-commerce_Quality_Control.git
cd Smart_Vision_System_for_E-commerce_Quality_Control
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Web_App.py
```
4. To view the Databases:
```bash
python3 sql_data.py
```
Then choose the databae to view

## 🔑 API Keys Required
- Groq API Key
- Roboflow API Key

## 📁 Project Structure
```
quality-control-assistant/
├── Web_App.py
├── requirements.txt
├── image.jpg
├── Lighting-2.jpg
├── heatma.jpg
├── gazebo_arm.mp4
└── README.md
```

## 🤝 Team Members
- Pranjal Srivastav (Team Lead)
- Aryan Chaudhary

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
