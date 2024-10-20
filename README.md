# Quality Control Assistant 📝
### Flipkart GRID 6.0 Submission | Team CIFAR v4.2

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7+-yellow.svg)](https://github.com/PaddlePaddle/PaddleOCR)

A comprehensive quality control system that combines OCR, computer vision, and AI to automate product inspection and quality assurance processes.

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
- Supports both image upload and webcam input

### 2. NOS (Number of Samples) Counter
- Infrared technology-based object counting
- Real-time counting on conveyor belts
- Color and transparency independent detection

### 3. Freshness Detection System
- AI-powered detection of fresh and rotten produce
- Supports multiple categories:
  - Fruits: Apple, Banana, Guava, Grapes, Orange, Pomegranate, Strawberry
  - Vegetables: Bitter Gourd, Capsicum, Tomato
- Real-time classification with bounding boxes
- Confidence score display

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

### 5. Conveyor Belt Simulation
- Smart lighting system simulation
- Robotic arm integration with mounted camera
- Automated segregation system
- Gazebo simulation using UR5 Robotic arm

## 🛠️ Technologies Used
- Streamlit - Web application framework
- OpenCV - Image processing
- PaddleOCR - Optical Character Recognition
- Groq - Large Language Model API
- Roboflow - Object detection models
- ROS Noetic - Robotic arm simulation
- Gazebo - 3D robotics simulator

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quality-control-assistant.git
cd quality-control-assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY='your_groq_api_key'
```

4. Run the application:
```bash
streamlit run Web_App.py
```

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
- [Team Member 1]
- [Team Member 2]
- [Team Member 3]

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Flipkart GRID 6.0 for the opportunity
- PaddleOCR community
- Roboflow for object detection models
- Groq for LLM API access
