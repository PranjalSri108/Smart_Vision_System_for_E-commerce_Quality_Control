import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import json

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ze2R08zoIqYfFDzJ66lH"
)

result = CLIENT.infer("test.jpg", model_id="bottle_quality/4")

print(json.dumps(result, indent=4))

predictions = result.get('predictions', [])

colors = {
    'Bottle': (255, 0, 0),         # Blue
    'Liquid_Level': (0, 255, 0),   # Green
    'Proper_Label': (0, 0, 255),   # Red
    'Damaged_Label': (255, 255, 0) # Cyan
}

image = cv2.imread("test.jpg")

for prediction in predictions:
    class_name = prediction['class']  
    color = colors.get(class_name, (0, 255, 255)) 

    x_min = int(prediction['x'] - prediction['width'] / 2)
    y_min = int(prediction['y'] - prediction['height'] / 2)
    x_max = int(prediction['x'] + prediction['width'] / 2)
    y_max = int(prediction['y'] + prediction['height'] / 2)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)

    text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
    label_bg_start = (x_min, y_min - text_size[1] - 10)
    label_bg_end = (x_min + text_size[0], y_min)
    
    cv2.rectangle(image, label_bg_start, label_bg_end, color, cv2.FILLED)

    cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

cv2.imwrite("bounding_box_result.png", image)
