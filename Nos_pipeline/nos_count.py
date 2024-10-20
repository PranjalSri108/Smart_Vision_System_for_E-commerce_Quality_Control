import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="poRUdDnwwYOb6TNeyxil"
)

image_path = "test-5.jpg" 
result = CLIENT.infer(image_path, model_id="grid_6.0_dataset-qrd2r/2")

if "predictions" in result:
    predictions = result["predictions"] 
    object_counts = {}

    for prediction in predictions:
        class_name = prediction['class']
        
        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

    print("Object Counts:")
    for obj_class, count in object_counts.items():
        print(f"{obj_class}: {count}")

else:
    print("No predictions found.")
