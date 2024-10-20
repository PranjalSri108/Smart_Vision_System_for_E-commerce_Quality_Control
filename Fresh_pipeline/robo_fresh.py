import cv2
from inference_sdk import InferenceHTTPClient

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

image_path = "test-5.jpg"  # Replace with the path to your image
result = CLIENT.infer(image_path, model_id="grid_6.0_dataset-qrd2r/2")

if "predictions" in result:
    predictions = result["predictions"]
    image = cv2.imread(image_path)

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
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence*100:.2f}%"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No predictions found.")
