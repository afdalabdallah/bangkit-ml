from flask import Flask, request, jsonify
import subprocess
import cv2
import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

def detect(image_path, image_name):
    detection_command = f"python ./yolov7/detect.py --weights ./yolov7/best.pt --conf 0.1 --source {image_path} --name {image_name}"
    result = subprocess.run(detection_command, shell=True, text=True, capture_output=True)

def predict_and_display(image_path, model):
    img_array = preprocess_image(image_path)
    CATEGORIES = ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return CATEGORIES[predicted_class]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    detect(image_path, image_name)
    json_path = f'./runs/detect/{image_name}/result.json'

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    image_segmented_path = data['path']
    image = cv2.imread(image_path)
    highest_confidence_per_label = {}
    for prediction in data['prediction']:
        bounding_box = prediction['bounding_box']
        label = prediction['label']
        confidence = prediction['confidence']

        bounding_box = [int(coord) for coord in bounding_box]
        if label not in highest_confidence_per_label or confidence > highest_confidence_per_label[label]['confidence']:
            highest_confidence_per_label[label] = {
                'bounding_box': bounding_box,
                'label': label,
                'confidence': confidence
            }
    
    result_prediction = []
    pred = {
            image_name:{
                
            }
            ,
            "path": image_segmented_path
        }
    for label, highest_confidence_prediction in highest_confidence_per_label.items():
        bounding_box = highest_confidence_prediction['bounding_box']
        label = highest_confidence_prediction['label']
        confidence = highest_confidence_prediction['confidence']

        cropped_roi = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        os.makedirs(f'./cropped/{image_name}/', exist_ok=True)
        save_path = f'./cropped/{image_name}/cropped_{label}_{confidence:.2f}.jpg'
        cv2.imwrite(save_path, cropped_roi)

        cropped_folder = f"./cropped/{image_name}"
        color_model = load_model('./color_classification_cnn_model.h5')
    
        filename = f'cropped_{label}_{confidence:.2f}.jpg'
        file_path = os.path.join(cropped_folder, filename)
        categories = predict_and_display(file_path, color_model)
        convert_putih = ["Pink","Cream","Gray","Red","Yellow"]
        convert_brown = ["Purple","Orange","Green","Blue"]
        if label == "skin":
            if categories in convert_putih:
                categories = "White"
            elif categories in convert_brown:
                categories = "Brown"

        pred[image_name][label] = categories
    result_prediction.append(pred)

    return result_prediction

@app.route('/process_image', methods=['POST'])
def process_image_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    image_path = '/result/'
    image_file.save(image_path)

    result = process_image(image_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
