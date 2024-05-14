import os
import torch
import pandas as pd
from PIL import Image
import numpy as np

from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Load the YOLOs model
image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

# Define the directory containing the dataset
dataset_dir = "../compressed_dataset"

# Define the path to the CSV file
csv_file_path = "car_or_road_results.csv"

# Check if the CSV file already exists
if not os.path.exists(csv_file_path):
    # Create a new CSV file with headers
    with open(csv_file_path, "w") as file:
        file.write("image_name,has_car_or_road\n")

# Iterate through the folders in the dataset directory
i = 0
for country_folder in os.listdir(dataset_dir):
    country_folder_path = os.path.join(dataset_dir, country_folder)
    
    
    images = np.array([])
    image_paths = []
    n = 0
    # Iterate through the images in the country folder
    for image_file in os.listdir(country_folder_path):

        image_path = os.path.join(country_folder_path, image_file)
        if image_path in open(csv_file_path).read():
            continue
        else:
            print(image_path)
        # Load the image
        image = Image.open(image_path)
        images = np.append(images, image)
        image_paths.append(image_path)
        n += 1
        
        if n == 5:
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
        
            # Preprocess the image
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
        
            # Post-process the object detection results
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)
            
            for image_path, r in zip(image_paths, results):
                # Check if 'car' or 'road' is in the detected COCO classes
                car_or_road = False
                for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
                    name = model.config.id2label[label.item()]
                    if name == "car" or name == "bus" or name == "motorcycle" or name == "truck" or name == "road" or name == "highway":
                        car_or_road = True
                        break
                # Write the result to the CSV file
                with open(csv_file_path, "a") as file:
                    file.write(f"{image_path},{int(car_or_road)}\n")
            n = 0
            images = np.array([])
            image_paths = []
            
    if len(image_paths) > 0:
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process the object detection results
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)

        for image_path, r in zip(image_paths, results):
            # Check if 'car' or 'road' is in the detected COCO classes
            car_or_road = False
            for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
                name = model.config.id2label[label.item()]
                if name == "car" or name == "bus" or name == "motorcycle" or name == "truck" or name == "road" or name == "highway":
                    car_or_road = True
                    break
                    # Write the result to the CSV file
            with open(csv_file_path, "a") as file:
                file.write(f"{image_path},{int(car_or_road)}\n")
        n = 0
        images = np.array([])
        image_paths = []

print("Results saved to:", csv_file_path)
