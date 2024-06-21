import os
import torch
from PIL import Image

# Loading the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Paths for folders
image_dir = 'images'
fixed_image_dir = 'cropped_resized_images'  # New folder name for cropped and resized images

# Creating the folder if it doesn't exist for cropped and resized images
if not os.path.exists(fixed_image_dir):
    os.makedirs(fixed_image_dir)

# Get all image files in the 'images' folder
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    # Open the image
    with Image.open(image_path) as img:
        results = model(img)
        detections = results.pandas().xyxy[0]

        if not detections.empty:
            # Get the first detection
            xmin, ymin, xmax, ymax = int(detections.iloc[0]['xmin']), int(detections.iloc[0]['ymin']), int(detections.iloc[0]['xmax']), int(detections.iloc[0]['ymax'])

            # Crop the object
            cropped_image = img.crop((xmin, ymin, xmax, ymax))

            # Resize and save
            cropped_image.thumbnail((800, 800))
            fixed_image_path = os.path.join(fixed_image_dir, image_file)
            cropped_image.save(fixed_image_path)
            print(f"Resized and saved: {fixed_image_path}")
        else:
            # If no object is detected, resize and save the original image
            img.thumbnail((800, 800))
            fixed_image_path = os.path.join(fixed_image_dir, image_file)
            img.save(fixed_image_path)
            print(f"Resized and saved: {fixed_image_path}")

print("Resize and save operation completed.")
