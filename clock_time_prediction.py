import cv2
import numpy as np
import os
import math
import glob
import csv

def detect_time(image_path, csv_file):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Image {image_path} not found or could not be opened.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Parameters for circle detection
    min_radius = 20
    max_radius = 200
    param1 = 50
    param2 = 30
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    if circles is None:
        print(f"Error: No circle detected in {image_path}.")
        return
    
    circles = np.round(circles[0, :]).astype("int")
    
    # Draw detected circles
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    center_x, center_y, radius = circles[0]
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Parameters for line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        print(f"Error: No lines detected in {image_path}.")
        return
    
    close_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate distance from line endpoints to circle center
            dist1 = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)
            dist2 = np.sqrt((center_x - x2)**2 + (center_y - y2)**2)
            
            # Check if either endpoint is close enough to the circle or if the line segment intersects the circle
            if (dist1 < radius * 1.5 or dist2 < radius * 1.5) or ((dist1 < radius * 2.0 and dist2 < radius * 2.0) and (dist1 > radius * 0.5 and dist2 > radius * 0.5)):
                close_lines.append((x1, y1, x2, y2))
    
    if len(close_lines) < 2:
        print(f"Error: Not enough close lines found in {image_path}.")
        return
    
    # Assuming two lines found, calculate angles for hour and minute hands
    angle1 = calculate_angle(*close_lines[0], center_x, center_y)
    angle2 = calculate_angle(*close_lines[1], center_x, center_y)
    
    # Determine which angle corresponds to hour and minute hands
    hour_angle = min(angle1, angle2)
    minute_angle = max(angle1, angle2)
    
    # Convert angles to time
    hour = int(hour_angle // 30) % 12
    minute = int(minute_angle // 6) % 60
    
    detected_time = "{:02d}:{:02d}".format(hour, minute)
    
    print(f"Detected Time in {image_path}: {detected_time}")
    
    # Write to CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([image_path, detected_time])

def calculate_angle(x1, y1, x2, y2, cx, cy):
    dx1, dy1 = x1 - cx, y1 - cy
    dx2, dy2 = x2 - cx, y2 - cy
    angle = math.degrees(math.atan2(dy1, dx1) - math.atan2(dy2, dx2))
    if angle < 0:
        angle += 360
    return angle

# Folder containing images
folder_path = 'images_cropped_fixed'  # Update with your folder path

# CSV file to store predictions
csv_file = 'predictions.csv'

# Check if CSV file exists and write header if it's a new file
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image_Path', 'Detected_Time'])

# Iterate through images sequentially from 0.jpg to 102.jpg
for i in range(103):  # Adjust range to match the number of images you have
    image_file = os.path.join(folder_path, f"{i}.jpg")
    if os.path.isfile(image_file):
        detect_time(image_file, csv_file)
