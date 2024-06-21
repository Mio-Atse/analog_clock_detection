# Analog Clock Time Estimation via Computer Vision
## Overview
This repository contains algorithms for estimating time on analog clocks using two approaches: brute force feature extraction and geometric approach algorithms. Additionally, it includes scripts for object detection within a dataset to isolate clock images and resize them accordingly.

## !!Check Pre-calculated Times
- The output and error margins obtained by using `clock_time_detection.py` for the specified data set are available in a csv file in the folder named `pre-calculated`. 
## Installation

1. **Python and Library Installation**
   - Ensure Python 3.x and pip are installed.
   - Install required Python libraries:
     ```
     pip install torch torchvision opencv-python pillow pandas csv
     ```
   - Install YOLOv5 model:
     ```
     pip install 'git+https://github.com/ultralytics/yolov5.git'
     ```

## Starting

### Dataset Preparation
- Download the dataset from [this link](https://www.kaggle.com/datasets/vctorsuarezvara/real-images-of-analogclocks).
- Place the `label.csv` file under the analog_clock_detection folder

### Object Detection and Resizing

**Running the Object Detection and Resize Script**
   - Place your dataset in a folder named `images` within the directory where the script is located.
   - Example file structure: `./analog_clock_detection/images/0.jpg, 1.jpg, 2.jpg, ...`
   - Run the `detect_clock_resize.py` script.
   - It processes images in the `images` folder (PNG, JPG, JPEG).

**Output**
   - Processed images are saved in the `imaged_cropped_fixed` folder.

**Notes**
   - If no objects are detected, the original image is resized and saved.

## Analog Clock Time Prediction using SIFT and Homography

This code contains Python code to predict the time shown on analog clocks in input images using computer vision techniques.

### Overview

The project utilizes OpenCV (Open Source Computer Vision Library) and pandas for handling image processing and data operations respectively. The main steps involve:
- Extracting SIFT (Scale-Invariant Feature Transform) features from a dataset of cropped analog clock images.
- Matching these features with those extracted from input images to identify the best match.
- Using homography to calculate perspective transformation and predict the time shown on the analog clock in the input image.

### Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- pandas

### Usage

1. **Setup:**
   - Ensure Python and necessary libraries (OpenCV, NumPy, pandas) are installed.

2. **Dataset Preparation:**
   - Save the images you want to estimate the time in a folder named `matches` that you created under the `analog_clock_detection` folder.
   - Ensure CSV file (`label.csv`) listing each image file name along with the corresponding time displayed on the clock.

3. **Running the Code:**
   - Run the Python script (`feature_ext_prediction.py`).
   - The script will process each image in the `matches/` directory, attempting to predict the time shown on the analog clock in each image.

4. **Output:**
   - The predicted time for each input image is displayed in the terminal.

### File Descriptions

- `feature_ext_prediction.py`: Main script that performs time prediction based on feature matching and homography.
- `label.csv`: CSV file containing the mapping of image file names to their respective times.
- `images_cropped_fixed/`: Directory containing cropped analog clock images used for training.
- `matches/`: Directory containing input images for which the time needs to be predicted.

### Notes

- Ensure sufficient feature matches are found (`good_matches`) for accurate time prediction.
- Adjust parameters such as the distance threshold in feature matching (`0.5 * n.distance`) and RANSAC threshold in homography calculation (`5.0`) based on specific image characteristics.


## Analog Clock Time Detection using Hough Transform and Line Detection

This code contains Python code to detect and predict the time shown on analog clocks in input images using computer vision techniques.

### Overview

The project utilizes OpenCV (Open Source Computer Vision Library) for circle and line detection:
- **Circle Detection:** Uses Hough Circle Transform to detect the circular shape of the clock face.
- **Line Detection:** Applies Canny edge detection followed by Hough Line Transform to detect hour and minute hands.

The angles between these hands are calculated to predict the time shown on the clock.

### Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- csv (for CSV operations)

### Usage

1. **Setup:**
   - Ensure Python and necessary libraries (OpenCV, NumPy) are installed.

2. **Running the Code:**
   - Run the Python script (`clock_time_detection.py`).
   - The script will sequentially process each image (`0.jpg` to `102.jpg` by default) and detect the time shown on the clock.

3. **Output:**
   - The detected time for each image is printed in the terminal.
   - Results are appended to `predictions.csv`, which contains the path of each image and its corresponding detected time.
   - Compare the times in the label.csv file with the output you receive.

### File Descriptions

- `clock_time_detection.py`: Main script that performs time detection based on circle and line detection.
- `predictions.csv`: CSV file containing the mapping of image paths to their detected times.
- `images_cropped_fixed/`: Directory containing cropped analog clock images used for detection.

### Notes

- Ensure sufficient and clear images of analog clocks are provided for accurate detection.
- Fine-tune parameters such as circle detection thresholds (`param1`, `param2`) and line detection parameters (`minLineLength`, `maxLineGap`) based on image characteristics and quality.
