import cv2
import numpy as np
import pandas as pd
import os

# CSV file containing images and times
clock_images_path = 'images_cropped_fixed/'  # Directory containing cropped analog clock images
csv_file = './pre_calculated/label.csv'  # CSV file containing clock times

# Load time data from CSV file
time_data = pd.read_csv(csv_file, sep=';')

# SIFT (Scale-Invariant Feature Transform) feature detector in OpenCV
sift = cv2.SIFT_create()

# Lists to store keypoints, descriptors, and times from images
clock_keypoints = []
clock_descriptors = []
clock_times = []

# Collecting features and times for each image
for index, row in time_data.iterrows():
    img_path = clock_images_path + row['ImageFileName']
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    clock_keypoints.append(kp)
    clock_descriptors.append(des)
    clock_times.append(row['Time'])

# Predicting the time for all images in the matches folder
matches_folder_path = 'matches/'
match_files = [f for f in os.listdir(matches_folder_path) if os.path.isfile(os.path.join(matches_folder_path, f))]

# Using Brute Force Matcher to find best matches
bf = cv2.BFMatcher()

for match_file in match_files:
    input_image_path = os.path.join(matches_folder_path, match_file)
    input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if input_img is None:
        print(f"File could not be read: {match_file}")
        continue

    # Detecting features in the input image
    kp_frame, des_frame = sift.detectAndCompute(input_img, None)

    best_match = None
    best_match_count = 0
    best_good_matches = []  # To store the best matches

    for i in range(len(clock_descriptors)):
        if clock_descriptors[i] is None:
            continue  # Skip if no descriptors found
        matches = bf.knnMatch(clock_descriptors[i], des_frame, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)
        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_match = i
            best_good_matches = good_matches  # Store the best matches

    if best_match is not None:
        # Found the best matching analog clock image
        best_img = cv2.imread(clock_images_path + time_data['ImageFileName'][best_match], cv2.IMREAD_GRAYSCALE)

        # Calculate perspective transformation matrix
        kp_img = clock_keypoints[best_match]

        if len(best_good_matches) < 4:
            print(f"{match_file}: Not enough good matches found!")
        else:
            src_pts = np.float32([kp_img[m.queryIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Print the predicted time in the terminal
            print(f"{match_file}: Predicted time: {clock_times[best_match]}")
    else:
        print(f"{match_file}: No match found!")
