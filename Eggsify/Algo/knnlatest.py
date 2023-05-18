import os
import cv2
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path


# Define the directories for training and testing data
data_dir = Path('D:\Documents\Modules SY. 2022-2023\CMSC 502 - Undergrad Thesis 2\Datasets')

def detect_eggs(image):
    image_copy = image.copy()
    
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    ret, thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = {}
    egg_contours = []
    egg_bounding_rects = []
    for i in range(len(contours)):
        cnt = contours[i]
        ar = cv2.contourArea(cnt)
        if(ar > 600 and ar < 1000):
            x, y, w, h = cv2.boundingRect(cnt)
            if(not h > 40 and not w > 40 and not w < 20 and not h < 20 and not x > 530):
                egg_contours.append(cnt)
                egg_bounding_rects.append((x, y, w, h))

    # Draw contours and bounding rectangles on the image
    result = image.copy()
    cv2.drawContours(result, egg_contours, -1, (0, 255, 0), 2)
    for rect in egg_bounding_rects:
        cv2.rectangle(result, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)

    # Show the output image
    cv2.imshow("res", result)
    cv2.waitKey(0)

    return egg_bounding_rects


def detect_egg_color(image, egg_bbox):
    # Get the region of interest (ROI) for the egg
    # means none
    if(image.shape == (0, 0, 3) or len(image) == 0):
        return 'invalid'
    
    # Convert the egg ROI to HSV color space
    hsv_roi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for the egg colors
    brown_lower = np.array([0, 10, 10], dtype=np.uint8)
    brown_upper = np.array([20, 255, 255], dtype=np.uint8)
    dark_brown_lower = np.array([20, 10, 10], dtype=np.uint8)
    dark_brown_upper = np.array([40, 255, 255], dtype=np.uint8)
    gray_lower = np.array([0, 0, 50], dtype=np.uint8)
    gray_upper = np.array([180, 50, 255], dtype=np.uint8)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 255, 255], dtype=np.uint8)
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
    
    # Threshold the egg ROI to get a binary mask for each color range
    brown_mask = cv2.inRange(hsv_roi, brown_lower, brown_upper)
    dark_brown_mask = cv2.inRange(hsv_roi, dark_brown_lower, dark_brown_upper)
    gray_mask = cv2.inRange(hsv_roi, gray_lower, gray_upper)
    white_mask = cv2.inRange(hsv_roi, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
    
    # Compute the number of pixels in each binary mask
    num_brown_pixels = np.sum(brown_mask == 255)
    num_dark_brown_pixels = np.sum(dark_brown_mask == 255)
    num_gray_pixels = np.sum(gray_mask == 255)
    num_white_pixels = np.sum(white_mask == 255)
    num_yellow_pixels = np.sum(yellow_mask == 255)
    
    # Compute the fraction of pixels in each binary mask
    total_pixels = image.shape[0] * image.shape[1]
    frac_brown_pixels = num_brown_pixels / total_pixels
    frac_dark_brown_pixels = num_dark_brown_pixels / total_pixels
    frac_gray_pixels = num_gray_pixels / total_pixels
    frac_white_pixels = num_white_pixels / total_pixels
    frac_yellow_pixels = num_yellow_pixels / total_pixels\

    # print("frac_brown_pixels",frac_brown_pixels)
    # print("frac_dark_brown_pixels",frac_dark_brown_pixels)
    # print("frac_gray_pixels",frac_gray_pixels)
    # print("frac_white_pixels",frac_white_pixels)
    # print("frac_yellow_pixels",frac_yellow_pixels)
    
    # Determine the dominant color based on the fraction of pixels in each binary mask
    if (frac_brown_pixels > 0.4 and frac_dark_brown_pixels > 0.1 and frac_white_pixels > 0.5) or frac_yellow_pixels > 0.3 :
        return 'table-egg'
    elif (frac_brown_pixels > 0.4 and frac_dark_brown_pixels > 0.3 and frac_white_pixels < 0.55) and frac_yellow_pixels < 0.25:
        return 'rotten-egg'
    else:
        return 'balut-penoy'
def classify_egg_color(egg_color):
    # Define a dictionary mapping egg colors to class labels
    class_map = {
        'balut-penoy': 1,
        'rotten-egg': 2,
        'table-egg': 3,
        'unknown': 4
    }
    
    # Map the egg color to a class label
    return class_map.get(egg_color, 4)


# initialize empty lists for training and testing data and labels
train_data = []
train_labels = []
test_data = []
test_labels = []

# loop over training images in each directory and extract features
for egg_type in ['balut-penoy','rotten-egg', 'table-egg']:
    # egg_type_dir = os.path.join('Datasets', 'Training', egg_type)
    egg_type_dir = data_dir / 'Training' / egg_type
    print("Training Now", egg_type_dir)
    for image_file in os.listdir(egg_type_dir):
        if image_file.startswith('.'):
            continue
        image_path = os.path.join(egg_type_dir, image_file)
        image = cv2.imread(image_path)
        eggs = detect_eggs(image)
        # print("eggs", len(eggs))
        for egg in eggs:
            x1, y1, x2, y2 = egg
            egg_image = image[y1:y1+y2, x1:x1+x2]
            # cv2.imshow('Egg Image', egg_image)
            # cv2.waitKey(0)
            egg_color = detect_egg_color(egg_image,egg)
            # print("egg_color",egg_color)
            if (egg_color == "invalid"):
                continue
            else:
                egg_type_label = classify_egg_color(egg_color)
                egg_image = cv2.resize(egg_image, (200, 300), interpolation=cv2.INTER_LINEAR)
                train_data.append(np.array(egg_image.flatten()))
                train_labels.append(egg_type_label)

# loop over testing images in each directory and extract features
for egg_type in ['balut-penoy', 'rotten-egg', 'table-egg']:
    egg_type_dir = data_dir / 'Testing' / egg_type
    print("Testing Now", egg_type_dir)
    for image_file in os.listdir(egg_type_dir):
        if image_file.startswith('.'):
            continue
        image_path = os.path.join(egg_type_dir, image_file)
        image = cv2.imread(image_path)
        eggs = detect_eggs(image)
        for egg in eggs:
            x1, y1, x2, y2 = egg
            egg_image = image[y1:y1+y2, x1:x1+x2]
            egg_color = detect_egg_color(egg_image,egg)
            if (egg_color == "invalid"):
                continue
            else:
                egg_type_label = classify_egg_color(egg_color)
                egg_image = cv2.resize(egg_image, (200, 300), interpolation=cv2.INTER_LINEAR)
                test_data.append(np.array(egg_image.flatten()))
                test_labels.append(egg_type_label)

# convert data and labels to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# train KNN classifier on training data and labels
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_data, train_labels)

# evaluate classifier on testing data and labels
accuracy = knn.score(test_data, test_labels)
print('Accuracy:', accuracy)


cv2.destroyAllWindows()