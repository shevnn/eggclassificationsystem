import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Define the directories for training and testing data
data_dir = Path('D:\Documents\Modules SY. 2022-2023\CMSC 502 - Undergrad Thesis 2\Datasets')

SMALL_EGG_AREA = 350
MEDIUM_EGG_AREA = 450
LARGE_EGG_AREA = 550



def classify_egg_size(egg_image, egg_sizes):
    # Convert image to grayscale and apply threshold
    gray = cv2.cvtColor(egg_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours in thresholded image
    
    # Apply thresholding to the grayscale image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Compute the sum of all the pixel values in the binary image
    egg_area = cv2.countNonZero(thresh)

    #Divisor to classify eggs according to size
    divisor_small = 7.78
    divisor_medium = 8.18
    divisor_large = 8.46

    num_list_small = [int(x) for x in egg_sizes['small']]
    num_list_medium = [int(x) for x in egg_sizes['medium']]
    num_list_large= [int(x) for x in egg_sizes['large']]
    
    # Sort the egg areas in ascending order
    
    # Classify egg size based on the median egg area
    for size,  size_info in egg_sizes.items():
        #small
        if int(egg_area // divisor_small) in num_list_small:
            return size
        elif int(egg_area // divisor_medium)  in num_list_medium:
            return size
        elif int(egg_area // divisor_large) in num_list_large:
            return size
        else:
            # If the egg size cannot be determined, return "invalid"
            return "invalid"



def detect_eggs(image):
    
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

def load_egg_sizes(csv_file):
    egg_sizes = {}
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        size = row['Size']
        weight = row['Weight (g)']
        if size in egg_sizes:
            egg_sizes[size].append(weight)
        else:
            egg_sizes[size] = [weight]
    return egg_sizes

def extract_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 64x64
    resized = cv2.resize(gray, (64, 64))

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Threshold the image to create a binary image
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute the Hu moments feature vector
    moments = cv2.HuMoments(cv2.moments(thresholded)).flatten()

    return moments


# initialize empty lists for training and testing data and labels
train_data = []
train_labels = []
test_data = []
test_labels = []
egg_sizes = []

egg_sizes = load_egg_sizes("egg_weight.csv")
print(egg_sizes)
# loop over training images in each directory and extract features
for egg_type in ['Table Egg (Large)-samples', 'Table Egg (Medium)-samples', 'Table Egg (Small)-samples']:
    # egg_type_dir = os.path.join('Datasets', 'Training', egg_type)
    egg_type_dir = data_dir / 'Training' / egg_type
    print("Training Now", egg_type_dir)
    for image_file in os.listdir(egg_type_dir):
        if image_file.startswith('.'):
            continue
        image_path = os.path.join(egg_type_dir, image_file)
        image = cv2.imread(image_path)
        eggs = detect_eggs(image)
        print("eggs", len(eggs))
        for egg in eggs:
            x1, y1, x2, y2 = egg
            egg_image = image[y1:y1+y2, x1:x1+x2]
            egg_type_label = classify_egg_size(egg_image,egg_sizes)
            egg_image = cv2.resize(egg_image, (200, 300), interpolation=cv2.INTER_LINEAR)
            train_data.append(np.array(egg_image.flatten()))
            train_labels.append(egg_type_label)

# loop over testing images in each directory and extract features
for egg_type in ['Table Egg (Large)-samples', 'Table Egg (Medium)-samples', 'Table Egg (Small)-samples']:
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
            egg_type_label = classify_egg_size(egg_image,egg_sizes)
            egg_image = cv2.resize(egg_image, (200, 300), interpolation=cv2.INTER_LINEAR)
            test_data.append(np.array(egg_image.flatten()))
            test_labels.append(egg_type_label)

# convert data and labels to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


# instantiate the model
lr = LinearRegression()

# create a LabelEncoder object
le = LabelEncoder()

# encode the labels
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

# fit the model to the training data
lr.fit(train_data, train_labels)

# make predictions on the test data
predictions = lr.predict(test_data)

# evaluate the model
score = lr.score(test_data, test_labels)

print("score",score)
# print("R-squared score: " , score)

cv2.destroyAllWindows()