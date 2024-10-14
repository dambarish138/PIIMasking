import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Read image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply thresholding
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(thresh, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    # Save the preprocessed image
    preprocessed_image_path = os.path.join(os.path.dirname(image_path), 'Q_' + os.path.basename(image_path))
    cv2.imwrite(preprocessed_image_path, img)
    
    return preprocessed_image_path
