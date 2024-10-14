import ssl
import certifi
import pytesseract
import spacy
from pdf2image import convert_from_path
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os

# Ensure SSL uses certifi certificates
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

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

def extract_text_with_coordinates(image_path):
    preprocessed_image_path = preprocess_image(image_path)
    data = pytesseract.image_to_data(preprocessed_image_path, output_type=pytesseract.Output.DICT)
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # confidence threshold
            left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            results.append(((left, top, left + width, top + height), data['text'][i]))
    return results

def identify_pii(text, analyzer):
    results = analyzer.analyze(text=text, entities=["CUSTOM_ID", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"], language='en')
    return results

def mask_pii_in_image(image_path, pii_results, ocr_results, text):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for pii_result in pii_results:
        entity_text = text[pii_result.start:pii_result.end]
        for ocr_result in ocr_results:
            if entity_text in ocr_result[1]:
                coordinates = ocr_result[0]
                draw.rectangle([coordinates[:2], coordinates[2:]], fill="black")
    
    return image

def process_receipt(file_path, analyzer):
    if file_path.endswith('.pdf'):
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            image_path = f"page_{i}.png"
            image.save(image_path)
            process_image(image_path, analyzer)
    else:
        process_image(file_path, analyzer)

def process_image(image_path, analyzer):
    ocr_results = extract_text_with_coordinates(image_path)
    combined_text = " ".join([result[1] for result in ocr_results])
    pii_results = identify_pii(combined_text, analyzer)
    masked_image = mask_pii_in_image(image_path, pii_results, ocr_results, combined_text)
    masked_image.save("masked_" + image_path)

def create_analyzer():
    # Load the spaCy model locally
    nlp = spacy.load("en_core_web_sm")
    nlp_engine = SpacyNlpEngine(nlp)
    
    # Create a custom recognizer (example)
    patterns = [Pattern(name="CUSTOM_ID_PATTERN", regex=r"\bCUST-\d{5}\b", score=0.8)]
    custom_recognizer = PatternRecognizer(supported_entity="CUSTOM_ID", patterns=patterns)
    
    # Initialize the analyzer engine with the custom recognizer
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(custom_recognizer)
    
    return analyzer

# Create the analyzer
analyzer = create_analyzer()

# Example usage
process_receipt("receipt2.jpeg", analyzer)  # or "receipt.jpg", analyzer
