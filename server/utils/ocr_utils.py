import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from spellchecker import SpellChecker  # For spell checking
import re  # For regular expressions
import pickle
import os
from nltk import CFG
from nltk.parse import EarleyChartParser
import logging

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="CRNN", gpu=True)

# Initialize spell checker
spell = SpellChecker()

# Add medical terms and units to the spell checker's dictionary
medical_terms = ["mg", "mL", "kg", "mmol", "g", "L", "mg/dL", "IU", "mcg", "cc", "cm", "mm", "mmHg"]
spell.word_frequency.load_words(medical_terms)

# List of known medicine names
medicine_names = ["ibuprofen", "paracetamol", "aspirin", "amoxicillin", "omeprazole", "atorvastatin"]

# Mapping of common OCR misreadings for units (case-insensitive)
ocr_unit_mapping = {
    "coma": "50mg",
    "l50mq": "150mg",
    "mq": "mg",
    "rnq": "mg",
    "rncg": "mcg",
}

# FSM for validating terms (used in spell checking)
class FSM:
    def __init__(self, valid_terms):
        self.valid_terms = valid_terms

    def check(self, word):
        return word.lower() in self.valid_terms

# CFG for parsing sentence structure
grammar = CFG.fromstring("""
    S -> NP VP
    VP -> V NP
    V -> "takes" | "prescribes"
    NP -> "John" | "Mary" | "the" N
    N -> "pill" | "medicine"
""")
parser = EarleyChartParser(grammar)

def assess_image_quality(image):
    """
    Assess the quality of the image to determine if preprocessing is needed.
    - Returns True if preprocessing is needed, False otherwise.
    """
    if image is None:
        raise ValueError("Image is not loaded correctly. Check the file path.")
    
    # Calculate the standard deviation of the image (indicator of contrast)
    std_dev = np.std(image)
    
    # If the standard deviation is low, the image may need preprocessing
    if std_dev < 50:  # Threshold can be adjusted based on your use case
        print("Preprocessing is needed (low contrast).")
        return True
    print("No preprocessing needed (good contrast).")
    return False

def preprocess_image(image):
    """
    Preprocess the image for OCR:
    - Convert to grayscale
    - Apply Gaussian blur
    - Use CLAHE (Contrast Enhancement)
    - Adaptive thresholding for better text extraction
    """
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    enhanced_image = clahe.apply(image)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1.8)

    # Adaptive Thresholding (for better contrast)
    thresh_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 9
    )

    return thresh_image

def correct_spelling(text):
    """
    Correct spelling errors in the final extracted text using a spell checker.
    - Handles medicine names and their associated units.
    - Corrects specific OCR misreadings like "coma" and "mq".
    """
    words = text.split()
    corrected_words = []
    fsm = FSM(medical_terms)

    for word in words:
        if fsm.check(word):
            corrected_words.append(word)
        elif word.lower() in ocr_unit_mapping:
            corrected_word = ocr_unit_mapping[word.lower()]
            corrected_words.append(corrected_word)
        else:
            corrected_word = spell.correction(word)
            if corrected_word:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

    return " ".join(corrected_words)

def extract_text_from_image(image_path, preprocess=False):
    """
    Use PaddleOCR to extract text from the image while maintaining proper formatting.
    - If preprocess is True, preprocess the image before OCR.
    """
    # Read the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Debugging statement to verify image loading
    if original_image is None:
        logging.error(f"Image at path '{image_path}' could not be loaded. Check the file path.")
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded. Check the file path.")

    # Check if preprocessing is needed
    if preprocess or assess_image_quality(original_image):
        print("Preprocessing image...")
        processed_image = preprocess_image(original_image)
    else:
        print("No preprocessing needed.")
        processed_image = original_image

    # Save the processed image temporarily for OCR
    temp_image_path = "temp_processed_image.jpg"
    cv2.imwrite(temp_image_path, processed_image)

    # Perform OCR
    result = ocr.ocr(temp_image_path, cls=True)
    if result is None or len(result) == 0:
        logging.error("No text detected in the image.")
        raise ValueError("No text detected in the image.")

    boxes = [res[0] for res in result[0]]
    texts = [res[1][0] for res in result[0]]
    scores = [res[1][1] for res in result[0]]

    # Print Extracted Text
    print("\n🔹 Extracted Text:\n")
    for i, (text, confidence) in enumerate(zip(texts, scores)):
        print(f"{i+1}. {text}  (Confidence: {confidence:.2f})")

    extracted_text = ""
    previous_y = None  # Track previous word position for spacing
    line_text = ""

    for line in result:
        for word_info in line:
            bbox, (text, confidence) = word_info  # Correct unpacking
            
            # Extract top-left y-coordinate
            y1 = bbox[0][1]  

            # If y-coordinate is significantly different, add a newline
            if previous_y is not None and abs(y1 - previous_y) > 15:
                extracted_text += line_text.strip() + "\n\n"  # Extra newline for formatting
                line_text = ""

            line_text += text + " "
            previous_y = y1  # Update last y-coordinate

        extracted_text += line_text.strip() + "\n"

    return extracted_text.strip()

def load_trained_data():
    try:
        with open('trained_data.pkl', 'rb') as f:
            trained_data = pickle.load(f)
    except FileNotFoundError:
        trained_data = {}
    return trained_data

def save_trained_data(trained_data):
    with open('trained_data.pkl', 'wb') as f:
        pickle.dump(trained_data, f)

def train_model_from_dataset(images_folder, labels_folder, num_epochs=1):
    trained_data = load_trained_data()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        images = os.listdir(images_folder)
        for image_file in images:
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(images_folder, image_file)
                text_file_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                if not os.path.exists(text_file_path):
                    continue

                with open(text_file_path, 'r') as file:
                    correct_text = file.read().strip()

                trained_data[image_path] = correct_text

    save_trained_data(trained_data)
    print("Training complete!")

def display_results(image_path, is_testing=False, correct_text=None):
    """
    Display original image, preprocessed image (if needed), and extracted OCR text.
    """
    # Read the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if original_image is None:
        print(f"Error: Image at path '{image_path}' could not be loaded. Check the file path.")
        return

    # Check if preprocessing is needed
    needs_preprocessing = assess_image_quality(original_image)
    if needs_preprocessing:
        print("Image requires preprocessing.")
        preprocessed_image = preprocess_image(original_image)
    else:
        print("Image does not require preprocessing.")
        preprocessed_image = original_image

    plt.figure(figsize=(12, 6))

    # Show Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Show Preprocessed Image (if applicable)
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title("Preprocessed Image (For OCR)" if needs_preprocessing else "No Preprocessing Needed")
    plt.axis('off')

    plt.show()

    # Extract OCR text
    extracted_text = extract_text_from_image(image_path, preprocess=needs_preprocessing)

    # Correct spelling in the final extracted text
    corrected_text = correct_spelling(extracted_text)

    print("\n📜 Extracted Text from Image:\n")
    print(extracted_text)  # Print original extracted text
    print("\n📜 Corrected Text:\n")
    print(corrected_text)  # Print corrected text
    print("-" * 50)

    if is_testing:
        trained_data = load_trained_data()
        
        if image_path in trained_data:
            refined_text = trained_data[image_path]
            print("\n📜 Refined Text from Trained Data:\n", refined_text)
        else:
            corrected_text = correct_spelling(extracted_text)
            print("\n📜 Corrected Text (After Spell Check):\n", corrected_text)
    else:
        if correct_text:
            print(f"Training with image: {image_path} -> {correct_text}")
            trained_data[image_path] = correct_text
            save_trained_data(trained_data)

