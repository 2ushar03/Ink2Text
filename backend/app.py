from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import pickle
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from ocr_utils import preprocess_image, extract_text_from_image, correct_spelling

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

TRAINING_DATA_PATH = 'training_data.pkl'
if os.path.exists(TRAINING_DATA_PATH):
    with open(TRAINING_DATA_PATH, 'rb') as f:
        trained_data = pickle.load(f)
else:
    trained_data = {}

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return "Flask server is running!"

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        
        if not os.path.exists(image_path):
            logging.error(f"File not saved correctly at path '{image_path}'")
            return jsonify({'error': f"File not saved correctly at path '{image_path}'"}), 500

        try:
            extracted_text = extract_text_from_image(image_path, preprocess=True)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            logging.error(f"Error during text extraction: {e}")
            return jsonify({'error': 'Error during text extraction'}), 500

        if image_path in trained_data:
            refined_text = trained_data[image_path]
        else:
            refined_text = correct_spelling(extracted_text)

        os.remove(image_path)

        return jsonify({'extracted_text': refined_text}), 200
    else:
        logging.error("Invalid file format. Only PNG, JPG, and JPEG are supported.")
        return jsonify({'error': 'Invalid file format. Only PNG, JPG, and JPEG are supported.'}), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000, host="0.0.0.0")
