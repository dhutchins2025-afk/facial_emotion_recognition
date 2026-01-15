from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from datetime import datetime
import base64
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load the trained emotion model
try:
    model = tf.keras.models.load_model('emotion_model.h5')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Emotion labels
EMOTIONS = ['Angry', 'Disgusted', 'Fearful',
            'Happy', 'Neutral', 'Sad', 'Surprised']

# Emotion descriptions and recommendations
EMOTION_DETAILS = {
    'Angry': {
        'description': 'You seem upset or frustrated.',
        'recommendation': 'Take a deep breath and step back for a moment. Try to identify what\'s bothering you.'
    },
    'Disgusted': {
        'description': 'You appear to feel repulsed or displeased.',
        'recommendation': 'Remember that not everything is worth your energy. Focus on what matters.'
    },
    'Fearful': {
        'description': 'You seem anxious or worried.',
        'recommendation': 'It\'s okay to feel nervous. Remember your strengths and past successes.'
    },
    'Happy': {
        'description': 'You\'re in a great mood!',
        'recommendation': 'Keep smiling! Share your positive energy with others around you.'
    },
    'Neutral': {
        'description': 'You appear calm and composed.',
        'recommendation': 'You\'re in control. Channel this calmness into productive activities.'
    },
    'Sad': {
        'description': 'You seem down or melancholic.',
        'recommendation': 'It\'s okay to feel sad. Reach out to someone you trust or engage in activities you enjoy.'
    },
    'Surprised': {
        'description': 'You look astonished or taken aback.',
        'recommendation': 'Surprises can be positive! Stay curious and open to new experiences.'
    }
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img, target_size=(48, 48)):
    """Preprocess image for model prediction"""
    try:
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to model input size
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to array and normalize
        img_array = np.array(img, dtype='float32')
        img_array = img_array / 255.0

        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_emotion(img_array):
    """Predict emotion from preprocessed image"""
    try:
        if model is None:
            return None, "Model not loaded"

        predictions = model.predict(img_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = EMOTIONS[emotion_idx]

        return emotion, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle emotion prediction from uploaded image or base64 data"""
    try:
        # Check if image is provided
        if 'image' not in request.files and 'imageData' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        img = None
        filename = None

        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400

            img = Image.open(file.stream)
            filename = secure_filename(file.filename)

        # Handle base64 image data (from webcam)
        elif 'imageData' in request.form:
            image_data = request.form['imageData']
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes))
                filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

        if img is None:
            return jsonify({'error': 'Could not process image'}), 400

        # Preprocess and predict
        img_array = preprocess_image(img)
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400

        emotion, confidence = predict_emotion(img_array)
        if emotion is None:
            return jsonify({'error': f'Prediction failed: {confidence}'}), 500

        # Save uploaded image
        if filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)

        # Get emotion details
        details = EMOTION_DETAILS.get(emotion, {})

        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'description': details.get('description', ''),
            'recommendation': details.get('recommendation', ''),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'emotions': EMOTIONS
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
