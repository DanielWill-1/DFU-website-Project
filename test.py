import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.applications import efficientnet

# ==========================================
# CONFIGURATION
# ==========================================
app = Flask(__name__)
MODEL_PATH = "best_model_stage2.h5"
CLASS_NAMES = ['Both', 'Infection', 'Ischaemia', 'None']
IMG_SIZE = (300, 300)

# ==========================================
# LOAD MODEL
# ==========================================
print("Loading Keras model...")
try:
    # compile=False is safer/faster for inference
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def preprocess_image(image):
    """Resize and preprocess image for EfficientNet"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return efficientnet.preprocess_input(img_array)

def get_image_base64(image):
    """Convert PIL image to base64 string to display in HTML"""
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('ascii')

# ==========================================
# ROUTES
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_data = None
    probs_dict = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file uploaded."
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected."
            elif model is None:
                error = "Model not loaded."
            else:
                try:
                    # 1. Read Image
                    image = Image.open(file.stream)
                    
                    # 2. Convert for Display (Base64)
                    img_data = get_image_base64(image)
                    
                    # 3. Preprocess & Predict
                    input_tensor = preprocess_image(image)
                    preds = model.predict(input_tensor, verbose=0)
                    probs = preds[0]
                    
                    # 4. Process Results
                    pred_idx = np.argmax(probs)
                    prediction = CLASS_NAMES[pred_idx]
                    confidence = round(float(probs[pred_idx]) * 100, 2)
                    
                    # Create dictionary for progress bars { 'Infection': 80.5, ... }
                    probs_dict = {cls: round(float(score)*100, 1) for cls, score in zip(CLASS_NAMES, probs)}
                    
                except Exception as e:
                    error = f"Error processing image: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           img_data=img_data, 
                           probs=probs_dict,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5000)