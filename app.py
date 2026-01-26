import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import efficientnet

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    IMG_SIZE = (300, 300)  # Must match training script
    CLASS_NAMES = ['Both', 'Infection', 'Ischaemia', 'None']
    MODEL_PATH = 'best_model_stage2.h5'  # The file output by your training script

cfg = Config()

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_tf_model():
    try:
        # compile=False is safer for inference (avoids optimizer version issues)
        model = tf.keras.models.load_model(cfg.MODEL_PATH, compile=False)
        return model, True
    except FileNotFoundError:
        return None, False
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocess_image(image):
    """
    1. Resize to (300, 300)
    2. Convert to Array
    3. Apply EfficientNet preprocessing (scaling/normalization)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(cfg.IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 300, 300, 3)
    
    # Critical: Use the exact same preprocessing function as training
    img_array = efficientnet.preprocess_input(img_array)
    return img_array

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="DFU Classifier", page_icon="🦶", layout="centered")

st.title("🩺 Diabetic Foot Ulcer Classifier")
st.markdown(f"**Model:** EfficientNetB3 + Transformer Head | **Format:** TensorFlow `.h5`")

# Sidebar for Model Status
st.sidebar.header("System Status")
model, status = load_tf_model()

if status is True:
    st.sidebar.success("✅ Model Loaded")
elif status is False:
    st.sidebar.error(f"❌ File Not Found")
    st.sidebar.warning(f"Please place `{cfg.MODEL_PATH}` in this folder.")
else:
    st.sidebar.error(f"❌ Error: {status}")

# Main Upload Area
uploaded_file = st.file_uploader("Upload a foot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Layout: Image on Left, Results on Right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption='Patient Image', use_column_width=True)
        # Open image for processing
        image = Image.open(uploaded_file)

    # Run Prediction
    if st.button('Analyze Condition', type="primary") and model:
        with st.spinner('Running AI Analysis...'):
            # 1. Preprocess
            input_tensor = preprocess_image(image)
            
            # 2. Predict
            preds = model.predict(input_tensor, verbose=0)
            probs = preds[0]  # Get first item in batch
            
            # 3. Process Result
            pred_idx = np.argmax(probs)
            pred_class = cfg.CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx]
            
            # 4. Display Results
            with col2:
                st.subheader("Assessment Result")
                
                # Dynamic Color Formatting
                if pred_class == "None":
                    st.success(f"### {pred_class}")
                    st.caption("No distinct pathology detected.")
                else:
                    st.error(f"### {pred_class}")
                    st.caption("Pathology detected.")
                
                st.metric("Confidence Score", f"{confidence:.1%}")
                
                # Bar Chart
                st.markdown("---")
                st.markdown("**Class Probabilities:**")
                chart_data = pd.DataFrame({
                    "Condition": cfg.CLASS_NAMES,
                    "Probability": probs
                })
                st.bar_chart(chart_data.set_index("Condition"))