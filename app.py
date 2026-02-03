import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration & Constants ---
st.set_page_config(
    page_title="Retail Product Classifier",
    page_icon="🛒",
    layout="centered"
)

# Image size used during training (from Cell 5 of your notebook)
IMG_SIZE = (224, 224)

# Class labels (from Cell 10 of your notebook)
CLASS_NAMES = [
    'Cake', 'Candy', 'Cereal', 'Chips', 'Chocolate', 'Coffee', 
    'Fish', 'Honey', 'Jam', 'Milk', 'Oil', 'Pasta', 
    'Rice', 'Soda', 'Sugar', 'Tea', 'Vinegar', 'Water'
]

# --- Helper Functions ---

@st.cache_resource
def load_classifier_model():
    """
    Loads the trained Keras model. 
    Using cache_resource to ensure it loads only once.
    """
    # Ensure this filename matches what you saved in Cell 15
    model_path = 'Classification_model_test4.keras' 
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please place the .keras file in the same directory as this script.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(uploaded_image):
    """
    Preprocesses the image to match the training data format:
    1. Resize to (224, 224)
    2. Convert to NumPy array
    3. Normalize pixel values (1./255)
    4. Expand dims to create a batch of 1
    """
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# --- UI Structure ---

st.title("🛒 Retail Product Classification")
st.markdown("""
This app uses a Deep Learning (CNN) model to classify retail products into 18 different categories.
**Upload an image of a grocery item to get started!**
""")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.markdown("""
* **Type:** CNN (Sequential)
* **Input Shape:** 224x224x3
* **Classes:** 18
* **Framework:** TensorFlow/Keras
""")

# Load Model
model = load_classifier_model()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Image")
        # Preprocess
        display_img, processed_img = preprocess_image(uploaded_file)
        st.image(display_img, caption='Uploaded Item', use_column_width=True)

    with col2:
        st.subheader("Prediction")
        
        # Add a classify button
        if st.button('Classify Product', type="primary"):
            with st.spinner('Analyzing...'):
                # Make Prediction
                predictions = model.predict(processed_img)
                score = tf.nn.softmax(predictions[0]) # Depending on model output, sometimes raw softmax is already returned
                
                # If the last layer was activation='softmax' (Cell 11), we just take the max
                confidence = np.max(predictions) * 100
                class_index = np.argmax(predictions)
                predicted_class = CLASS_NAMES[class_index]
                
                # Display Result
                st.success(f"**It's a {predicted_class}!**")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Optional: Display probability distribution graph
                st.markdown("---")
                st.write("**Top 3 Probabilities:**")
                
                # Get top 3 predictions
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                for i in top_3_indices:
                    st.write(f"- **{CLASS_NAMES[i]}:** {predictions[0][i]*100:.2f}%")
                    st.progress(int(predictions[0][i]*100))

# Footer
st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")
