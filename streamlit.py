import joblib
import streamlit as st
import numpy as np
import joblib
import geocoder
from tensorflow.keras.models import load_model
from PIL import Image
from preprocessing import clean_text
from traffic_signal import sign_classes
from road_risk import risk_mapping    
from sentiment_model import  sentiment_mapping, get_category

#-----------load models-------------
cnn_model = load_model('traffic_signal_model.keras')
loaded_risk_model = joblib.load('road_risk_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_sentiment_model = joblib.load('sentiment_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

#-----------Image Preprocessing------------
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

#-----------Streamlit App----------------
st.title("Smart Mobility System")

# Image input
uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((32,32))

    st.image(image, caption='Uploaded Image', use_container_width=True)
  
    # Preprocess and predict traffic sign
    preprocessed_image = preprocess_image(image)

    sign_pred = cnn_model.predict(preprocessed_image)
    class_id = np.argmax(sign_pred)
    confidence_score = round(np.max(sign_pred) * 100, 2)
    sign_name = sign_classes[class_id]

    if confidence_score < 50:
        sign_name = 'Uncertain Sign'
    
    st.write(f"Detected Sign: {sign_name}")
    st.success(f"🚦 Traffic Sign: {sign_name} with {confidence_score:.2f}% confindence")
    st.info(f"Confidence Score: {confidence_score:.2f}%")

#location
city = st.text_input('Enter your location')
st.header("Your Location")
g = geocoder.ip('me')
if city:
    st.success(f"you are accessing from:{city}")
elif g.ok and g.city:
    st.info(f"Using approximate IP based location: {g.city}")
else:
    st.error("Unable to fetch location")
# Road risk input
st.subheader("Road Risk Assessment")
road_data_input = st.text_input("Enter road segment data (comma-separated values)")
if road_data_input:
    try:  
        road_data = list(map(float, road_data_input.split(',')))
        risk_input = np.array(road_data).reshape(1, -1)
        risk_input_scaled = loaded_scaler.transform(risk_input)
        risk_pred = loaded_risk_model.predict(risk_input_scaled)[0]
        risk_label = risk_mapping.get(risk_pred, "Unknown")
        if risk_label == "High Risk":
            st.error("This area requries immediate attention")
        st.success(f"🚧 Risk Level: {risk_label}")
    except:
        st.error("Invalid input. Please enter comma-separated numeric values.")

# Sentiment analysis input
st.subheader("Complaint Sentiment Analysis")
text = st.text_area("Enter a social media post or complaint")
if text:
    cleaned_text = clean_text(text)
    vec = loaded_vectorizer.transform([cleaned_text])
    sent_pred = loaded_sentiment_model.predict(vec)[0]
    sent_label = sentiment_mapping.get(sent_pred, "Unknown")
    category = get_category(cleaned_text)
    st.success(f"💬 Sentiment: {sent_label}")
    st.info(f"Complaint Category: {category}")