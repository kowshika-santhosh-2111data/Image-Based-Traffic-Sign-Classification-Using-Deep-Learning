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


#final integration system
def smart_mobility_system(image, road_data, social_media_post):

    #---------------------Image---------------------------------
    image = np.array(image) / 255.0

    if image.shape != (32, 32, 3):
        raise ValueError("Image must be 32x32x3")
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension for prediction

    #----------------------Traffic Sign--------------------------
    sign_pred = cnn_model.predict(image)
    class_id = np.argmax(sign_pred)
    confidence_score = round(np.max(sign_pred) * 100, 2)

    if confidence_score < 50:
        sign_name = "Uncertain Sign"
    else:
        sign_name = sign_classes[class_id]

    #debug
    print("Raw prediction:", sign_pred)

    #----------------------Road Risk-----------------------------
    risk_input = pd.DataFrame([road_data], columns = [
        'traffic_density',
        'road_condition',
        'visibility',
        'accident_history'
    ])
    risk_input_scaled = loaded_scaler.transform(risk_input)
    risk_pred = loaded_risk_model.predict(risk_input_scaled)[0]
    risk_label = risk_mapping.get(risk_pred, "Unknown")
    
    #----------------------Sentiment Analysis---------------------
    cleaned_text = clean_text(social_media_post)
    vec = loaded_vectorizer.transform([cleaned_text])
    sent_pred = loaded_sentiment_model.predict(vec)[0]
    sent_label = sentiment_mapping.get(sent_pred, "Unknown")

    #----------------------Category-------------------------------
    category = get_category(cleaned_text)

    return sign_name, confidence_score, risk_label,sent_label, category



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

    st.image(image, caption='Uploaded Image', use_container_width=True)
  
#location
st.header("Your Location")

city = st.text_input('Enter your location')
g = geocoder.ip('me')

location = "Unknown"
if city:
    location = city
    st.success(f"you are accessing from:{city}")
elif g.ok and g.city:
    location = g.city
    st.info(f"Using approximate IP based location: {g.city}")
else:
    st.error("Unable to fetch location")

# Road risk input
st.subheader("Road Risk Assessment")
road_data_input = st.text_input("Enter road segment data (comma-separated values)")

# Sentiment analysis input
st.subheader("Complaint Sentiment Analysis")
text = st.text_area("Enter a social media post or complaint")

#-------------------Run system-------------
st.subheader("Run smart mobility system")

if st.button("Analyze"):
    if uploaded_file and road_data_input and text:

        #image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((32,32))

        #road data
        try:
            road_data = list(map(float,road_data_input.split(',')))
        except:
            st.error("Invalid road input")
            st.stop()
        
        #call system
        sign_name, confidence, risk_label, sent_label, category = smart_mobility_system(
            image, road_data, text
        )
        # Output
        st.markdown("##Results")
        col1,col2 = st.columns(2)

        with col1:
            st.image(image, caption = "uploaded image")
        with col2:
            st.success(f"🚦 Traffic Sign: {sign_name} ({confidence:.2f}%)")

            if risk_label == "High Risk":
                st.error(f"🚧 Road Risk: {risk_label}")
            else:
                st.success(f"🚧 Road Risk: {risk_label}")

            if sent_label == "Negative":
                st.warning(f"sentiment: {sent_label}")
            else:
                st.success(f"💬 Sentiment: {sent_label}")
            st.info(f"Complaint Category: {category}")

    else:
        st.warning("Please provide all inputs")
