import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import joblib
import warnings
warnings.filterwarnings("ignore")

#importing the modules
from data_loader import load_images
from preprocessing import preprocess_image_data,clean_text
from traffic_signal import build_cnn, train_model, sign_classes
from road_risk import train_tabular_model, risk_mapping
from sentiment_model import  train_sentiment_model, sentiment_mapping, get_category
from evaluation import evaluate_model

#-----------load data-------------
train_df = pd.read_csv(r"D:\DOWNLOADS\German_traffic\Train.csv")

#train_df = pd.read_csv("Train.csv")
base_path = "D:/DOWNLOADS/German_traffic"

#Add test here
print("CSV Path:",train_df['Path'][0])

test_path = os.path.normpath(os.path.join(base_path,train_df['Path'][0]))
print("Full Image Path:",test_path)

print("File Exists:",os.path.exists(test_path))

img = cv2.imread(test_path)
print("Image Loaded:",img is not None)

# -------- LOAD OR CREATE DATA --------
if os.path.exists("x_data.npy") and os.path.exists("y_data.npy"):
    print("Loading data from .npy files...")
    x = np.load("x_data.npy")
    y = np.load("y_data.npy")
else:
    print("Loading images from dataset...")
    x, y = load_images(train_df, base_path)

    x = np.array(x).astype('float32')
    y = np.array(y)

    print("Saving data to .npy files...")
    np.save("x_data.npy", x)
    np.save("y_data.npy", y)

# -------- PREPROCESS --------
x = preprocess_image_data(x)
y = to_categorical(y, num_classes=43)

#tabular data (road_data)
road_df = pd.DataFrame({
    'traffic_density': np.random.randint(1, 10, 200),
    'road_condition': np.random.randint(1, 10, 200),
    'visibility': np.random.randint(1, 10, 200),
    'accident_history': np.random.randint(0, 5, 200),
    'RiskLevel': np.random.randint(0, 3, 200)
})

def calculate_risk(risk):
    if risk['traffic_density'] > 7 or risk['accident_history'] >3:
        return 2
    elif risk['road_condition'] < 4:
        return 1
    else:
        return 0
road_df['RiskLevel'] = road_df.apply(calculate_risk,axis =1)
#text data
complaint_data = pd.DataFrame({
    "text": [
        "Road has potholes",
        "Traffic is smooth",
        "Accident happened here",
        "Road is damaged badly",
        "No issues on road",
        "Heavy traffic jam today",
        "Street lights are not working",
        "Road is clean and well maintained",
        "Too much congestion in this area",
        "Dangerous curve without sign",
        "Very safe road to drive",
        "Frequent accidents happening here",
        "Signals are working properly",
        "Road construction causing delay",
        "No traffic at all, very smooth",
        "Bad road conditions everywhere",
        "Road is excellent and safe",
        "Vehicles moving slowly due to damage",
        "Clear road, no issues",
        "Potholes making driving difficult"
    ],
    "sentiment": [
        0,2,0,0,1,
        0,0,2,0,0,
        2,0,2,0,2,
        0,2,0,2,0
    ]
})
#traffic signal classification
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
#CNN
cnn_model = build_cnn()
cnn_model, cnn_history = train_model(cnn_model, x_train, y_train, x_test, y_test)
#save the model
cnn_model.save('traffic_signal_model.keras')


#road risk assessment
risk_model, scaler, x_test_tab, y_test_tab = train_tabular_model(road_df)
#save the model
joblib.dump(risk_model, 'road_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


#sentiment analysis
sentiment_model,vectorizer = train_sentiment_model(complaint_data)
#save the model
joblib.dump(sentiment_model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

#evaluate the models
#cnn
print("Evaluating CNN Model:")
evaluate_model(cnn_model, x_test, y_test)
#road risk
print("Road Risk Accuracy:",risk_model.score(x_test_tab,y_test_tab))
#evaluate_model(risk_model, x_test_tab, y_test_tab)

#sentiment analysis
print("Sentiment Model Evaluation already done during training")

#load and test the saved models
loaded_cnn_model = load_model('traffic_signal_model.keras')
loaded_risk_model = joblib.load('road_risk_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_sentiment_model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#final integration system
def smart_mobility_system(image, road_data, social_media_post):

    #---------------------Image---------------------------------
    image = np.array(image)

    if image.shape != (32, 32, 3):
        raise ValueError("Image must be 32x32x3")
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension for prediction

    #----------------------Traffic Sign--------------------------
    sign_pred = loaded_cnn_model.predict(image)
    class_id = np.argmax(sign_pred)
    sign_name = sign_classes[class_id]
    confidence_score = round(np.max(sign_pred) * 100, 2)
    if confidence_score < 70:
        sign_name = "Uncertain Sign"
    #----------------------Road Risk-----------------------------
    risk_input = pd.DataFrame([road_data], columns = [
        'traffic_density',
        'road_accident',
        'visibility',
        'accident_history'
    ])
    risk_input_scaled = loaded_scaler.transform(risk_input)
    risk_pred = loaded_risk_model.predict(risk_input_scaled)[0]
    risk_label = risk_mapping.get(risk_pred, "Unknown")
    
    #----------------------Sentiment Analysis---------------------
    cleaned_text = clean_text(social_media_post)
    vec = vectorizer.transform([cleaned_text])
    sent_pred = loaded_sentiment_model.predict(vec)[0]
    sent_label = sentiment_mapping.get(sent_pred, "Unknown")

    #----------------------Category-------------------------------
    category = get_category(cleaned_text)
    
    #----------------------FINAL OUTPUT-------------------------------
    print("\n======FINAL OUTPUT =======")
    print(f"Detected Sign: {sign_name}, Confidence Score: {confidence_score:.2f}%")
    print(f"Road Segment Risk Level: {risk_label}")   
    print(f"Complaint Sentiment: {sent_label}")
    print(f"Complaint Category: {category}")


#----------------------Test call-------------------------------
test_image = np.random.rand(32, 32, 3)  # Example image
test_road_data = [5,6,7,2]  # Example road data
test_social_media_post = "There is a huge pothole on Main Street causing damage to cars!"  # Example post
smart_mobility_system(test_image, test_road_data, test_social_media_post)