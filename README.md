# Image-Based-Traffic-Sign-Classification-Using-Deep-Learning
## Overview
The project develops a smart mobility system using machine learning and deep learning techniques. It uses CNN for traffic sign classification from images, a Random Forest model for road risk prediction based on tabular data , and a NLP  model using TF-IDF and Logistic Regression for sentiment analysis of  user complaints.

## Problem statement
  For road safety, traffic signals are used to conevy the instructions to the drivers. An automated recognition system used to improve safety and efficieny on roads. for example, a system should recognize "speed limit sign board", it alerts the driver to drive the car at that particular speed.

## Dataset
  The dataset used for this project is the German Traffic Sign REcognition Benchmark(GTSRB). It consists of 50000 images categorize into 43 classes of traffic signs.
 - Train.csv: Contains paths & labels for training set.
 - Test.csv: Contains paths & labels for test set.
   
### Data Structure
```text
GTSRB/
│
├── Train/
│   ├── 0/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   │
│   ├── 1/
│   ├── 2/
│   └── ...
│
├── Test/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── Train.csv
└── Test.csv
```
## Project Structure
```text
traffic-sign-recognition/
│
├── data/
│   ├── Meta/
│   ├── Test/
│   ├── Train/
│   └── ...
│
├── Scripts/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── traffic_signal.py
│   ├── road_risk.py
│   ├── sentiment_model.py
│   ├── evaluation.py
│   └── streamlit.py
│
├── main.py
├── requirements.txt
└── README.md
```
### Link
https://drive.google.com/drive/folders/1ocqH2PH-XfRYgsLJqHOvEG7rIlw_IMH_?usp=drive_link

## Model & Data Files
```text
-x_data.npy : Preprocessed image dat afor faster training.
-vectorizer.pkl : Converts text into numerical features (TD-IDF)
-road_risk_model.pkl : Random Forest model for road risk prediction.
-sentiment_model.pkl : Model for sentiment analysis.
-traffic_signal_model.keras : CNN model for traffic sign classification.
```
> Note: Due to Github file size limitation, model files are provided via external link.

# Usage
## Step 1: Train Models
Run:
python main.py

This will:
  - Train all models
  - Save .Keras and .pkl files
Skip this step if model files already exist.

# Step 2: Run the Application
streamlit run streamlit.py
The app will open in your browser.

## Output
The system will display:
- Detected Traffic Sign + Confidence
- Road Risk Level
- Sentiment(Positive /Negative)
- Complaint Category
- Location

## Notes
```text
~Ensure image is clear for better accuracy
~Enter exactly 4 numeric values for road data
~Models must be trained before running the app
~Location is used only for display (not prediction)
```
