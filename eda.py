import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

base_path = "D:/DOWNLOADS/German_traffic"

def eda(train_data,base_path):
   
    # Display basic information about the dataset
    print("Dataset Information:")
    train_data.info()   
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(train_data.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(train_data.isnull().sum())
    
    print("\nclass counts:")
    print(train_data['ClassId'].value_counts())
    
    # class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='ClassId', data=train_data)
    plt.title('Class Distribution')
    plt.show()

    # Visualize sample images from the dataset
    plt.figure(figsize=(10, 6))
    sample = train_data.sample(9)  # Randomly sample 9 images
    for i,(_,row) in enumerate(sample.iterrows()):
        img_path = os.path.join(base_path, row['Path'])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read image at {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
        plt.subplot(3, 3, i+ 1)
        plt.imshow(img)
        plt.title(f"Class ID: {row['ClassId']}")
        plt.axis('off')

        #print(f"Image {i+1} - Class ID: {row['ClassId']} - Path: {img_path}")
    plt.tight_layout()        
    plt.show()  

#image check
def check_images(train_data,base_path):
    for i in range(len(train_data)):
        img_path = os.path.join(base_path, train_data['Path'][i])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read image at {img_path}. Skipping.")
            continue
        print(f"Successfully read image at {img_path}")


if __name__ == "__main__":
    # Load the dataset
    train_data = pd.read_csv('train.csv')
        
    # Perform EDA
    eda(train_data,base_path) 
    # Check images
    check_images(train_data,base_path)
