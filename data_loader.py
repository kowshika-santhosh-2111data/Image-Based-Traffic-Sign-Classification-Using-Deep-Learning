import os
import cv2
import numpy as np


def load_images(df,base_path):
    image = []
    labels = []
    #base_path = r"C:\Users\Kowsh\Documents\mini project\Final project 1\german-traffic-sign"

    for i in range(len(df)):
        img_path = os.path.normpath(os.path.join(base_path, df['Path'][i]))

        img = cv2.imread(img_path)

        if img is None:
            print("Error loading:", img_path)
            continue

        img = cv2.resize(img,(32,32))
        image.append(img)
        labels.append(df['ClassId'][i])

    return np.array(image), np.array(labels)