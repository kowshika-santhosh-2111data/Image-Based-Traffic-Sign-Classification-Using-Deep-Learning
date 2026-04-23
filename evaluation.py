from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    y_pred = model.predict(x_test)

    print("Classification Report:")
    
    if len(y_test.shape) > 1 :
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_test_labels = y_test
        y_pred_labels = y_pred


    print(classification_report(y_test_labels, y_pred_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))
    print("Evaluation complete.")
