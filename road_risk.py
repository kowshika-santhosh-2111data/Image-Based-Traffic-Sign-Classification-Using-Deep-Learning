from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_tabular_data


def train_tabular_model(df, target_column="RiskLevel"):
    # Preprocess the tabular data
    x, y, scaler = preprocess_tabular_data(df, target_column)
    
    # Split the data into training and testing sets
    x_train, x_test_tab, y_train, y_test_tab = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(x_test_tab, y_test_tab)
    print(f"Tabular Model Accuracy: {accuracy:.2f}")
    
    return model, scaler, x_test_tab, y_test_tab


# --- mapping ---
risk_mapping = {
    0: 'Low Risk',
    1: 'Medium Risk',
    2: 'High Risk'
}
