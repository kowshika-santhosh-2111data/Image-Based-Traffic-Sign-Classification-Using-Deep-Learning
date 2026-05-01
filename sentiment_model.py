from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing import clean_text, tokenize_text, remove_stop_words


def train_sentiment_model(df):
    # Clean the text data
    df['CleanedText'] = df[df.columns[0]].apply(clean_text)
    
    # Tokenize the cleaned text
    df['Tokens'] = df['CleanedText'].apply(tokenize_text)
    
    # Remove stop words
    stop_words = set(['the', 'is', 'in', 'and', 'to', 'of'])  # Example stop words
    df['FilteredTokens'] = df['Tokens'].apply(lambda tokens: remove_stop_words(tokens, stop_words))
    
    # Join filtered tokens back into strings for vectorization
    df['ProcessedText'] = df['FilteredTokens'].apply(lambda tokens: ' '.join(tokens))
    
    #create vectorizer 
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['ProcessedText'])
    
    # Split the data into training and testing sets
    X_train, X_test_text, y_train, y_test_text = train_test_split(tfidf_matrix, df['sentiment'], test_size=0.2, random_state=42)
    
    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test_text, y_test_text)
    print(f"Sentiment Model Accuracy: {accuracy:.2f}")
    
    return model,vectorizer

# --- mapping ---
sentiment_mapping = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}


#sentiment prediction
def get_category(text):
    text = text.lower()
    if "damage" in text or "pothole" in text:
        return "Road Condition" 
    elif "traffic" in text:
        return "Traffic Issue"
    else:
        return "General Complaint"
