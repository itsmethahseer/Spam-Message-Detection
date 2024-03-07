from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the FastAPI app
app = FastAPI()

# Load the pre-trained model
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('tokenizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Define the input data model
class Message(BaseModel):
    text: str

# Define the endpoint to classify messages
@app.post("/classify/")
async def classify_message(message: Message):
    # Preprocess the input message
    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        wnl = WordNetLemmatizer()
        processed_text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
        processed_text = processed_text.lower()
        processed_text_words = processed_text.split()
        processed_text_words = [word for word in processed_text_words if word not in stop_words]
        final_message = [wnl.lemmatize(word) for word in processed_text_words]
        final_message = ' '.join(final_message)
        return final_message

    # Preprocess the input message
    processed_message = preprocess_text(message.text)

    # Transform the preprocessed message using TF-IDF vectorizer
    text_features = tfidf.transform([processed_message]).toarray()

    # Make prediction using the model
    prediction = model.predict(text_features)[0]
    
    # Return the prediction
    return {"message": message.text, "is_spam": bool(prediction)}
