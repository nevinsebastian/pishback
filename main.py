from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
from tokenizer import makeTokens  

# Read phishing URLs data
phish_data = pd.read_csv("datasets/verified_online.csv")
phish_urls = phish_data.sample(n=5000, random_state=12).copy().reset_index(drop=True)

# Read legitimate URLs data
legit_data = pd.read_csv("datasets/Benign_url_file.csv")
legit_data.columns = ['URLs']

# Combine phishing and legitimate URLs
combined_data = pd.concat([phish_urls, legit_data], ignore_index=True)
combined_data['Label'] = [1] * len(phish_urls) + [0] * len(legit_data)
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Handle NaN values
combined_data['URLs'].fillna('', inplace=True)

# Split data into features and labels
X_train = combined_data['URLs']
y_train = combined_data['Label']

# Initialize CountVectorizer
vectorizer = CountVectorizer(tokenizer=makeTokens, token_pattern=None)

# Transform features
X_train_transformed = vectorizer.fit_transform(X_train)

# Initialize Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X_train_transformed, y_train)


class InputData(BaseModel):
    urls: str


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    # Preprocess input URLs
    urls_processed = [re.sub(r'^((?!www\.)[a-zA-Z0-9-]+\.[a-z]+)$', r'www.\1', input_data.urls)]
    
    # Transform input URLs
    X_predict = vectorizer.transform(urls_processed)
    
    # Make predictions
    predictions = logit.predict(X_predict)
    
    # Map prediction labels
    label_mapping = {0: "bad", 1: "good"}
    predictions_labels = [label_mapping[pred] for pred in predictions]

    return {"url": input_data.urls, "prediction": predictions_labels[0]}

