from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tokenizer import makeTokens


urls_data = pd.read_csv("datasets/urlbadgood.csv")
X_train = urls_data["url"]
y_train = urls_data["label"]

vectorizer = TfidfVectorizer(tokenizer=makeTokens)
X_train_transformed = vectorizer.fit_transform(X_train)

#Initialize the model
logit = LogisticRegression()
logit.fit(X_train_transformed, y_train)


class InputData(BaseModel):
    urls: list

app = FastAPI()


@app.post("/predict")
async def predict(data: InputData):
    
    X_predict = vectorizer.transform(data.urls)
    
    predictions = logit.predict(X_predict)
    return {"predictions": predictions.tolist()}
