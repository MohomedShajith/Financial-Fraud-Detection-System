import os
import torch
import joblib
import pandas as pd
from torch import nn
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient
from model.train import Fraud_model


app =FastAPI()
load_model =  Fraud_model()
load_model.load_state_dict(torch.load(r"Models\fraud prediction neural network model.pth"))
scaler = joblib.load('model/scaler.pkl')

load_model.eval()

load_dotenv()

url = os.getenv("MONGO_URI")
client = MongoClient(url)
db = client["frauddb"]
collection = db["predictions"]


class Transaction(BaseModel):
    Time : float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float   
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    

@app.post('/predict')
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.dict()])
    df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
    
    df = df.values
    df = torch.FloatTensor(df)

    pred_logist = load_model(df)
    preds = torch.round(pred_logist)
    collection.insert_one({"transaction":transaction.dict(),
    "predictions":int(preds.item()),
    "probability":round(pred_logist.item(),3),
    "timestamp":datetime.utcnow()})

    return {"Fraud": int(preds.item()),"Fraud_probability": round(pred_logist.item(),3) }
    