# Financial Fraud Detection System
A financial fraud detection system built using PyTorch, FastAPI, and Gradio.
This project uses an Artificial Neural Network (ANN) trained on credit card transaction data to identify potentially fraudulent transactions. The application provides both a FastAPI backend for predictions and a Gradio interface for batch CSV uploads.
The goal of the project was to build a simple end-to-end machine learning system that can take real transaction data, process it through a trained model, and return fraud predictions with probability scores.
---
# Features
- Fraud detection using a PyTorch ANN model
- FastAPI backend for serving predictions
- Gradio dashboard for batch transaction uploads
- CSV-based prediction workflow
- Fraud probability scoring
- Simple deployment-ready structure
---
# Tech Stack
## Backend
- Python
- FastAPI
- Uvicorn
## Machine Learning
- PyTorch
- Pandas
- NumPy
- Joblib
## Frontend
- Gradio
---
# Model Performance
The model was trained on a highly imbalanced fraud detection dataset.
| Metric | Score |
|---|---|
| Recall | 0.84 |
| F1-Score | 0.75 |
The focus was improving recall so the system can detect as many fraudulent transactions as possible.
---
# Dataset
This project uses the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
### Dataset Details
- 284,807 transactions
- Highly imbalanced classes
- Features `V1` to `V28` are PCA-transformed
- `Amount` represents transaction amount
- `Class`
  - `0` → Legitimate
  - `1` → Fraud
---
# Project Structure
```bash
Financial-Fraud-Detection-System/
│
├── api/
│   └── main.py
│
├── app/
│   └── gradio_app.py
│
├── model/
│   ├── model.pth
│   └── scaler.pkl
│
├── data/
│   └── creditcard.csv
│
├── requirements.txt
└── README.md
```
---
# Installation
## 1. Clone the repository
```bash
git clone https://github.com/MohomedShajith/Financial-Fraud-Detection-System.git
cd Financial-Fraud-Detection-System
```
---
## 2. Create a virtual environment
### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
---
## 3. Install dependencies
```bash
pip install -r requirements.txt
```
---
# Running the Project
## Start the FastAPI server
```bash
uvicorn api.main:app --reload
```
API runs on:
```bash
http://127.0.0.1:8000
```
Swagger docs:
```bash
http://127.0.0.1:8000/docs
```
---
## Start the Gradio app
```bash
python app/gradio_app.py
```
The Gradio dashboard will open in your browser.
---
# Batch Prediction Workflow
1. Export transactions as a CSV file
2. Upload the file into the Gradio app
3. Each transaction is sent to the FastAPI endpoint
4. Predictions and fraud probabilities are returned in a table
---
# Expected CSV Format
```text
Time,V1,V2,V3,...,V28,Amount
```
Optional column:
```text
Class
```
If the CSV contains the `Class` column, it is automatically ignored during prediction.
---
# Example API Response
```json
{
  "Fraud": 1,
  "Fraud_probability": 0.913
}
```
---
# Future Improvements
- Docker support
- Cloud deployment
- Model monitoring
- SHAP explainability
- Real-time transaction streaming
- Better ANN architectures
- Database integration
---
# Author
Mohomed Shajith
GitHub:
https://github.com/MohomedShajith
Project Repository:
https://github.com/MohomedShajith/Financial-Fraud-Detection-System