import gradio as gr
import pandas as pd
import requests
import plotly.express as px

def predict_from_csv(file):
   
    df = pd.read_csv(file.name)

   
    predictions = []
    probabilities = []

  
    for index, row in df.iterrows():

        
        payload = row.to_dict()

       
        payload.pop("Class", None)

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            
            result = response.json()
            

        
            predictions.append(result["Fraud"])
            probabilities.append(result["Fraud_probability"])
        except:
            predictions.append(-1)
            probabilities.append(-1)

    
    df["Prediction"] = predictions
    df["Probability"] = probabilities
    counts = df["Prediction"].value_counts().reset_index()
    bar_chart = px.bar(counts, x="Prediction", y="count", title="Fraud vs Legitimate")
    scatter_plt = px.scatter(df, x="Amount", y="Probability",color="Prediction", title="Fraud vs Legitimate")

    return df,bar_chart,scatter_plt
  

app = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[gr.Dataframe(label="Prediction Results"),gr.Plot(label="Prediction Barchart"),gr.Plot(label="Prediction Scatterplot")],
    title="Fraud Detection Batch Prediction"
)





app.launch()