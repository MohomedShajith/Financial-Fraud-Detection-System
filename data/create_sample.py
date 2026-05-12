import pandas as pd

data = pd.read_csv(r'data\creditcard.csv')
data_sample = pd.concat([data[data['Class']==0].head(),data[data['Class']==1].head()])
data_sample = data_sample.drop(columns=["Class"], errors="ignore")

data_sample.to_csv('test_sample.csv',index=False)