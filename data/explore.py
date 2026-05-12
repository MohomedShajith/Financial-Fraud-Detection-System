import pandas as pd
data = pd.read_csv("data/creditcard.csv")

#print(data.shape)
#print(data.columns)
#print(data.head())
#print(data['Class'].value_counts())

print(data.isnull().sum().sum())
print(data.dtypes)