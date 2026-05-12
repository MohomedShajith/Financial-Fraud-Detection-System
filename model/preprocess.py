import os
import pandas as pd
import joblib
from dotenv import load_dotenv
from sqlalchemy import create_engine
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

load_dotenv()

def get_data():

    url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(url)

    query = "select * from transactions;"
    data = pd.read_sql(query, engine)
    #print(data.head())
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

    sc = StandardScaler()
    X_train[['Time','Amount']] = sc.fit_transform(X_train[['Time','Amount']])
    X_test[['Time','Amount']] = sc.transform(X_test[['Time','Amount']])
    
    joblib.dump(sc, "model/scaler.pkl")
    sm = SMOTE(random_state = 42)
   
    X_train,y_train = sm.fit_resample(X_train,y_train)
    joblib.dump(X_train,"model/X_train.pkl")
    joblib.dump(X_test,"model/X_test.pkl")
    joblib.dump(y_train,"model/y_train.pkl")
    joblib.dump(y_test,"model/y_test.pkl")
    return X_train,X_test,y_train,y_test
    