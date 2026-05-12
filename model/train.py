import torch
import joblib
import pandas as pd
from torch import nn
from pathlib import Path 
from sklearn.metrics import confusion_matrix,classification_report

class Fraud_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=30,out_features=60)
        self.layer2 = nn.Linear(in_features=60,out_features=60)
        self.layer3 = nn.Linear(in_features=60,out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self,x):
        return self.sigmoid(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))))
    
if __name__ == "__main__":
    torch.manual_seed(42)
    model = Fraud_model()
    model.state_dict()
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(params= model.parameters(),lr=0.001)

    X_train = joblib.load('model/X_train.pkl')
    X_test = joblib.load('model/X_test.pkl')
    y_train = joblib.load('model/y_train.pkl')
    y_test = joblib.load('model/y_test.pkl')

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)


    epochs = 2000

    for epoch in range(epochs):
        model.train()
        train_logits = model(X_train)
        train_preds = torch.round(train_logits)
        loss = loss_fn(train_logits,y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()

        model.eval()
        with torch.inference_mode():  
            test_logits = model(X_test)
            test_preds = torch.round(test_logits)
            test_loss = loss_fn(test_logits,y_test)

        if epoch % 100 == 0 :
            print(f"Epoch :{epoch},Train_Loss{loss:.2f},Test_Loss{test_loss :.2f}")

    model.eval()
    with torch.no_grad():
        pred_logits = model(X_test)
        y_preds = torch.round(pred_logits)
        y_loss = loss_fn(pred_logits,y_test)

    y_preds = y_preds.detach().numpy()
    y_test = y_test.detach().numpy()


    cf = confusion_matrix(y_test,y_preds)
    cl_report = classification_report(y_test,y_preds)

    print(cf)
    print(cl_report)


    model_path = Path('Models')
    model_path.mkdir(parents = True,exist_ok= True)
    model_name = "fraud prediction neural network model.pth"
    model_save_path = model_path/model_name
    torch.save(obj = model.state_dict(),f=model_save_path)
