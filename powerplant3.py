import pandas as pd 
import numpy as np
df=pd.read_csv("powerplant_data.csv")
df.head()
df.isnull().sum()
X=df.drop("PE",axis=1)
y=df["PE"]
#split are data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
X_test
df.shape
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
X_train_scaled
import torch
import torch.nn as nn

X_train_tensor=torch.tensor(X_train_scaled,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)

X_test_tensor=torch.tensor(X_test_scaled,dtype=torch.float32)
y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32).view(-1,1)

type(X_train_scaled)
type(y_train)
from torch.utils.data import TensorDataset,DataLoader
train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
test_dataset=TensorDataset(X_test_tensor,y_test_tensor)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
#starting deep learning
import torch
import torch.nn as nn
import torch.optim as optim

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model=nn.Sequential(
               nn.Linear(X_train.shape[1],6),
               nn.ReLU(),
               nn.Linear(6,6),
               nn.ReLU(),
               nn.Linear(6,1)
       )

    def forward(self,x):
            return self.model(x)

