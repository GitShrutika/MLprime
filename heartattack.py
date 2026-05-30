import pandas as pd 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score
heartdata=pd.read_csv("heart.csv")
heartdata.head()
heartdata.columns
heartdata.info()
heartdata["target"].unique()
X=heartdata.drop("target",axis=1)
y=heartdata["target"]
  y.head
  X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
  y_train[y_train==1]
y_train[y_train==0]
model=LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("accuracy:",accuracy_score(y_test,y_pred)*100,"%")
print("precision:",precision_score(y_test,y_pred)*100,"%")
y_pred
y_test
