import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
heartdata=pd.read_csv("heart.csv")
X=heartdata.drop("target",axis=1)
y=heartdata["target"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
gnb_model=GaussianNB()
gnb_model.fit(X_train,y_train)
y_pred=gnb_model.predict(X_test)
#evaluation
print("recall_score:",recall_score(y_test,y_pred))
print("precision_score:",precision_score(y_test,y_pred))
print("accuracy_score:",accuracy_score(y_test,y_pred))
