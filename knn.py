import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
heartdata=pd.read_csv("heart.csv")
X=heartdata.drop("target",axis=1)
y=heartdata["target"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
X.head()
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
X_train_scaled
knn_classifier=KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled,y_train)
y_pred=knn_classifier.predict(X_test_scaled)

#evaluation
print("recall_score:",recall_score(y_test,y_pred))
print("precision_score:",precision_score(y_test,y_pred))
print("accuracy_score:",accuracy_score(y_test,y_pred))
