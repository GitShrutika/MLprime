from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

  df=datasets.load_iris(as_frame=True).frame
  df.isnull().sum()
  X=df.drop("target",axis=1)
y=df["target"]
  X.head()
  X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42,stratify=y
)
  X_train.head()
  from sklearn .preprocessing import StandardScaler

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
    #model
svc=SVC()
svc.fit(X_train_scaled,y_train)
    y_pred=svc.predict(X_test_scaled)
    from sklearn.metrics import accuracy_score, classification_report

print("accuracy:",accuracy_score(y_test,y_pred))
print("classification_report:",classification_report(y_test,y_pred))
      # linear karneal
svc=SVC(kernel="linear")
svc.fit(X_train_scaled,y_train)

y_pred=svc.predict(X_test_scaled)
print("accuracy:",accuracy_score(y_test,y_pred))
      # polynomial kernal
svc=SVC(kernel="poly")
svc.fit(X_train_scaled,y_train)

y_pred=svc.predict(X_test_scaled)
print("accuracy:",accuracy_score(y_test,y_pred))
      
#sigmoid kernal
svc=SVC(kernel="sigmoid")
svc.fit(X_train_scaled,y_train)

y_pred=svc.predict(X_test_scaled)
print("accuracy:",accuracy_score(y_test,y_pred))
      C_vals=[1,2,3,4,5]

for c_val in C_vals:
    svc=SVC(C=c_val,kernel="rbf")
    svc.fit(X_train_scaled,y_train)

    y_pred=svc.predict(X_test_scaled)
    print("C=",c_val,"accuracy:",accuracy_score(y_test,y_pred))
