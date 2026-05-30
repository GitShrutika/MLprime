#standardization
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
X_train=Scaler.fit_transform(X_train)
X_test=Scaler.transform(X_test)
X_train
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("accuracy:",accuracy_score(y_test,y_pred)*100,"%")
print("precision:",precision_score(y_test,y_pred)*100,"%")
#evolution matrix
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy score:",accuracy_score(y_test,y_pred))
print("precision score:",precision_score(y_test,y_pred))
print("recall score:",recall_score(y_test,y_pred))
print("f1score:",f1_score(y_test,y_pred))
