#staking
from sklearn.ensemble import StackingClassifier

meta_model=LogisticRegression()

sta_clf=StackingClassifier(
    estimators=[
        ("lr",lr),
        ("svc",svc),
        ("dtc",dtc)
    ],
    final_estimator=meta_model,
    cv=5
)

sta_clf.fit(X_train,y_train)

y_pred=sta_clf.predict(X_test)

print("accuracy score",accuracy_score(Y_test,y_pred))
