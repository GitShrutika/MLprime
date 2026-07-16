from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score , classification_report

X,y=make_classification(
    n_samples=500,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

lr=LogisticRegression()
svc=SVC()
dtc=DecisionTreeClassifier(max_depth=3)

voting_clf=VotingClassifier(
    estimators=[
        ("lr",lr),
        ("svc",svc),
        ("dtc",dtc)
    ]
)

voting_clf.fit(X_train,y_train)

y_pred=voting_clf.predict(X_test)

print("accuracy",accuracy_score(y_pred,y_test))
print("cr",classification_report(y_pred,y_test))

from sklearn.datasets import make_regression

X,y=make_regression(
    n_samples=500,
    n_features=20,
    n_informative=5,
    random_state=42
)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


from sklearn.ensemble import VotingRegressor
lin_reg=LinearRegression()
svr=SVR()
dtr=DecisionTreeRegressor(max_depth=3)

vr=VotingRegressor(
    estimators=[
        ("lregg",lin_reg),
        ("svr",svr),
        ("dtr",dtr)
    ]
)

vr.fit(X_train,y_train)

y_pred=vr.predict(X_test)
from sklearn.metrics import r2_score
print("r2",r2_score(y_test,y_pred))
