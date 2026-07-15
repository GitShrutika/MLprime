from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
X,y=make_regression(
    n_samples=1000,
    n_features=10,
    noise=20,
    random_state=42
)

X_train,X_test,y_train,y_test=train_test_split(

    X,y,test_size=0.2,random_state=42
)
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=3,
    max_depth=42,
    subsample=0.8
)
gbr.fit(X_train,y_train)
y_pred=gbr.predict(X_test)

print("r2",r2_score(y_test,y_pred))
#use classifier
from sklearn.datasets import make_classification
X,y=make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    random_state=42
)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)
from sklearn.ensemble import GradientBoostingClassifier
gbc.fit(X_train,y_train)



gbc=GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=150,
    max_depth=3,
    random_state=42
    
)
from sklearn.metrics import accuracy_score

y_pred=gbc.predict(X_test)


print("accuracy",accuracy_score(y_test,y_pred))
