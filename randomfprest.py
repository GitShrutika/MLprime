import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

titanic=sns.load_dataset("titanic")

features=["pclass","sex","embarked","age"]
target=["survived"]
#missing data
from sklearn.impute import SimpleImputer

imp_median=SimpleImputer(strategy="median")
titanic[["age"]]=imp_median.fit_transform(titanic[["age"]])

imp_freq=SimpleImputer(strategy="most_frequent")
titanic[["embarked"]]=imp_freq.fit_transform(titanic[["embarked"]])


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

titanic["sex"]=le.fit_transform(titanic["sex"])
titanic["embarked"]=le.fit_transform(titanic["embarked"])

X = titanic[features]
y = titanic["survived"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

#desion tree model
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=model.predict(X_test)

print("accuracy: ",accuracy_score(y_test,y_pred))

from sklearn.tree import plot_tree

plt.figure(figsize=(18,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Died","Survived"],
    filled=True
)

plt.tight_layout()
plt.show()

#random forest
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(
    n_estimators=201,
    oob_score=True
)

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print("oob score",rf.oob_score_)
print("testing accuracy",accuracy_score(y_test,y_pred))

#bagging classifier
from sklearn.ensemble import BaggingClassifier

base_model=DecisionTreeClassifier(max_depth=4)

bagging=BaggingClassifier(
    base_model,
    n_estimators=201
)

bagging.fit(X_train,y_train)

Y_pred=bagging.predict(X_test)
print("accuracy",accuracy_score(y_test,y_pred))

#LogisticReggression
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

base_model=LogisticRegression(max_iter=1000)

bagging=BaggingClassifier(
    base_model,
    n_estimators=201
)

bagging.fit(X_train,y_train)

Y_pred=bagging.predict(X_test)
print("accuracy",accuracy_score(y_test,y_pred))
