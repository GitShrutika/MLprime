import numpy as np

class LinearRegression():

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape

        X_b = np.c_[np.ones((m, 1)), X]

        theta = np.dot(
            np.linalg.inv(np.dot(X_b.T, X_b)),
            np.dot(X_b.T, y)
        )

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        y_pred = self.bias + np.dot(X, self.weights)
        return y_pred


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(y_pred)


'''import numpy as np
class LinearRegression():
    def __init__(self):
        self.weights= None
        self.bias= None 

def fit(self,X,y):
    m,n=X.shape

    X_b=np.c_[np.ones(m,1), X]

    theta=np.dot(np.linalg.inv(np.dot(X_b.T,X),np.dot(X_b.T,y)))
    
    self.bias=theta[0]
    self.weights=theta[1:]

def predict(self,X):
    y_pred=self.bias + np.dot(X,self.weights)
    return y_pred    X=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,6,8,10])
model=LinearRegression()
model.fit(X, y)

y_pred=model.pred(X)
print(y_pred)'''
