import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.bias = None
        self.weight = None
        self.n_iter = n_iter
        self.lr = learning_rate

    def fit(self, X, y):
        m, n = X.shape

        # Step 1: Initialize parameters
        self.bias = 0
        self.weight = np.zeros(n)

        # Step 2: Gradient Descent
        for i in range(self.n_iter):
            y_pred = self.bias + np.dot(X, self.weight)

            # Step 3: Calculate gradients
            db = (1 / m) * np.sum(y_pred - y)
            dw = (1 / m) * np.dot(X.T, (y_pred - y))

            # Step 4: Update parameters
            self.bias = self.bias - self.lr * db
            self.weight = self.weight - self.lr * dw

    def pred(self, X):
        y_pred = self.bias + np.dot(X, self.weight)
        return y_pred
X=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,6,8,10])
model=LinearRegression()
model.fit(X, y)

y_pred=model.pred(X)
print(y_pred)
