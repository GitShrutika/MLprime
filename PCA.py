import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
iris=load_iris()

X=pd.DataFrame(iris.data)
y=iris.target
X.head()
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
#pca
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

X_pca=pca.fit_transform(X_scaled)

X_pca.shape
print("explaned ratio:",pca.explained_variance_ratio_)
print(pca.components_)
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y
)

plt.xlabel("pca1")
plt.ylabel("pca2")
plt.title("pca for iris")
