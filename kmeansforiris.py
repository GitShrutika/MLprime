import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
iris=load_iris()
X=iris.data
y=iris.target
# visulise data
sns.scatterplot(x=X[:,0],y=X[:,2],c=y)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
#optimal dimentionality reduction using pca
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

pca_data=pca.fit_transform(X_scaled)
wcss=[]

for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

sns.lineplot(x=range(1,11),y=wcss,marker='o')
# apply k means
kmeans=KMeans(n_clusters=3)
labels=kmeans.fit_predict(X_scaled)

sns.scatterplot(x=X[:,0],y=X[:,2],c=y)

