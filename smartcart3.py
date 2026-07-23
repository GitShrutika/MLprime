#feature encoding
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
cat_cols=["Education","Living_with"]

enc_cols=ohe.fit_transform(df_cleaned[cat_cols])

enc_df=pd.DataFrame(enc_cols.toarray(),columns=ohe.get_feature_names_out(cat_cols),index=df_cleaned.index)

enc_df.head()
#scaling
from sklearn.preprocessing import StandardScaler
X=df_encoded
scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)
#visulise
from sklearn.decomposition import PCA
pca=PCA(n_components=3)

X_pca=pca.fit_transform(X_scaled)
#3d
'''fig=plt.figure(figsize=(8,6))

ax=fig.add_subplot(111, projection="3d")

ax.scatter(X_pca[:, 0],X_pca[:, 1],X_pca[:, 2])

ax.set_xlabel=pca1
ax.set_ylabel=pca2
ax.set_zlabel=pca3
ax.set_title=pca1'''
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3D PCA Scatter Plot")

plt.show()

#analyse k val
#elbow method
from sklearn.cluster import KMeans
from kneed import KneeLocator

wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)
  knee=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
optimal_k=knee.elbow
print("bestk",optimal_k)
