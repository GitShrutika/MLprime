#plot
plt.plot(range(1,11),wcss,marker='o')
plt.xlabel("k")
plt.ylabel("wcss")
plt.title("graph")
#silwitscore
from sklearn.metrics import silhouette_score

scores=[]

for k in range(2,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    labels=kmeans.fit_predict(X_pca)
    score=silhouette_score(X_pca,labels)
    scores.append(score)

    #plot
plt.plot(range(2,11),scores,marker='o')
plt.xlabel("X")
plt.ylabel("sc")
#combined plot
k_range=range(2,11)

fig,ax1=plt.subplots(figsize=(10,8))

ax1.plot(k_range,wcss[:len(k_range)],marker="o",color="blue")
ax1.set_xlabel("k")
ax1.set_xlabel("wcss")

ax2=ax1.twinx()
ax2.plot(k_range,scores[:len(k_range)],marker="x",color="red",linestyle="--")
ax2.set_ylabel("sc")
#clustering
#kmeans

kmeans=KMeans(n_clusters=4,random_state=42)
labels_kmeans=kmeans.fit_predict(X_pca)
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],c=labels_kmeans)
#agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agg_clf=AgglomerativeClustering(n_clusters=4,linkage="ward")
labels_agg=agg_clf.fit_predict(X_pca)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],c=labels_agg)
#characterization of clusters

'''df_cleaned.drop("labels_agg",axis=1)'''
X["cluster"]= labels_agg
X.head()
pal=["red","blue","yellow","green"]
sns.countplot(x=df_cleaned["cluster"],palette=pal,hue=X["cluster"])
#income and spending patterns
sns.scatterplot(x=X["Total_Spending"],y=X["Income"],hue=df_cleaned["cluster"],palette=pal)
#cluster summary
cluster_summary=X.groupby("cluster").mean()
print(cluster_summary)
