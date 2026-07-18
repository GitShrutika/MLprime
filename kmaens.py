import seaborn as sns
from sklearn.datasets import make_blobs
X,y=make_blobs(
    n_samples=1000,
    n_features=2,
    centers=4,
    random_state=42
    
)
#visulise the data
sns.scatterplot(x=X[:,0],y=X[:,1],c=y)
#Kmeans
from sklearn.cluster import KMeans
K=4

kmeans=KMeans(
    n_clusters=K,
    random_state=42
)

labels=kmeans.fit_predict(X)
cluster_no=labels
sns.scatterplot(x=X[:,0],y=X[:,1],c=labels)
#choose k value 1.elbow ,selwet score

wcss=[]
for k in range(1,21):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
sns.lineplot(x=range(1,21),y=wcss,marker='o')

#kneed module
!pip install kneed
from kneed import KneeLocator
knee=KneeLocator(range(1,21),wcss,curve="convex",direction="decreasing")
print("optimal K",knee.elbow)
from sklearn.metrics import silhouette_score 
ss=[]

for k in range(2,21):
    kmeans=KMeans(n_clusters=k)
    labels=kmeans.fit_predict(X)
    score=silhouette_score(X,labels)

    ss.append(score)
  #plot k selweet scoerew

sns.lineplot(x=range(2,21),y=ss,marker='o')


