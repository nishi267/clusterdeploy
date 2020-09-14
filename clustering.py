import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from sklearn.cluster import KMeans

df = pd.read_csv("testcasesrepo.csv", encoding='unicode_escape')
df1 = df.dropna(axis=0, how='any')
df1['combine6'] = df1.iloc[:, 1] + df1.iloc[:, 2] + df1.iloc[:, 3]
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
vec.fit(df1.combine6.values)
features = vec.transform(df1.combine6.values)

clust = KMeans(init='k-means++', n_clusters=3, n_init=10)
clust.fit(features)
df1['cluster_labels'] = clust.labels_
df1.to_csv("test_cluster1.csv")
pickle_out = open("cluster.pkl", "wb")
pickle.dump(clust, pickle_out)
pickle_out.close()
