from sklearn.cluster import KMeans
from sympy.polys.polyoptions import Series
__author__ = 'isaac waweru'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
if __name__ == "__main__":
    df = pd.read_csv("roofing_materials_by_county.csv", index_col=0)
    df = df.fillna(0)
    df1 = df.as_matrix()
    # features = list(df.columns[10:20])
    features = list(df.columns[1:5])
    print features
    x = df[features]
    # x = x[features].fillna(0, inplace=True)
    x = x.apply(lambda x: x.str.replace('[%]', '').astype(float))
    x = x.fillna(0)

    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(x)
        distortions.append(km.inertia_)
    plt.plot(range(1,11), distortions, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")
    plt.show()
    x = x.as_matrix()
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=111)
    kmeans.fit(x)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        # select only data observations with cluster label == i
        ds = x[np.where(labels==i)]
        # plot the data observations
        plt.plot(ds[:,0],ds[:,1],'o')
        # plot the centroids
        lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
        # make the centroid x's bigger
        plt.setp(lines,ms=15.0)
        plt.setp(lines,mew=2.0)
    plt.title("customer segments clusters")
    plt.show()
    print labels
    se = pd.Series(labels)
    df['class'] = se.values
    print df