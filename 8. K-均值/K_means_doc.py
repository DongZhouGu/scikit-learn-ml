from __future__ import print_function
from time import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics


def main():
    print("loading documents ...")
    t = time()
    docs = load_files('data')
    print("summary: {0} documents in {1} categories.".format(
        len(docs.data), len(docs.target_names)))
    print("done in {0} seconds".format(time() - t))

    max_features = 20000
    print("vectorizing documents ...")
    t = time()
    vectorizer = TfidfVectorizer(max_df=0.4,
                                 min_df=2,
                                 max_features=max_features,
                                 encoding='latin-1')
    X = vectorizer.fit_transform((d for d in docs.data))
    print("n_samples: %d, n_features: %d" % X.shape)
    print("number of non-zero features in sample [{0}]: {1}".format(
        docs.filenames[0], X[0].getnnz()))
    print("done in {0} seconds".format(time() - t))

    print("clustering documents ...")
    t = time()
    n_clusters = 4
    kmean = KMeans(n_clusters=n_clusters,
                   max_iter=100,
                   tol=0.01,
                   verbose=1,
                   n_init=3)
    kmean.fit(X);
    print("kmean: k={}, cost={}".format(n_clusters, int(kmean.inertia_)))
    print("done in {0} seconds".format(time() - t))

    print("Top terms per cluster:")
    order_centroids = kmean.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(n_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

    # 算法性能评估指标
    labels = docs.target
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmean.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, kmean.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmean.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, kmean.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, kmean.labels_, sample_size=1000))

if __name__ == '__main__':
    main()
